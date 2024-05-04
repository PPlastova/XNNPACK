// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>

#include <riscv_vector.h>

#include <xnnpack/common.h>
#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/math.h>
#include <xnnpack/raddstoreexpminusmax.h>
#include <xnnpack/vbinary.h>
#include <xnnpack/vunary.h>
#include <xnnpack/dwconv.h>
#include <xnnpack/gemm.h>
#include <xnnpack/igemm.h>


static inline vfloat32m4_t eval_poly_horner(vfloat32m4_t x,
                                                  float c6, float c5,
                                                  float c4, float c3, float c2,
                                                  float c1, float c0, size_t vl) {
  vfloat32m4_t z;
  vfloat32m4_t y = vfmv_v_f_f32m4(c5, vl);
  y = vfmacc_vf_f32m4(y, c6, x, vl);

  z = vfmv_v_f_f32m4(c4, vl);
  y = vfmadd_vv_f32m4(y, x, z, vl);

  z = vfmv_v_f_f32m4(c3, vl);
  y = vfmadd_vv_f32m4(y, x, z, vl);

  z = vfmv_v_f_f32m4(c2, vl);
  y = vfmadd_vv_f32m4(y, x, z, vl);

  z = vfmv_v_f_f32m4(c1, vl);
  y = vfmadd_vv_f32m4(y, x, z, vl);

  z = vfmv_v_f_f32m4(c0, vl);
  y = vfmadd_vv_f32m4(y, x, z, vl);
  return y;
}

/// @brief Computes the exponential function on vector of float32 values with a
/// 1-ULP error bound in the range [-87, 0]. Smaller inputs are flushed to
/// exp(-0x1.5d589ep6f) ~= 0x1.6a0a64p-127f while the result is undefined for
/// inputs greater than zero as well as NaNs.
///
/// This function is intended for use in computing softmax, whose inputs are
/// pre-normalized by subtracting the maximum, resulting in inputs in (-inf, 0).
/// One of these inputs will contribute exp(0) = 1 to the final sum, so any
/// inputs flushed upwards to -0x1.5d589ep6f and thus contributing at most
/// 0x1.6a0a64p-127f to the total, will not result of softmax unless at least
/// ~2^100 of them are summed in ascending order.
///
/// Exploitation of these properties results in a faster exponential by avoiding
/// the need to handle edge cases that arise from very large or small exponents.
///
/// @param[in] x Input vector of float32 values
/// @param[in] vl Length of vector x
/// @return Result of applying softexp() to elements of x
static inline vfloat32m4_t softexp_f32m4(
    vfloat32m4_t x, size_t vl,
    const union xnn_f32_expminus_params params[restrict XNN_MIN_ELEMENTS(1)]) {
  // Ensure that q = RN(x/log(2)) >= e_min, so that 2^q can be computed safely
  // with a simple shift into the exponent field.
  // xmin = round(-126.5 * log(2), single, RU) ~ -87.68311309814453125

  const float xmin = params->rvv_rr2_p6.x_min;
  const float r_ln2f = params->rvv_rr2_p6.log2e;
  const float l2uf = params->rvv_rr2_p6.ln2_hi;
  const float l2lf = params->rvv_rr2_p6.ln2_lo;
  const float c6 = params->rvv_rr2_p6.c6;
  const float c5 = params->rvv_rr2_p6.c5;
  const float c4 = params->rvv_rr2_p6.c4;
  const float c3 = params->rvv_rr2_p6.c3;
  const float c2 = params->rvv_rr2_p6.c2;

  // const float xmin = -0x1.5ebb82p6;
  x = vfmax_vf_f32m4(x, xmin, vl);

  // 0. Reduction x = s * q ln(2)
  // const float r_ln2f = 0x1.715476p0f;  // single(1/log(2));
  // const float l2uf = 0x1.62e4p-1f;     // round(log(2), 24-8, RN);
  // const float l2lf = 0x1.7f7d1cp-20f;  // round(log(2) - l2uf, single, RN);
  vfloat32m4_t v = vfmul_vf_f32m4(x, r_ln2f, vl);

  vint16m2_t q = vfncvt_x_f_w_i16m2(v, vl);
  vfloat32m4_t z = vfwcvt_f_x_v_f32m4(q, vl);

  // Use Cody-Waite range reduction method (note two constants to represent log(2)) to improve accuracy.
  vfloat32m4_t s = vfnmsac_vf_f32m4(x, l2uf, z, vl);
  s = vfnmsac_vf_f32m4(s, l2lf, z, vl);

  // 1. Approximate e^s by degree-6 polynomial approximation
  vfloat32m4_t u = eval_poly_horner(s, c6, c5, c4, c3, c2, 1.0f, 1.0f, vl);

  // 2. Reconstruction: compute u = u*2^q
  const int16_t p = (24 - 1);
  const int16_t bias = (128 - 1);
  vint32m4_t qw = vwadd_vx_i32m4(q, bias, vl);
  vint32m4_t qq = vsll_vx_i32m4(qw, p, vl);
  vfloat32m4_t qf = vreinterpret_v_i32m4_f32m4(qq);
  u = vfmul_vv_f32m4(u, qf, vl);
  return u;
}

void xnn_f32_vadd_minmax_ukernel__rvv_u8v(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float output_min = params->scalar.min;
  const float output_max = params->scalar.max;
  size_t n = batch >> 2;

  do {
    size_t vl = vsetvl_e32m8(n);
    n -= vl;
    vfloat32m8_t va = vle32_v_f32m8(input_a, vl);
    input_a += vl;
    vfloat32m8_t vb = vle32_v_f32m8(input_b, vl);
    input_b += vl;
    vfloat32m8_t vacc = vfadd_vv_f32m8(va, vb, vl);
    vacc = vfmax_vf_f32m8(vacc, output_min, vl);
    vacc = vfmin_vf_f32m8(vacc, output_max, vl);
    vse32_v_f32m8(output, vacc, vl);
    output += vl;
  } while (n > 0);
}

void xnn_f32_vaddc_minmax_ukernel__rvv_u8v(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float output_min = params->scalar.min;
  const float output_max = params->scalar.max;
  const float b = *input_b;
  size_t n = batch >> 2;

  do {
    size_t vl = vsetvl_e32m8(n);
    n -= vl;
    vfloat32m8_t va = vle32_v_f32m8(input_a, vl);
    input_a += vl;
    vfloat32m8_t vacc = vfadd_vf_f32m8(va, b, vl);
    vacc = vfmax_vf_f32m8(vacc, output_min, vl);
    vacc = vfmin_vf_f32m8(vacc, output_max, vl);
    vse32_v_f32m8(output, vacc, vl);
    output += vl;
  } while (n > 0);
}

void xnn_f32_vdiv_minmax_ukernel__rvv_u8v(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float output_min = params->scalar.min;
  const float output_max = params->scalar.max;
  size_t n = batch >> 2;

  do {
    size_t vl = vsetvl_e32m8(n);
    n -= vl;
    vfloat32m8_t va = vle32_v_f32m8(input_a, vl);
    input_a += vl;
    vfloat32m8_t vb = vle32_v_f32m8(input_b, vl);
    input_b += vl;
    vfloat32m8_t vacc = vfdiv_vv_f32m8(va, vb, vl);
    vacc = vfmax_vf_f32m8(vacc, output_min, vl);
    vacc = vfmin_vf_f32m8(vacc, output_max, vl);
    vse32_v_f32m8(output, vacc, vl);
    output += vl;
  } while (n > 0);
}

void xnn_f32_vdivc_minmax_ukernel__rvv_u8v(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float output_min = params->scalar.min;
  const float output_max = params->scalar.max;
  const float b = *input_b;
  size_t n = batch >> 2;

  do {
    size_t vl = vsetvl_e32m8(n);
    n -= vl;
    vfloat32m8_t va = vle32_v_f32m8(input_a, vl);
    input_a += vl;
    vfloat32m8_t vacc = vfdiv_vf_f32m8(va, b, vl);
    vacc = vfmax_vf_f32m8(vacc, output_min, vl);
    vacc = vfmin_vf_f32m8(vacc, output_max, vl);
    vse32_v_f32m8(output, vacc, vl);
    output += vl;
  } while (n > 0);
}

void xnn_f32_vmax_ukernel__rvv_u8v(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  size_t n = batch >> 2;

  do {
    size_t vl = vsetvl_e32m8(n);
    n -= vl;
    vfloat32m8_t va = vle32_v_f32m8(input_a, vl);
    input_a += vl;
    vfloat32m8_t vb = vle32_v_f32m8(input_b, vl);
    input_b += vl;
    vfloat32m8_t vacc = vfmax_vv_f32m8(va, vb, vl);
    vse32_v_f32m8(output, vacc, vl);
    output += vl;
  } while (n > 0);
}

void xnn_f32_vmaxc_ukernel__rvv_u8v(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float b = *input_b;
  size_t n = batch >> 2;

  do {
    size_t vl = vsetvl_e32m8(n);
    n -= vl;
    vfloat32m8_t va = vle32_v_f32m8(input_a, vl);
    input_a += vl;
    vfloat32m8_t vacc = vfmax_vf_f32m8(va, b, vl);
    vse32_v_f32m8(output, vacc, vl);
    output += vl;
  } while (n > 0);
}

void xnn_f32_vmin_ukernel__rvv_u8v(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  size_t n = batch >> 2;

  do {
    size_t vl = vsetvl_e32m8(n);
    n -= vl;
    vfloat32m8_t va = vle32_v_f32m8(input_a, vl);
    input_a += vl;
    vfloat32m8_t vb = vle32_v_f32m8(input_b, vl);
    input_b += vl;
    vfloat32m8_t vacc = vfmin_vv_f32m8(va, vb, vl);
    vse32_v_f32m8(output, vacc, vl);
    output += vl;
  } while (n > 0);
}

void xnn_f32_vminc_ukernel__rvv_u8v(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float b = *input_b;
  size_t n = batch >> 2;

  do {
    size_t vl = vsetvl_e32m8(n);
    n -= vl;
    vfloat32m8_t va = vle32_v_f32m8(input_a, vl);
    input_a += vl;
    vfloat32m8_t vacc = vfmin_vf_f32m8(va, b, vl);
    vse32_v_f32m8(output, vacc, vl);
    output += vl;
  } while (n > 0);
}

void xnn_f32_vmul_minmax_ukernel__rvv_u8v(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float output_min = params->scalar.min;
  const float output_max = params->scalar.max;
  size_t n = batch >> 2;

  do {
    size_t vl = vsetvl_e32m8(n);
    n -= vl;
    vfloat32m8_t va = vle32_v_f32m8(input_a, vl);
    input_a += vl;
    vfloat32m8_t vb = vle32_v_f32m8(input_b, vl);
    input_b += vl;
    vfloat32m8_t vacc = vfmul_vv_f32m8(va, vb, vl);
    vacc = vfmax_vf_f32m8(vacc, output_min, vl);
    vacc = vfmin_vf_f32m8(vacc, output_max, vl);
    vse32_v_f32m8(output, vacc, vl);
    output += vl;
  } while (n > 0);
}

void xnn_f32_vmulc_minmax_ukernel__rvv_u8v(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float output_min = params->scalar.min;
  const float output_max = params->scalar.max;
  const float b = *input_b;
  size_t n = batch >> 2;

  do {
    size_t vl = vsetvl_e32m8(n);
    n -= vl;
    vfloat32m8_t va = vle32_v_f32m8(input_a, vl);
    input_a += vl;
    vfloat32m8_t vacc = vfmul_vf_f32m8(va, b, vl);
    vacc = vfmax_vf_f32m8(vacc, output_min, vl);
    vacc = vfmin_vf_f32m8(vacc, output_max, vl);
    vse32_v_f32m8(output, vacc, vl);
    output += vl;
  } while (n > 0);
}

void xnn_f32_vrdivc_minmax_ukernel__rvv_u8v(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float output_min = params->scalar.min;
  const float output_max = params->scalar.max;
  const float b = *input_b;
  size_t n = batch >> 2;

  do {
    size_t vl = vsetvl_e32m8(n);
    n -= vl;
    vfloat32m8_t va = vle32_v_f32m8(input_a, vl);
    input_a += vl;
    vfloat32m8_t vacc = vfrdiv_vf_f32m8(va, b, vl);
    vacc = vfmax_vf_f32m8(vacc, output_min, vl);
    vacc = vfmin_vf_f32m8(vacc, output_max, vl);
    vse32_v_f32m8(output, vacc, vl);
    output += vl;
  } while (n > 0);
}

void xnn_f32_vrsubc_minmax_ukernel__rvv_u8v(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float output_min = params->scalar.min;
  const float output_max = params->scalar.max;
  const float b = *input_b;
  size_t n = batch >> 2;

  do {
    size_t vl = vsetvl_e32m8(n);
    n -= vl;
    vfloat32m8_t va = vle32_v_f32m8(input_a, vl);
    input_a += vl;
    vfloat32m8_t vacc = vfrsub_vf_f32m8(va, b, vl);
    vacc = vfmax_vf_f32m8(vacc, output_min, vl);
    vacc = vfmin_vf_f32m8(vacc, output_max, vl);
    vse32_v_f32m8(output, vacc, vl);
    output += vl;
  } while (n > 0);
}

void xnn_f32_vsqrdiff_ukernel__rvv_u8v(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  size_t n = batch >> 2;

  do {
    size_t vl = vsetvl_e32m8(n);
    n -= vl;
    vfloat32m8_t va = vle32_v_f32m8(input_a, vl);
    input_a += vl;
    vfloat32m8_t vb = vle32_v_f32m8(input_b, vl);
    input_b += vl;
    vfloat32m8_t vacc = vfsub_vv_f32m8(va, vb, vl);
    vacc = vfmul_vv_f32m8(vacc, vacc, vl);
    vse32_v_f32m8(output, vacc, vl);
    output += vl;
  } while (n > 0);
}

void xnn_f32_vsqrdiffc_ukernel__rvv_u8v(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float b = *input_b;
  size_t n = batch >> 2;

  do {
    size_t vl = vsetvl_e32m8(n);
    n -= vl;
    vfloat32m8_t va = vle32_v_f32m8(input_a, vl);
    input_a += vl;
    vfloat32m8_t vacc = vfsub_vf_f32m8(va, b, vl);
    vacc = vfmul_vv_f32m8(vacc, vacc, vl);
    vse32_v_f32m8(output, vacc, vl);
    output += vl;
  } while (n > 0);
}

void xnn_f32_vsub_minmax_ukernel__rvv_u8v(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float output_min = params->scalar.min;
  const float output_max = params->scalar.max;
  size_t n = batch >> 2;

  do {
    size_t vl = vsetvl_e32m8(n);
    n -= vl;
    vfloat32m8_t va = vle32_v_f32m8(input_a, vl);
    input_a += vl;
    vfloat32m8_t vb = vle32_v_f32m8(input_b, vl);
    input_b += vl;
    vfloat32m8_t vacc = vfsub_vv_f32m8(va, vb, vl);
    vacc = vfmax_vf_f32m8(vacc, output_min, vl);
    vacc = vfmin_vf_f32m8(vacc, output_max, vl);
    vse32_v_f32m8(output, vacc, vl);
    output += vl;
  } while (n > 0);
}

void xnn_f32_vsubc_minmax_ukernel__rvv_u8v(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float output_min = params->scalar.min;
  const float output_max = params->scalar.max;
  const float b = *input_b;
  size_t n = batch >> 2;

  do {
    size_t vl = vsetvl_e32m8(n);
    n -= vl;
    vfloat32m8_t va = vle32_v_f32m8(input_a, vl);
    input_a += vl;
    vfloat32m8_t vacc = vfsub_vf_f32m8(va, b, vl);
    vacc = vfmax_vf_f32m8(vacc, output_min, vl);
    vacc = vfmin_vf_f32m8(vacc, output_max, vl);
    vse32_v_f32m8(output, vacc, vl);
    output += vl;
  } while (n > 0);
}

void xnn_qs8_vmul_minmax_fp32_ukernel__rvv_u2v(
    size_t batch,
    const int8_t* input_a,
    const int8_t* input_b,
    int8_t* output,
    const union xnn_qs8_mul_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const int32_t a_zero_point = params->fp32_scalar.a_zero_point;
  const int32_t b_zero_point = params->fp32_scalar.b_zero_point;
  const float scale = params->fp32_scalar.scale;
  const float output_min_less_zero_point = params->fp32_scalar.output_min_less_zero_point;
  const float output_max_less_zero_point = params->fp32_scalar.output_max_less_zero_point;
  const float magic_bias = params->fp32_scalar.magic_bias;
  const int32_t magic_bias_less_output_zero_point = params->fp32_scalar.magic_bias_less_output_zero_point;

  do {
    const size_t n = vsetvl_e8m2(batch);

    vint8m2_t in_a_i8v = vle8_v_i8m2(input_a, n); input_a += n;
    vint8m2_t in_b_i8v = vle8_v_i8m2(input_b, n); input_b += n;
    vint16m4_t a_i16v = vwsub_vx_i16m4(in_a_i8v, a_zero_point, n);
    vint16m4_t b_i16v = vwsub_vx_i16m4(in_b_i8v, b_zero_point, n);

    vint32m8_t acc_i32v = vwmul_vv_i32m8(a_i16v, b_i16v, n);
    vfloat32m8_t acc_f32v = vfcvt_f_x_v_f32m8(acc_i32v, n);
    acc_f32v = vfmul_vf_f32m8(acc_f32v, scale, n);
    acc_f32v = vfmin_vf_f32m8(vfmax_vf_f32m8(acc_f32v, output_min_less_zero_point, n), output_max_less_zero_point, n);
    acc_f32v = vfadd_vf_f32m8(acc_f32v, magic_bias, n);

    vint32m8_t out_i32v = vfcvt_x_f_v_i32m8(acc_f32v, n);
    out_i32v = vsub_vx_i32m8(out_i32v, magic_bias_less_output_zero_point, n);
    vint16m4_t out_i16v = vncvt_x_x_w_i16m4(out_i32v, n);
    vint8m2_t out_i8v = vncvt_x_x_w_i8m2(out_i16v, n);
    vse8_v_i8m2(output, out_i8v, n); output += n;

    batch -= n;
  } while (batch != 0);
}

void xnn_qs8_vmulc_minmax_fp32_ukernel__rvv_u2v(
    size_t batch,
    const int8_t* input_a,
    const int8_t* input_b,
    int8_t* output,
    const union xnn_qs8_mul_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const int32_t a_zero_point = params->fp32_scalar.a_zero_point;
  const float scale = params->fp32_scalar.scale;
  const float output_min_less_zero_point = params->fp32_scalar.output_min_less_zero_point;
  const float output_max_less_zero_point = params->fp32_scalar.output_max_less_zero_point;
  const float magic_bias = params->fp32_scalar.magic_bias;
  const int32_t magic_bias_less_output_zero_point = params->fp32_scalar.magic_bias_less_output_zero_point;
  const int32_t vb = (int32_t) *input_b - params->fp32_scalar.b_zero_point;

  do {
    const size_t n = vsetvl_e8m2(batch);

    vint8m2_t in_a_i8v = vle8_v_i8m2(input_a, n); input_a += n;
    vint16m4_t a_i16v = vwsub_vx_i16m4(in_a_i8v, a_zero_point, n);

    vint32m8_t acc_i32v = vwmul_vx_i32m8(a_i16v, vb, n);
    vfloat32m8_t acc_f32v = vfcvt_f_x_v_f32m8(acc_i32v, n);
    acc_f32v = vfmul_vf_f32m8(acc_f32v, scale, n);
    acc_f32v = vfmin_vf_f32m8(vfmax_vf_f32m8(acc_f32v, output_min_less_zero_point, n), output_max_less_zero_point, n);
    acc_f32v = vfadd_vf_f32m8(acc_f32v, magic_bias, n);

    vint32m8_t out_i32v = vfcvt_x_f_v_i32m8(acc_f32v, n);
    out_i32v = vsub_vx_i32m8(out_i32v, magic_bias_less_output_zero_point, n);
    vint16m4_t out_i16v = vncvt_x_x_w_i16m4(out_i32v, n);
    vint8m2_t out_i8v = vncvt_x_x_w_i8m2(out_i16v, n);
    vse8_v_i8m2(output, out_i8v, n); output += n;

    batch -= n;
  } while (batch != 0);
}

void xnn_qu8_vmul_minmax_fp32_ukernel__rvv_u2v(
    size_t batch,
    const uint8_t* input_a,
    const uint8_t* input_b,
    uint8_t* output,
    const union xnn_qu8_mul_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const int32_t a_zero_point = params->fp32_scalar.a_zero_point;
  const int32_t b_zero_point = params->fp32_scalar.b_zero_point;
  const float scale = params->fp32_scalar.scale;
  const float output_min_less_zero_point = params->fp32_scalar.output_min_less_zero_point;
  const float output_max_less_zero_point = params->fp32_scalar.output_max_less_zero_point;
  const float magic_bias = params->fp32_scalar.magic_bias;
  const int32_t magic_bias_less_output_zero_point = params->fp32_scalar.magic_bias_less_output_zero_point;

  do {
    const size_t n = vsetvl_e8m2(batch);

    vuint8m2_t in_a_u8v = vle8_v_u8m2(input_a, n); input_a += n;
    vuint8m2_t in_b_u8v = vle8_v_u8m2(input_b, n); input_b += n;
    vuint16m4_t a_u16v = vwsubu_vx_u16m4(in_a_u8v, a_zero_point, n);
    vuint16m4_t b_u16v = vwsubu_vx_u16m4(in_b_u8v, b_zero_point, n);
    vint16m4_t a_i16v = vreinterpret_v_u16m4_i16m4(a_u16v);
    vint16m4_t b_i16v = vreinterpret_v_u16m4_i16m4(b_u16v);

    vint32m8_t acc_i32v = vwmul_vv_i32m8(a_i16v, b_i16v, n);
    vfloat32m8_t acc_f32v = vfcvt_f_x_v_f32m8(acc_i32v, n);
    acc_f32v = vfmul_vf_f32m8(acc_f32v, scale, n);
    acc_f32v = vfmin_vf_f32m8(vfmax_vf_f32m8(acc_f32v, output_min_less_zero_point, n), output_max_less_zero_point, n);
    acc_f32v = vfadd_vf_f32m8(acc_f32v, magic_bias, n);

    vuint32m8_t out_u32v = vfcvt_xu_f_v_u32m8(acc_f32v, n);
    out_u32v = vsub_vx_u32m8(out_u32v, magic_bias_less_output_zero_point, n);
    vuint16m4_t out_u16v = vncvt_x_x_w_u16m4(out_u32v, n);
    vuint8m2_t out_u8v = vncvt_x_x_w_u8m2(out_u16v, n);
    vse8_v_u8m2(output, out_u8v, n); output += n;

    batch -= n;
  } while (batch != 0);
}

void xnn_qu8_vmulc_minmax_fp32_ukernel__rvv_u2v(
    size_t batch,
    const uint8_t* input_a,
    const uint8_t* input_b,
    uint8_t* output,
    const union xnn_qu8_mul_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const int32_t a_zero_point = params->fp32_scalar.a_zero_point;
  const float scale = params->fp32_scalar.scale;
  const float output_min_less_zero_point = params->fp32_scalar.output_min_less_zero_point;
  const float output_max_less_zero_point = params->fp32_scalar.output_max_less_zero_point;
  const float magic_bias = params->fp32_scalar.magic_bias;
  const int32_t magic_bias_less_output_zero_point = params->fp32_scalar.magic_bias_less_output_zero_point;
  const int32_t vb = (int32_t) *input_b - params->fp32_scalar.b_zero_point;

  do {
    const size_t n = vsetvl_e8m2(batch);

    vuint8m2_t in_a_u8v = vle8_v_u8m2(input_a, n); input_a += n;
    vuint16m4_t a_u16v = vwsubu_vx_u16m4(in_a_u8v, a_zero_point, n);
    vint16m4_t a_i16v = vreinterpret_v_u16m4_i16m4(a_u16v);

    vint32m8_t acc_i32v = vwmul_vx_i32m8(a_i16v, vb, n);
    vfloat32m8_t acc_f32v = vfcvt_f_x_v_f32m8(acc_i32v, n);
    acc_f32v = vfmul_vf_f32m8(acc_f32v, scale, n);
    acc_f32v = vfmin_vf_f32m8(vfmax_vf_f32m8(acc_f32v, output_min_less_zero_point, n), output_max_less_zero_point, n);
    acc_f32v = vfadd_vf_f32m8(acc_f32v, magic_bias, n);

    vuint32m8_t out_u32v = vfcvt_xu_f_v_u32m8(acc_f32v, n);
    out_u32v = vsub_vx_u32m8(out_u32v, magic_bias_less_output_zero_point, n);
    vuint16m4_t out_u16v = vncvt_x_x_w_u16m4(out_u32v, n);
    vuint8m2_t out_u8v = vncvt_x_x_w_u8m2(out_u16v, n);
    vse8_v_u8m2(output, out_u8v, n); output += n;

    batch -= n;
  } while (batch != 0);
}

void xnn_f32_gemm_minmax_ukernel_1x4v__rvv(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  const float* a0 = a;
  float* c0 = c;

  const size_t nr = vsetvlmax_e32m4();
  size_t vl = nr;
  do {
    if XNN_UNLIKELY(nc < nr) {
      vl = vsetvl_e32m4(nc);
    }
    nc = nc - vl;

    vfloat32m4_t vacc0 =  vle32_v_f32m4(w, vl);
    w = w + nr;

    size_t k = kc;
    do {
      const float va0 = *a0++;
      vfloat32m4_t vb = vle32_v_f32m4(w, vl);
      w = w + nr;
      vacc0 = vfmacc_vf_f32m4(vacc0, va0, vb, vl);
      k -= sizeof(float);
    } while (k != 0);
    // clamp results with min & max
    vacc0 = vfmax_vf_f32m4(vacc0, vmin, vl);

    vacc0 = vfmin_vf_f32m4(vacc0, vmax, vl);
    // store 1 x vl results to c
    vse32_v_f32m4(c0, vacc0, vl);
    c0 = (float*) ((uintptr_t) c0 + cn_stride);
    a0 = (const float*) ((uintptr_t) a0 - kc);
  } while (nc != 0);
}

void xnn_f32_gemm_relu_ukernel_1x4v__rvv(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float vmin = 0.0f;
  const float* a0 = a;
  float* c0 = c;

  const size_t nr = vsetvlmax_e32m4();
  size_t vl = nr;
  do {
    if XNN_UNLIKELY(nc < nr) {
      vl = vsetvl_e32m4(nc);
    }
    nc = nc - vl;

    vfloat32m4_t vacc0 =  vle32_v_f32m4(w, vl);
    w = w + nr;

    size_t k = kc;
    do {
      const float va0 = *a0++;
      vfloat32m4_t vb = vle32_v_f32m4(w, vl);
      w = w + nr;
      vacc0 = vfmacc_vf_f32m4(vacc0, va0, vb, vl);
      k -= sizeof(float);
    } while (k != 0);
    // apply ReLU to results
    vacc0 = vfmax_vf_f32m4(vacc0, vmin, vl);
    // store 1 x vl results to c
    vse32_v_f32m4(c0, vacc0, vl);
    c0 = (float*) ((uintptr_t) c0 + cn_stride);
    a0 = (const float*) ((uintptr_t) a0 - kc);
  } while (nc != 0);
}

void xnn_f32_gemm_ukernel_1x4v__rvv(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;

  const size_t nr = vsetvlmax_e32m4();
  size_t vl = nr;
  do {
    if XNN_UNLIKELY(nc < nr) {
      vl = vsetvl_e32m4(nc);
    }
    nc = nc - vl;

    vfloat32m4_t vacc0 =  vle32_v_f32m4(w, vl);
    w = w + nr;

    size_t k = kc;
    do {
      const float va0 = *a0++;
      vfloat32m4_t vb = vle32_v_f32m4(w, vl);
      w = w + nr;
      vacc0 = vfmacc_vf_f32m4(vacc0, va0, vb, vl);
      k -= sizeof(float);
    } while (k != 0);
    // store 1 x vl results to c
    vse32_v_f32m4(c0, vacc0, vl);
    c0 = (float*) ((uintptr_t) c0 + cn_stride);
    a0 = (const float*) ((uintptr_t) a0 - kc);
  } while (nc != 0);
}

void xnn_f32_gemm_minmax_ukernel_7x4v__rvv(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 7);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  const float* a0 = a;
  float* c0 = c;
  const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    a3 = a2;
    c3 = c2;
  }
  const float* a4 = (const float*) ((uintptr_t) a3 + a_stride);
  float* c4 = (float*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    a4 = a3;
    c4 = c3;
  }
  const float* a5 = (const float*) ((uintptr_t) a4 + a_stride);
  float* c5 = (float*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 6) {
    a5 = a4;
    c5 = c4;
  }
  const float* a6 = (const float*) ((uintptr_t) a5 + a_stride);
  float* c6 = (float*) ((uintptr_t) c5 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 6) {
    a6 = a5;
    c6 = c5;
  }

  const size_t nr = vsetvlmax_e32m4();
  size_t vl = nr;
  do {
    if XNN_UNLIKELY(nc < nr) {
      vl = vsetvl_e32m4(nc);
    }
    nc = nc - vl;

    vfloat32m4_t vacc0 =  vle32_v_f32m4(w, vl);
    w = w + nr;
    vfloat32m4_t vacc1 =  vmv_v_v_f32m4(vacc0, vl);
    vfloat32m4_t vacc2 =  vmv_v_v_f32m4(vacc0, vl);
    vfloat32m4_t vacc3 =  vmv_v_v_f32m4(vacc0, vl);
    vfloat32m4_t vacc4 =  vmv_v_v_f32m4(vacc0, vl);
    vfloat32m4_t vacc5 =  vmv_v_v_f32m4(vacc0, vl);
    vfloat32m4_t vacc6 =  vmv_v_v_f32m4(vacc0, vl);

    size_t k = kc;
    do {
      const float va0 = *a0++;
      const float va1 = *a1++;
      const float va2 = *a2++;
      const float va3 = *a3++;
      const float va4 = *a4++;
      const float va5 = *a5++;
      const float va6 = *a6++;
      vfloat32m4_t vb = vle32_v_f32m4(w, vl);
      w = w + nr;
      vacc0 = vfmacc_vf_f32m4(vacc0, va0, vb, vl);
      vacc1 = vfmacc_vf_f32m4(vacc1, va1, vb, vl);
      vacc2 = vfmacc_vf_f32m4(vacc2, va2, vb, vl);
      vacc3 = vfmacc_vf_f32m4(vacc3, va3, vb, vl);
      vacc4 = vfmacc_vf_f32m4(vacc4, va4, vb, vl);
      vacc5 = vfmacc_vf_f32m4(vacc5, va5, vb, vl);
      vacc6 = vfmacc_vf_f32m4(vacc6, va6, vb, vl);
      k -= sizeof(float);
    } while (k != 0);
    // clamp results with min & max
    vacc0 = vfmax_vf_f32m4(vacc0, vmin, vl);
    vacc1 = vfmax_vf_f32m4(vacc1, vmin, vl);
    vacc2 = vfmax_vf_f32m4(vacc2, vmin, vl);
    vacc3 = vfmax_vf_f32m4(vacc3, vmin, vl);
    vacc4 = vfmax_vf_f32m4(vacc4, vmin, vl);
    vacc5 = vfmax_vf_f32m4(vacc5, vmin, vl);
    vacc6 = vfmax_vf_f32m4(vacc6, vmin, vl);

    vacc0 = vfmin_vf_f32m4(vacc0, vmax, vl);
    vacc1 = vfmin_vf_f32m4(vacc1, vmax, vl);
    vacc2 = vfmin_vf_f32m4(vacc2, vmax, vl);
    vacc3 = vfmin_vf_f32m4(vacc3, vmax, vl);
    vacc4 = vfmin_vf_f32m4(vacc4, vmax, vl);
    vacc5 = vfmin_vf_f32m4(vacc5, vmax, vl);
    vacc6 = vfmin_vf_f32m4(vacc6, vmax, vl);
    // store 7 x vl results to c
    vse32_v_f32m4(c0, vacc0, vl);
    c0 = (float*) ((uintptr_t) c0 + cn_stride);
    vse32_v_f32m4(c1, vacc1, vl);
    c1 = (float*) ((uintptr_t) c1 + cn_stride);
    vse32_v_f32m4(c2, vacc2, vl);
    c2 = (float*) ((uintptr_t) c2 + cn_stride);
    vse32_v_f32m4(c3, vacc3, vl);
    c3 = (float*) ((uintptr_t) c3 + cn_stride);
    vse32_v_f32m4(c4, vacc4, vl);
    c4 = (float*) ((uintptr_t) c4 + cn_stride);
    vse32_v_f32m4(c5, vacc5, vl);
    c5 = (float*) ((uintptr_t) c5 + cn_stride);
    vse32_v_f32m4(c6, vacc6, vl);
    c6 = (float*) ((uintptr_t) c6 + cn_stride);
    a0 = (const float*) ((uintptr_t) a0 - kc);
    a1 = (const float*) ((uintptr_t) a1 - kc);
    a2 = (const float*) ((uintptr_t) a2 - kc);
    a3 = (const float*) ((uintptr_t) a3 - kc);
    a4 = (const float*) ((uintptr_t) a4 - kc);
    a5 = (const float*) ((uintptr_t) a5 - kc);
    a6 = (const float*) ((uintptr_t) a6 - kc);
  } while (nc != 0);
}

void xnn_f32_gemm_relu_ukernel_7x4v__rvv(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 7);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float vmin = 0.0f;
  const float* a0 = a;
  float* c0 = c;
  const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    a3 = a2;
    c3 = c2;
  }
  const float* a4 = (const float*) ((uintptr_t) a3 + a_stride);
  float* c4 = (float*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    a4 = a3;
    c4 = c3;
  }
  const float* a5 = (const float*) ((uintptr_t) a4 + a_stride);
  float* c5 = (float*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 6) {
    a5 = a4;
    c5 = c4;
  }
  const float* a6 = (const float*) ((uintptr_t) a5 + a_stride);
  float* c6 = (float*) ((uintptr_t) c5 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 6) {
    a6 = a5;
    c6 = c5;
  }

  const size_t nr = vsetvlmax_e32m4();
  size_t vl = nr;
  do {
    if XNN_UNLIKELY(nc < nr) {
      vl = vsetvl_e32m4(nc);
    }
    nc = nc - vl;

    vfloat32m4_t vacc0 =  vle32_v_f32m4(w, vl);
    w = w + nr;
    vfloat32m4_t vacc1 =  vmv_v_v_f32m4(vacc0, vl);
    vfloat32m4_t vacc2 =  vmv_v_v_f32m4(vacc0, vl);
    vfloat32m4_t vacc3 =  vmv_v_v_f32m4(vacc0, vl);
    vfloat32m4_t vacc4 =  vmv_v_v_f32m4(vacc0, vl);
    vfloat32m4_t vacc5 =  vmv_v_v_f32m4(vacc0, vl);
    vfloat32m4_t vacc6 =  vmv_v_v_f32m4(vacc0, vl);

    size_t k = kc;
    do {
      const float va0 = *a0++;
      const float va1 = *a1++;
      const float va2 = *a2++;
      const float va3 = *a3++;
      const float va4 = *a4++;
      const float va5 = *a5++;
      const float va6 = *a6++;
      vfloat32m4_t vb = vle32_v_f32m4(w, vl);
      w = w + nr;
      vacc0 = vfmacc_vf_f32m4(vacc0, va0, vb, vl);
      vacc1 = vfmacc_vf_f32m4(vacc1, va1, vb, vl);
      vacc2 = vfmacc_vf_f32m4(vacc2, va2, vb, vl);
      vacc3 = vfmacc_vf_f32m4(vacc3, va3, vb, vl);
      vacc4 = vfmacc_vf_f32m4(vacc4, va4, vb, vl);
      vacc5 = vfmacc_vf_f32m4(vacc5, va5, vb, vl);
      vacc6 = vfmacc_vf_f32m4(vacc6, va6, vb, vl);
      k -= sizeof(float);
    } while (k != 0);
    // apply ReLU to results
    vacc0 = vfmax_vf_f32m4(vacc0, vmin, vl);
    vacc1 = vfmax_vf_f32m4(vacc1, vmin, vl);
    vacc2 = vfmax_vf_f32m4(vacc2, vmin, vl);
    vacc3 = vfmax_vf_f32m4(vacc3, vmin, vl);
    vacc4 = vfmax_vf_f32m4(vacc4, vmin, vl);
    vacc5 = vfmax_vf_f32m4(vacc5, vmin, vl);
    vacc6 = vfmax_vf_f32m4(vacc6, vmin, vl);
    // store 7 x vl results to c
    vse32_v_f32m4(c0, vacc0, vl);
    c0 = (float*) ((uintptr_t) c0 + cn_stride);
    vse32_v_f32m4(c1, vacc1, vl);
    c1 = (float*) ((uintptr_t) c1 + cn_stride);
    vse32_v_f32m4(c2, vacc2, vl);
    c2 = (float*) ((uintptr_t) c2 + cn_stride);
    vse32_v_f32m4(c3, vacc3, vl);
    c3 = (float*) ((uintptr_t) c3 + cn_stride);
    vse32_v_f32m4(c4, vacc4, vl);
    c4 = (float*) ((uintptr_t) c4 + cn_stride);
    vse32_v_f32m4(c5, vacc5, vl);
    c5 = (float*) ((uintptr_t) c5 + cn_stride);
    vse32_v_f32m4(c6, vacc6, vl);
    c6 = (float*) ((uintptr_t) c6 + cn_stride);
    a0 = (const float*) ((uintptr_t) a0 - kc);
    a1 = (const float*) ((uintptr_t) a1 - kc);
    a2 = (const float*) ((uintptr_t) a2 - kc);
    a3 = (const float*) ((uintptr_t) a3 - kc);
    a4 = (const float*) ((uintptr_t) a4 - kc);
    a5 = (const float*) ((uintptr_t) a5 - kc);
    a6 = (const float*) ((uintptr_t) a6 - kc);
  } while (nc != 0);
}

void xnn_f32_gemm_ukernel_7x4v__rvv(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 7);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;
  const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    a3 = a2;
    c3 = c2;
  }
  const float* a4 = (const float*) ((uintptr_t) a3 + a_stride);
  float* c4 = (float*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    a4 = a3;
    c4 = c3;
  }
  const float* a5 = (const float*) ((uintptr_t) a4 + a_stride);
  float* c5 = (float*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 6) {
    a5 = a4;
    c5 = c4;
  }
  const float* a6 = (const float*) ((uintptr_t) a5 + a_stride);
  float* c6 = (float*) ((uintptr_t) c5 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 6) {
    a6 = a5;
    c6 = c5;
  }

  const size_t nr = vsetvlmax_e32m4();
  size_t vl = nr;
  do {
    if XNN_UNLIKELY(nc < nr) {
      vl = vsetvl_e32m4(nc);
    }
    nc = nc - vl;

    vfloat32m4_t vacc0 =  vle32_v_f32m4(w, vl);
    w = w + nr;
    vfloat32m4_t vacc1 =  vmv_v_v_f32m4(vacc0, vl);
    vfloat32m4_t vacc2 =  vmv_v_v_f32m4(vacc0, vl);
    vfloat32m4_t vacc3 =  vmv_v_v_f32m4(vacc0, vl);
    vfloat32m4_t vacc4 =  vmv_v_v_f32m4(vacc0, vl);
    vfloat32m4_t vacc5 =  vmv_v_v_f32m4(vacc0, vl);
    vfloat32m4_t vacc6 =  vmv_v_v_f32m4(vacc0, vl);

    size_t k = kc;
    do {
      const float va0 = *a0++;
      const float va1 = *a1++;
      const float va2 = *a2++;
      const float va3 = *a3++;
      const float va4 = *a4++;
      const float va5 = *a5++;
      const float va6 = *a6++;
      vfloat32m4_t vb = vle32_v_f32m4(w, vl);
      w = w + nr;
      vacc0 = vfmacc_vf_f32m4(vacc0, va0, vb, vl);
      vacc1 = vfmacc_vf_f32m4(vacc1, va1, vb, vl);
      vacc2 = vfmacc_vf_f32m4(vacc2, va2, vb, vl);
      vacc3 = vfmacc_vf_f32m4(vacc3, va3, vb, vl);
      vacc4 = vfmacc_vf_f32m4(vacc4, va4, vb, vl);
      vacc5 = vfmacc_vf_f32m4(vacc5, va5, vb, vl);
      vacc6 = vfmacc_vf_f32m4(vacc6, va6, vb, vl);
      k -= sizeof(float);
    } while (k != 0);
    // store 7 x vl results to c
    vse32_v_f32m4(c0, vacc0, vl);
    c0 = (float*) ((uintptr_t) c0 + cn_stride);
    vse32_v_f32m4(c1, vacc1, vl);
    c1 = (float*) ((uintptr_t) c1 + cn_stride);
    vse32_v_f32m4(c2, vacc2, vl);
    c2 = (float*) ((uintptr_t) c2 + cn_stride);
    vse32_v_f32m4(c3, vacc3, vl);
    c3 = (float*) ((uintptr_t) c3 + cn_stride);
    vse32_v_f32m4(c4, vacc4, vl);
    c4 = (float*) ((uintptr_t) c4 + cn_stride);
    vse32_v_f32m4(c5, vacc5, vl);
    c5 = (float*) ((uintptr_t) c5 + cn_stride);
    vse32_v_f32m4(c6, vacc6, vl);
    c6 = (float*) ((uintptr_t) c6 + cn_stride);
    a0 = (const float*) ((uintptr_t) a0 - kc);
    a1 = (const float*) ((uintptr_t) a1 - kc);
    a2 = (const float*) ((uintptr_t) a2 - kc);
    a3 = (const float*) ((uintptr_t) a3 - kc);
    a4 = (const float*) ((uintptr_t) a4 - kc);
    a5 = (const float*) ((uintptr_t) a5 - kc);
    a6 = (const float*) ((uintptr_t) a6 - kc);
  } while (nc != 0);
}

void xnn_f32_igemm_minmax_ukernel_1x4v__rvv(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float** restrict a,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  float* c0 = c;

  const size_t nr = vsetvlmax_e32m4();
  size_t vl = nr;
  do {
    if XNN_UNLIKELY(nc < nr) {
      vl = vsetvl_e32m4(nc);
    }
    nc = nc - vl;
    vfloat32m4_t vacc0 =  vle32_v_f32m4(w, vl);
    w = w + nr;

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = kc;
      do {
        const float va0 = *a0++;
        vfloat32m4_t vb = vle32_v_f32m4(w, vl);
        w = w + nr;
        vacc0 = vfmacc_vf_f32m4(vacc0, va0, vb, vl);

        k -= sizeof(float);
      } while (k != 0);
      p -= 1 * sizeof(void*);
    } while (p != 0);
    // clamp results with min & max
    vacc0 = vfmax_vf_f32m4(vacc0, vmin, vl);

    vacc0 = vfmin_vf_f32m4(vacc0, vmax, vl);
    // store 1 x vl results to c
    vse32_v_f32m4(c0, vacc0, vl);
    c0 = (float*) ((uintptr_t) c0 + cn_stride);

    a = (const float**restrict) ((uintptr_t) a - ks);
  } while (nc != 0);
}

void xnn_f32_igemm_relu_ukernel_1x4v__rvv(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float** restrict a,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float vmin = 0.0f;
  float* c0 = c;

  const size_t nr = vsetvlmax_e32m4();
  size_t vl = nr;
  do {
    if XNN_UNLIKELY(nc < nr) {
      vl = vsetvl_e32m4(nc);
    }
    nc = nc - vl;
    vfloat32m4_t vacc0 =  vle32_v_f32m4(w, vl);
    w = w + nr;

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = kc;
      do {
        const float va0 = *a0++;
        vfloat32m4_t vb = vle32_v_f32m4(w, vl);
        w = w + nr;
        vacc0 = vfmacc_vf_f32m4(vacc0, va0, vb, vl);

        k -= sizeof(float);
      } while (k != 0);
      p -= 1 * sizeof(void*);
    } while (p != 0);
    // apply ReLU to results
    vacc0 = vfmax_vf_f32m4(vacc0, vmin, vl);
    // store 1 x vl results to c
    vse32_v_f32m4(c0, vacc0, vl);
    c0 = (float*) ((uintptr_t) c0 + cn_stride);

    a = (const float**restrict) ((uintptr_t) a - ks);
  } while (nc != 0);
}

void xnn_f32_igemm_ukernel_1x4v__rvv(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float** restrict a,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  float* c0 = c;

  const size_t nr = vsetvlmax_e32m4();
  size_t vl = nr;
  do {
    if XNN_UNLIKELY(nc < nr) {
      vl = vsetvl_e32m4(nc);
    }
    nc = nc - vl;
    vfloat32m4_t vacc0 =  vle32_v_f32m4(w, vl);
    w = w + nr;

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = kc;
      do {
        const float va0 = *a0++;
        vfloat32m4_t vb = vle32_v_f32m4(w, vl);
        w = w + nr;
        vacc0 = vfmacc_vf_f32m4(vacc0, va0, vb, vl);

        k -= sizeof(float);
      } while (k != 0);
      p -= 1 * sizeof(void*);
    } while (p != 0);
    // store 1 x vl results to c
    vse32_v_f32m4(c0, vacc0, vl);
    c0 = (float*) ((uintptr_t) c0 + cn_stride);

    a = (const float**restrict) ((uintptr_t) a - ks);
  } while (nc != 0);
}

void xnn_f32_igemm_minmax_ukernel_7x4v__rvv(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float** restrict a,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 7);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (7 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  float* c0 = c;
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    c3 = c2;
  }
  float* c4 = (float*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    c4 = c3;
  }
  float* c5 = (float*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 6) {
    c5 = c4;
  }
  float* c6 = (float*) ((uintptr_t) c5 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 6) {
    c6 = c5;
  }

  const size_t nr = vsetvlmax_e32m4();
  size_t vl = nr;
  do {
    if XNN_UNLIKELY(nc < nr) {
      vl = vsetvl_e32m4(nc);
    }
    nc = nc - vl;
    vfloat32m4_t vacc0 =  vle32_v_f32m4(w, vl);
    w = w + nr;
    vfloat32m4_t vacc1 =  vmv_v_v_f32m4(vacc0, vl);
    vfloat32m4_t vacc2 =  vmv_v_v_f32m4(vacc0, vl);
    vfloat32m4_t vacc3 =  vmv_v_v_f32m4(vacc0, vl);
    vfloat32m4_t vacc4 =  vmv_v_v_f32m4(vacc0, vl);
    vfloat32m4_t vacc5 =  vmv_v_v_f32m4(vacc0, vl);
    vfloat32m4_t vacc6 =  vmv_v_v_f32m4(vacc0, vl);

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      const float* restrict a1 = a[1];
      assert(a1 != NULL);
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const float*) ((uintptr_t) a1 + a_offset);
      }
      const float* restrict a2 = a[2];
      assert(a2 != NULL);
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const float*) ((uintptr_t) a2 + a_offset);
      }
      const float* restrict a3 = a[3];
      assert(a3 != NULL);
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const float*) ((uintptr_t) a3 + a_offset);
      }
      const float* restrict a4 = a[4];
      assert(a4 != NULL);
      if XNN_UNPREDICTABLE(a4 != zero) {
        a4 = (const float*) ((uintptr_t) a4 + a_offset);
      }
      const float* restrict a5 = a[5];
      assert(a5 != NULL);
      if XNN_UNPREDICTABLE(a5 != zero) {
        a5 = (const float*) ((uintptr_t) a5 + a_offset);
      }
      const float* restrict a6 = a[6];
      assert(a6 != NULL);
      if XNN_UNPREDICTABLE(a6 != zero) {
        a6 = (const float*) ((uintptr_t) a6 + a_offset);
      }
      a += 7;

      size_t k = kc;
      do {
        const float va0 = *a0++;
        const float va1 = *a1++;
        const float va2 = *a2++;
        const float va3 = *a3++;
        const float va4 = *a4++;
        const float va5 = *a5++;
        const float va6 = *a6++;
        vfloat32m4_t vb = vle32_v_f32m4(w, vl);
        w = w + nr;
        vacc0 = vfmacc_vf_f32m4(vacc0, va0, vb, vl);
        vacc1 = vfmacc_vf_f32m4(vacc1, va1, vb, vl);
        vacc2 = vfmacc_vf_f32m4(vacc2, va2, vb, vl);
        vacc3 = vfmacc_vf_f32m4(vacc3, va3, vb, vl);
        vacc4 = vfmacc_vf_f32m4(vacc4, va4, vb, vl);
        vacc5 = vfmacc_vf_f32m4(vacc5, va5, vb, vl);
        vacc6 = vfmacc_vf_f32m4(vacc6, va6, vb, vl);

        k -= sizeof(float);
      } while (k != 0);
      p -= 7 * sizeof(void*);
    } while (p != 0);
    // clamp results with min & max
    vacc0 = vfmax_vf_f32m4(vacc0, vmin, vl);
    vacc1 = vfmax_vf_f32m4(vacc1, vmin, vl);
    vacc2 = vfmax_vf_f32m4(vacc2, vmin, vl);
    vacc3 = vfmax_vf_f32m4(vacc3, vmin, vl);
    vacc4 = vfmax_vf_f32m4(vacc4, vmin, vl);
    vacc5 = vfmax_vf_f32m4(vacc5, vmin, vl);
    vacc6 = vfmax_vf_f32m4(vacc6, vmin, vl);

    vacc0 = vfmin_vf_f32m4(vacc0, vmax, vl);
    vacc1 = vfmin_vf_f32m4(vacc1, vmax, vl);
    vacc2 = vfmin_vf_f32m4(vacc2, vmax, vl);
    vacc3 = vfmin_vf_f32m4(vacc3, vmax, vl);
    vacc4 = vfmin_vf_f32m4(vacc4, vmax, vl);
    vacc5 = vfmin_vf_f32m4(vacc5, vmax, vl);
    vacc6 = vfmin_vf_f32m4(vacc6, vmax, vl);
    // store 7 x vl results to c
    vse32_v_f32m4(c6, vacc6, vl);
    c6 = (float*) ((uintptr_t) c6 + cn_stride);
    vse32_v_f32m4(c5, vacc5, vl);
    c5 = (float*) ((uintptr_t) c5 + cn_stride);
    vse32_v_f32m4(c4, vacc4, vl);
    c4 = (float*) ((uintptr_t) c4 + cn_stride);
    vse32_v_f32m4(c3, vacc3, vl);
    c3 = (float*) ((uintptr_t) c3 + cn_stride);
    vse32_v_f32m4(c2, vacc2, vl);
    c2 = (float*) ((uintptr_t) c2 + cn_stride);
    vse32_v_f32m4(c1, vacc1, vl);
    c1 = (float*) ((uintptr_t) c1 + cn_stride);
    vse32_v_f32m4(c0, vacc0, vl);
    c0 = (float*) ((uintptr_t) c0 + cn_stride);

    a = (const float**restrict) ((uintptr_t) a - ks);
  } while (nc != 0);
}

void xnn_f32_igemm_relu_ukernel_7x4v__rvv(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float** restrict a,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 7);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (7 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float vmin = 0.0f;
  float* c0 = c;
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    c3 = c2;
  }
  float* c4 = (float*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    c4 = c3;
  }
  float* c5 = (float*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 6) {
    c5 = c4;
  }
  float* c6 = (float*) ((uintptr_t) c5 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 6) {
    c6 = c5;
  }

  const size_t nr = vsetvlmax_e32m4();
  size_t vl = nr;
  do {
    if XNN_UNLIKELY(nc < nr) {
      vl = vsetvl_e32m4(nc);
    }
    nc = nc - vl;
    vfloat32m4_t vacc0 =  vle32_v_f32m4(w, vl);
    w = w + nr;
    vfloat32m4_t vacc1 =  vmv_v_v_f32m4(vacc0, vl);
    vfloat32m4_t vacc2 =  vmv_v_v_f32m4(vacc0, vl);
    vfloat32m4_t vacc3 =  vmv_v_v_f32m4(vacc0, vl);
    vfloat32m4_t vacc4 =  vmv_v_v_f32m4(vacc0, vl);
    vfloat32m4_t vacc5 =  vmv_v_v_f32m4(vacc0, vl);
    vfloat32m4_t vacc6 =  vmv_v_v_f32m4(vacc0, vl);

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      const float* restrict a1 = a[1];
      assert(a1 != NULL);
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const float*) ((uintptr_t) a1 + a_offset);
      }
      const float* restrict a2 = a[2];
      assert(a2 != NULL);
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const float*) ((uintptr_t) a2 + a_offset);
      }
      const float* restrict a3 = a[3];
      assert(a3 != NULL);
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const float*) ((uintptr_t) a3 + a_offset);
      }
      const float* restrict a4 = a[4];
      assert(a4 != NULL);
      if XNN_UNPREDICTABLE(a4 != zero) {
        a4 = (const float*) ((uintptr_t) a4 + a_offset);
      }
      const float* restrict a5 = a[5];
      assert(a5 != NULL);
      if XNN_UNPREDICTABLE(a5 != zero) {
        a5 = (const float*) ((uintptr_t) a5 + a_offset);
      }
      const float* restrict a6 = a[6];
      assert(a6 != NULL);
      if XNN_UNPREDICTABLE(a6 != zero) {
        a6 = (const float*) ((uintptr_t) a6 + a_offset);
      }
      a += 7;

      size_t k = kc;
      do {
        const float va0 = *a0++;
        const float va1 = *a1++;
        const float va2 = *a2++;
        const float va3 = *a3++;
        const float va4 = *a4++;
        const float va5 = *a5++;
        const float va6 = *a6++;
        vfloat32m4_t vb = vle32_v_f32m4(w, vl);
        w = w + nr;
        vacc0 = vfmacc_vf_f32m4(vacc0, va0, vb, vl);
        vacc1 = vfmacc_vf_f32m4(vacc1, va1, vb, vl);
        vacc2 = vfmacc_vf_f32m4(vacc2, va2, vb, vl);
        vacc3 = vfmacc_vf_f32m4(vacc3, va3, vb, vl);
        vacc4 = vfmacc_vf_f32m4(vacc4, va4, vb, vl);
        vacc5 = vfmacc_vf_f32m4(vacc5, va5, vb, vl);
        vacc6 = vfmacc_vf_f32m4(vacc6, va6, vb, vl);

        k -= sizeof(float);
      } while (k != 0);
      p -= 7 * sizeof(void*);
    } while (p != 0);
    // apply ReLU to results
    vacc0 = vfmax_vf_f32m4(vacc0, vmin, vl);
    vacc1 = vfmax_vf_f32m4(vacc1, vmin, vl);
    vacc2 = vfmax_vf_f32m4(vacc2, vmin, vl);
    vacc3 = vfmax_vf_f32m4(vacc3, vmin, vl);
    vacc4 = vfmax_vf_f32m4(vacc4, vmin, vl);
    vacc5 = vfmax_vf_f32m4(vacc5, vmin, vl);
    vacc6 = vfmax_vf_f32m4(vacc6, vmin, vl);
    // store 7 x vl results to c
    vse32_v_f32m4(c6, vacc6, vl);
    c6 = (float*) ((uintptr_t) c6 + cn_stride);
    vse32_v_f32m4(c5, vacc5, vl);
    c5 = (float*) ((uintptr_t) c5 + cn_stride);
    vse32_v_f32m4(c4, vacc4, vl);
    c4 = (float*) ((uintptr_t) c4 + cn_stride);
    vse32_v_f32m4(c3, vacc3, vl);
    c3 = (float*) ((uintptr_t) c3 + cn_stride);
    vse32_v_f32m4(c2, vacc2, vl);
    c2 = (float*) ((uintptr_t) c2 + cn_stride);
    vse32_v_f32m4(c1, vacc1, vl);
    c1 = (float*) ((uintptr_t) c1 + cn_stride);
    vse32_v_f32m4(c0, vacc0, vl);
    c0 = (float*) ((uintptr_t) c0 + cn_stride);

    a = (const float**restrict) ((uintptr_t) a - ks);
  } while (nc != 0);
}

void xnn_f32_igemm_ukernel_7x4v__rvv(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float** restrict a,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 7);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (7 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  float* c0 = c;
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    c3 = c2;
  }
  float* c4 = (float*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    c4 = c3;
  }
  float* c5 = (float*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 6) {
    c5 = c4;
  }
  float* c6 = (float*) ((uintptr_t) c5 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 6) {
    c6 = c5;
  }

  const size_t nr = vsetvlmax_e32m4();
  size_t vl = nr;
  do {
    if XNN_UNLIKELY(nc < nr) {
      vl = vsetvl_e32m4(nc);
    }
    nc = nc - vl;
    vfloat32m4_t vacc0 =  vle32_v_f32m4(w, vl);
    w = w + nr;
    vfloat32m4_t vacc1 =  vmv_v_v_f32m4(vacc0, vl);
    vfloat32m4_t vacc2 =  vmv_v_v_f32m4(vacc0, vl);
    vfloat32m4_t vacc3 =  vmv_v_v_f32m4(vacc0, vl);
    vfloat32m4_t vacc4 =  vmv_v_v_f32m4(vacc0, vl);
    vfloat32m4_t vacc5 =  vmv_v_v_f32m4(vacc0, vl);
    vfloat32m4_t vacc6 =  vmv_v_v_f32m4(vacc0, vl);

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      const float* restrict a1 = a[1];
      assert(a1 != NULL);
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const float*) ((uintptr_t) a1 + a_offset);
      }
      const float* restrict a2 = a[2];
      assert(a2 != NULL);
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const float*) ((uintptr_t) a2 + a_offset);
      }
      const float* restrict a3 = a[3];
      assert(a3 != NULL);
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const float*) ((uintptr_t) a3 + a_offset);
      }
      const float* restrict a4 = a[4];
      assert(a4 != NULL);
      if XNN_UNPREDICTABLE(a4 != zero) {
        a4 = (const float*) ((uintptr_t) a4 + a_offset);
      }
      const float* restrict a5 = a[5];
      assert(a5 != NULL);
      if XNN_UNPREDICTABLE(a5 != zero) {
        a5 = (const float*) ((uintptr_t) a5 + a_offset);
      }
      const float* restrict a6 = a[6];
      assert(a6 != NULL);
      if XNN_UNPREDICTABLE(a6 != zero) {
        a6 = (const float*) ((uintptr_t) a6 + a_offset);
      }
      a += 7;

      size_t k = kc;
      do {
        const float va0 = *a0++;
        const float va1 = *a1++;
        const float va2 = *a2++;
        const float va3 = *a3++;
        const float va4 = *a4++;
        const float va5 = *a5++;
        const float va6 = *a6++;
        vfloat32m4_t vb = vle32_v_f32m4(w, vl);
        w = w + nr;
        vacc0 = vfmacc_vf_f32m4(vacc0, va0, vb, vl);
        vacc1 = vfmacc_vf_f32m4(vacc1, va1, vb, vl);
        vacc2 = vfmacc_vf_f32m4(vacc2, va2, vb, vl);
        vacc3 = vfmacc_vf_f32m4(vacc3, va3, vb, vl);
        vacc4 = vfmacc_vf_f32m4(vacc4, va4, vb, vl);
        vacc5 = vfmacc_vf_f32m4(vacc5, va5, vb, vl);
        vacc6 = vfmacc_vf_f32m4(vacc6, va6, vb, vl);

        k -= sizeof(float);
      } while (k != 0);
      p -= 7 * sizeof(void*);
    } while (p != 0);
    // store 7 x vl results to c
    vse32_v_f32m4(c6, vacc6, vl);
    c6 = (float*) ((uintptr_t) c6 + cn_stride);
    vse32_v_f32m4(c5, vacc5, vl);
    c5 = (float*) ((uintptr_t) c5 + cn_stride);
    vse32_v_f32m4(c4, vacc4, vl);
    c4 = (float*) ((uintptr_t) c4 + cn_stride);
    vse32_v_f32m4(c3, vacc3, vl);
    c3 = (float*) ((uintptr_t) c3 + cn_stride);
    vse32_v_f32m4(c2, vacc2, vl);
    c2 = (float*) ((uintptr_t) c2 + cn_stride);
    vse32_v_f32m4(c1, vacc1, vl);
    c1 = (float*) ((uintptr_t) c1 + cn_stride);
    vse32_v_f32m4(c0, vacc0, vl);
    c0 = (float*) ((uintptr_t) c0 + cn_stride);

    a = (const float**restrict) ((uintptr_t) a - ks);
  } while (nc != 0);
}


void xnn_f32_dwconv_minmax_ukernel_3p4c__rvv(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;

  do {
    const float* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
    }
    const float* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
    }
    const float* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
    }
    input = (const float**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* w = weights;
    for (; c >= 4; c -= 4) {
      vfloat32m4_t vacc0123p0 = vle32_v_f32m4(w, 4);
      
      const vfloat32m4_t vi0x0123 = vle32_v_f32m4(i0, 4);
      i0 += 4;
      const vfloat32m4_t vk0x0123 = vle32_v_f32m4(w + 4, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi0x0123, vk0x0123, 4), 4);

      const vfloat32m4_t vi1x0123 = vle32_v_f32m4(i1, 4);
      i1 += 4;
      const vfloat32m4_t vk1x0123 = vle32_v_f32m4(w + 8, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi1x0123, vk1x0123, 4), 4);

      const vfloat32m4_t vi2x0123 = vle32_v_f32m4(i2, 4);
      i2 += 4;
      const vfloat32m4_t vk2x0123 = vle32_v_f32m4(w + 12, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi2x0123, vk2x0123, 4), 4);

      w += 16;

      vfloat32m4_t vacc0123 = vfmax_vf_f32m4(vacc0123p0,vmin, 4);
      vacc0123 = vfmin_vf_f32m4( vacc0123, vmax,4);

      vse32_v_f32m4(output, vacc0123, 4);
      output += 4;
    }

    if XNN_UNLIKELY(c != 0) {
      vfloat32m4_t vacc0123p0 = vle32_v_f32m4(w, 4);

      const vfloat32m4_t vi0x0123 = vle32_v_f32m4(i0, 4);
      const vfloat32m4_t vk0x0123 = vle32_v_f32m4(w + 4, 4);
      vacc0123p0 = vfmacc_vv_f32m4(vacc0123p0, vi0x0123, vk0x0123, 4);

      const vfloat32m4_t vi1x0123 = vle32_v_f32m4(i1, 4);
      const vfloat32m4_t vk1x0123 = vle32_v_f32m4(w + 8, 4);
      vacc0123p0 = vfmacc_vv_f32m4(vacc0123p0, vi1x0123, vk1x0123, 4);

      const vfloat32m4_t vi2x0123 = vle32_v_f32m4(i2, 4);
      const vfloat32m4_t vk2x0123 = vle32_v_f32m4(w + 12, 4);
      vacc0123p0 = vfmacc_vv_f32m4(vacc0123p0, vi2x0123, vk2x0123, 4);

      vfloat32m4_t vacc0123 = vfmax_vf_f32m4(vacc0123p0, vmin, 4);
      vacc0123 = vfmin_vf_f32m4(vacc0123, vmax, 4);

      if (c & 2) {
        vse32_v_f32m4(output, vacc0123, 2);
        output += 2;
        vacc0123 = vrgather_vx_f32m4(vacc0123, 2, 1);
      }
      if (c & 1) {
        vse32_v_f32m4(output, vacc0123, 1);
        output += 1;
      }
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}


void xnn_f32_dwconv_ukernel_3p4c__rvv(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);

  do {
    const float* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
    }
    const float* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
    }
    const float* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
    }
    input = (const float**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* w = weights;
    for (; c >= 4; c -= 4) {
      vfloat32m4_t vacc0123p0 = vle32_v_f32m4(w, 4);
      
      const vfloat32m4_t vi0x0123 = vle32_v_f32m4(i0, 4);
      i0 += 4;
      const vfloat32m4_t vk0x0123 = vle32_v_f32m4(w + 4, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi0x0123, vk0x0123, 4), 4);

      const vfloat32m4_t vi1x0123 = vle32_v_f32m4(i1, 4);
      i1 += 4;
      const vfloat32m4_t vk1x0123 = vle32_v_f32m4(w + 8, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi1x0123, vk1x0123, 4), 4);

      const vfloat32m4_t vi2x0123 = vle32_v_f32m4(i2, 4);
      i2 += 4;
      const vfloat32m4_t vk2x0123 = vle32_v_f32m4(w + 12, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi2x0123, vk2x0123, 4), 4);

      w += 16;

      const vfloat32m4_t vacc0123 = vacc0123p0;
      vse32_v_f32m4(output, vacc0123, 4);
      output += 4;     
    }

    if XNN_UNLIKELY(c != 0) {
      vfloat32m4_t vacc0123p0 = vle32_v_f32m4(w, 4);

      const vfloat32m4_t vi0x0123 = vle32_v_f32m4(i0, 4);
      const vfloat32m4_t vk0x0123 = vle32_v_f32m4(w + 4, 4);
      vacc0123p0 = vfmacc_vv_f32m4(vacc0123p0, vi0x0123, vk0x0123, 4);

      const vfloat32m4_t vi1x0123 = vle32_v_f32m4(i1, 4);
      const vfloat32m4_t vk1x0123 = vle32_v_f32m4(w + 8, 4);
      vacc0123p0 = vfmacc_vv_f32m4(vacc0123p0, vi1x0123, vk1x0123, 4);

      const vfloat32m4_t vi2x0123 = vle32_v_f32m4(i2, 4);
      const vfloat32m4_t vk2x0123 = vle32_v_f32m4(w + 12, 4);
      vacc0123p0 = vfmacc_vv_f32m4(vacc0123p0, vi2x0123, vk2x0123, 4);


      vfloat32m4_t vacc0123 = vacc0123p0;

      if (c & 2) {
        vse32_v_f32m4(output, vacc0123, 2);
        output += 2;
        vacc0123 = vrgather_vx_f32m4(vacc0123, 2, 1);
      }
      if (c & 1) {
        vse32_v_f32m4(output, vacc0123, 1);
        output += 1;
      }
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}


void xnn_f32_dwconv_ukernel_25p8c__rvv(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  do {
    const float* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
    }
    const float* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
    }
    const float* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
    }
    const float* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
    }
    const float* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
    }
    const float* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
    }
    const float* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
    }
    const float* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const float*) ((uintptr_t) i7 + input_offset);
    }
    const float* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const float*) ((uintptr_t) i8 + input_offset);
    }
    const float* i9 = input[9];
    assert(i9 != NULL);
    if XNN_UNPREDICTABLE(i9 != zero) {
      i9 = (const float*) ((uintptr_t) i9 + input_offset);
    }
    const float* i10 = input[10];
    assert(i10 != NULL);
    if XNN_UNPREDICTABLE(i10 != zero) {
      i10 = (const float*) ((uintptr_t) i10 + input_offset);
    }
    const float* i11 = input[11];
    assert(i11 != NULL);
    if XNN_UNPREDICTABLE(i11 != zero) {
      i11 = (const float*) ((uintptr_t) i11 + input_offset);
    }
    const float* i12 = input[12];
    assert(i12 != NULL);
    if XNN_UNPREDICTABLE(i12 != zero) {
      i12 = (const float*) ((uintptr_t) i12 + input_offset);
    }
    const float* i13 = input[13];
    assert(i13 != NULL);
    if XNN_UNPREDICTABLE(i13 != zero) {
      i13 = (const float*) ((uintptr_t) i13 + input_offset);
    }
    const float* i14 = input[14];
    assert(i14 != NULL);
    if XNN_UNPREDICTABLE(i14 != zero) {
      i14 = (const float*) ((uintptr_t) i14 + input_offset);
    }
    const float* i15 = input[15];
    assert(i15 != NULL);
    if XNN_UNPREDICTABLE(i15 != zero) {
      i15 = (const float*) ((uintptr_t) i15 + input_offset);
    }
    const float* i16 = input[16];
    assert(i16 != NULL);
    if XNN_UNPREDICTABLE(i16 != zero) {
      i16 = (const float*) ((uintptr_t) i16 + input_offset);
    }
    const float* i17 = input[17];
    assert(i17 != NULL);
    if XNN_UNPREDICTABLE(i17 != zero) {
      i17 = (const float*) ((uintptr_t) i17 + input_offset);
    }
    const float* i18 = input[18];
    assert(i18 != NULL);
    if XNN_UNPREDICTABLE(i18 != zero) {
      i18 = (const float*) ((uintptr_t) i18 + input_offset);
    }
    const float* i19 = input[19];
    assert(i19 != NULL);
    if XNN_UNPREDICTABLE(i19 != zero) {
      i19 = (const float*) ((uintptr_t) i19 + input_offset);
    }
    const float* i20 = input[20];
    assert(i20 != NULL);
    if XNN_UNPREDICTABLE(i20 != zero) {
      i20 = (const float*) ((uintptr_t) i20 + input_offset);
    }
    const float* i21 = input[21];
    assert(i21 != NULL);
    if XNN_UNPREDICTABLE(i21 != zero) {
      i21 = (const float*) ((uintptr_t) i21 + input_offset);
    }
    const float* i22 = input[22];
    assert(i22 != NULL);
    if XNN_UNPREDICTABLE(i22 != zero) {
      i22 = (const float*) ((uintptr_t) i22 + input_offset);
    }
    const float* i23 = input[23];
    assert(i23 != NULL);
    if XNN_UNPREDICTABLE(i23 != zero) {
      i23 = (const float*) ((uintptr_t) i23 + input_offset);
    }
    const float* i24 = input[24];
    assert(i24 != NULL);
    if XNN_UNPREDICTABLE(i24 != zero) {
      i24 = (const float*) ((uintptr_t) i24 + input_offset);
    }
    input = (const float**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* w = weights;
    for (; c >= 8; c -= 8) {
      vfloat32m4_t vacc0123p0 = vle32_v_f32m4(w, 4);
      vfloat32m4_t vacc4567p0 = vle32_v_f32m4(w + 4, 4);


      const vfloat32m4_t vi0x0123 = vle32_v_f32m4(i0, 4);
      const vfloat32m4_t vi0x4567 = vle32_v_f32m4(i0 + 4, 4);
      i0 += 8;

      const vfloat32m4_t vk0x0123 = vle32_v_f32m4(w + 8, 4);
      const vfloat32m4_t vk0x4567 = vle32_v_f32m4(w + 12, 4);

      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi0x0123, vk0x0123, 4), 4);
      vacc4567p0 = vfadd_vv_f32m4(vacc4567p0, vfmul_vv_f32m4(vi0x4567, vk0x4567, 4), 4);

      const vfloat32m4_t vi1x0123 = vle32_v_f32m4(i1, 4);
      const vfloat32m4_t vi1x4567 = vle32_v_f32m4(i1 + 4, 4);
      i1 += 8;

      const vfloat32m4_t vk1x0123 = vle32_v_f32m4(w + 16, 4);
      const vfloat32m4_t vk1x4567 = vle32_v_f32m4(w + 20, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi1x0123, vk1x0123, 4), 4);
      vacc4567p0 = vfadd_vv_f32m4(vacc4567p0, vfmul_vv_f32m4(vi1x4567, vk1x4567, 4), 4);

      const vfloat32m4_t vi2x0123 = vle32_v_f32m4(i2, 4);
      const vfloat32m4_t vi2x4567 = vle32_v_f32m4(i2 + 4, 4);
      i2 += 8;

      const vfloat32m4_t vk2x0123 = vle32_v_f32m4(w + 24, 4);
      const vfloat32m4_t vk2x4567 = vle32_v_f32m4(w + 28, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi2x0123, vk2x0123, 4), 4);
      vacc4567p0 = vfadd_vv_f32m4(vacc4567p0, vfmul_vv_f32m4(vi2x4567, vk2x4567, 4), 4);

      const vfloat32m4_t vi3x0123 = vle32_v_f32m4(i3, 4);
      const vfloat32m4_t vi3x4567 = vle32_v_f32m4(i3 + 4, 4);
      i3 += 8;

      const vfloat32m4_t vk3x0123 = vle32_v_f32m4(w + 32, 4);
      const vfloat32m4_t vk3x4567 = vle32_v_f32m4(w + 36, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi3x0123, vk3x0123, 4), 4);
      vacc4567p0 = vfadd_vv_f32m4(vacc4567p0, vfmul_vv_f32m4(vi3x4567, vk3x4567, 4), 4);

      const vfloat32m4_t vi4x0123 = vle32_v_f32m4(i4, 4);
      const vfloat32m4_t vi4x4567 = vle32_v_f32m4(i4 + 4, 4);
      i4 += 8;

      const vfloat32m4_t vk4x0123 = vle32_v_f32m4(w + 40, 4);
      const vfloat32m4_t vk4x4567 = vle32_v_f32m4(w + 44, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi4x0123, vk4x0123, 4), 4);
      vacc4567p0 = vfadd_vv_f32m4(vacc4567p0, vfmul_vv_f32m4(vi4x4567, vk4x4567, 4), 4);

      const vfloat32m4_t vi5x0123 = vle32_v_f32m4(i5, 4);
      const vfloat32m4_t vi5x4567 = vle32_v_f32m4(i5 + 4, 4);
      i5 += 8;

      const vfloat32m4_t vk5x0123 = vle32_v_f32m4(w + 48, 4);
      const vfloat32m4_t vk5x4567 = vle32_v_f32m4(w + 52, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi5x0123, vk5x0123, 4), 4);
      vacc4567p0 = vfadd_vv_f32m4(vacc4567p0, vfmul_vv_f32m4(vi5x4567, vk5x4567, 4), 4);

      const vfloat32m4_t vi6x0123 = vle32_v_f32m4(i6, 4);
      const vfloat32m4_t vi6x4567 = vle32_v_f32m4(i6 + 4, 4);
      i6 += 8;

      const vfloat32m4_t vk6x0123 = vle32_v_f32m4(w + 56, 4);
      const vfloat32m4_t vk6x4567 = vle32_v_f32m4(w + 60, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi6x0123, vk6x0123, 4), 4);
      vacc4567p0 = vfadd_vv_f32m4(vacc4567p0, vfmul_vv_f32m4(vi6x4567, vk6x4567, 4), 4);

      const vfloat32m4_t vi7x0123 = vle32_v_f32m4(i7, 4);
      const vfloat32m4_t vi7x4567 = vle32_v_f32m4(i7 + 4, 4);
      i7 += 8;

      const vfloat32m4_t vk7x0123 = vle32_v_f32m4(w + 64, 4);
      const vfloat32m4_t vk7x4567 = vle32_v_f32m4(w + 68, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi7x0123, vk7x0123, 4), 4);
      vacc4567p0 = vfadd_vv_f32m4(vacc4567p0, vfmul_vv_f32m4(vi7x4567, vk7x4567, 4), 4);

      const vfloat32m4_t vi8x0123 = vle32_v_f32m4(i8, 4);
      const vfloat32m4_t vi8x4567 = vle32_v_f32m4(i8 + 4, 4);
      i8 += 8;

      const vfloat32m4_t vk8x0123 = vle32_v_f32m4(w + 72, 4);
      const vfloat32m4_t vk8x4567 = vle32_v_f32m4(w + 76, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi8x0123, vk8x0123, 4), 4);
      vacc4567p0 = vfadd_vv_f32m4(vacc4567p0, vfmul_vv_f32m4(vi8x4567, vk8x4567, 4), 4);

      const vfloat32m4_t vi9x0123 = vle32_v_f32m4(i9, 4);
      const vfloat32m4_t vi9x4567 = vle32_v_f32m4(i9 + 4, 4);
      i9 += 8;

      const vfloat32m4_t vk9x0123 = vle32_v_f32m4(w + 80, 4);
      const vfloat32m4_t vk9x4567 = vle32_v_f32m4(w + 84, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi9x0123, vk9x0123, 4), 4);
      vacc4567p0 = vfadd_vv_f32m4(vacc4567p0, vfmul_vv_f32m4(vi9x4567, vk9x4567, 4), 4);

      const vfloat32m4_t vi10x0123 = vle32_v_f32m4(i10, 4);
      const vfloat32m4_t vi10x4567 = vle32_v_f32m4(i10 + 4, 4);
      i10 += 8;

      const vfloat32m4_t vk10x0123 = vle32_v_f32m4(w + 88, 4);
      const vfloat32m4_t vk10x4567 = vle32_v_f32m4(w + 92, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi10x0123, vk10x0123, 4), 4);
      vacc4567p0 = vfadd_vv_f32m4(vacc4567p0, vfmul_vv_f32m4(vi10x4567, vk10x4567, 4), 4);

      const vfloat32m4_t vi11x0123 = vle32_v_f32m4(i11, 4);
      const vfloat32m4_t vi11x4567 = vle32_v_f32m4(i11 + 4, 4);
      i11 += 8;

      const vfloat32m4_t vk11x0123 = vle32_v_f32m4(w + 96, 4);
      const vfloat32m4_t vk11x4567 = vle32_v_f32m4(w + 100, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi11x0123, vk11x0123, 4), 4);
      vacc4567p0 = vfadd_vv_f32m4(vacc4567p0, vfmul_vv_f32m4(vi11x4567, vk11x4567, 4), 4);

      const vfloat32m4_t vi12x0123 = vle32_v_f32m4(i12, 4);
      const vfloat32m4_t vi12x4567 = vle32_v_f32m4(i12 + 4, 4);
      i12 += 8;

      const vfloat32m4_t vk12x0123 = vle32_v_f32m4(w + 104, 4);
      const vfloat32m4_t vk12x4567 = vle32_v_f32m4(w + 108, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi12x0123, vk12x0123, 4), 4);
      vacc4567p0 = vfadd_vv_f32m4(vacc4567p0, vfmul_vv_f32m4(vi12x4567, vk12x4567, 4), 4);

      const vfloat32m4_t vi13x0123 = vle32_v_f32m4(i13, 4);
      const vfloat32m4_t vi13x4567 = vle32_v_f32m4(i13 + 4, 4);
      i13 += 8;

      const vfloat32m4_t vk13x0123 = vle32_v_f32m4(w + 112, 4);
      const vfloat32m4_t vk13x4567 = vle32_v_f32m4(w + 116, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi13x0123, vk13x0123, 4), 4);
      vacc4567p0 = vfadd_vv_f32m4(vacc4567p0, vfmul_vv_f32m4(vi13x4567, vk13x4567, 4), 4);

      const vfloat32m4_t vi14x0123 = vle32_v_f32m4(i14, 4);
      const vfloat32m4_t vi14x4567 = vle32_v_f32m4(i14 + 4, 4);
      i14 += 8;

      const vfloat32m4_t vk14x0123 = vle32_v_f32m4(w + 120, 4);
      const vfloat32m4_t vk14x4567 = vle32_v_f32m4(w + 124, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi14x0123, vk14x0123, 4), 4);
      vacc4567p0 = vfadd_vv_f32m4(vacc4567p0, vfmul_vv_f32m4(vi14x4567, vk14x4567, 4), 4);

      const vfloat32m4_t vi15x0123 = vle32_v_f32m4(i15, 4);
      const vfloat32m4_t vi15x4567 = vle32_v_f32m4(i15 + 4, 4);
      i15 += 8;

      const vfloat32m4_t vk15x0123 = vle32_v_f32m4(w + 128, 4);
      const vfloat32m4_t vk15x4567 = vle32_v_f32m4(w + 132, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi15x0123, vk15x0123, 4), 4);
      vacc4567p0 = vfadd_vv_f32m4(vacc4567p0, vfmul_vv_f32m4(vi15x4567, vk15x4567, 4), 4);

      const vfloat32m4_t vi16x0123 = vle32_v_f32m4(i16, 4);
      const vfloat32m4_t vi16x4567 = vle32_v_f32m4(i16 + 4, 4);
      i16 += 8;

      const vfloat32m4_t vk16x0123 = vle32_v_f32m4(w + 136, 4);
      const vfloat32m4_t vk16x4567 = vle32_v_f32m4(w + 140, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi16x0123, vk16x0123, 4), 4);
      vacc4567p0 = vfadd_vv_f32m4(vacc4567p0, vfmul_vv_f32m4(vi16x4567, vk16x4567, 4), 4);

      const vfloat32m4_t vi17x0123 = vle32_v_f32m4(i17, 4);
      const vfloat32m4_t vi17x4567 = vle32_v_f32m4(i17 + 4, 4);
      i17 += 8;

      const vfloat32m4_t vk17x0123 = vle32_v_f32m4(w + 144, 4);
      const vfloat32m4_t vk17x4567 = vle32_v_f32m4(w + 148, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi17x0123, vk17x0123, 4), 4);
      vacc4567p0 = vfadd_vv_f32m4(vacc4567p0, vfmul_vv_f32m4(vi17x4567, vk17x4567, 4), 4);

      const vfloat32m4_t vi18x0123 = vle32_v_f32m4(i18, 4);
      const vfloat32m4_t vi18x4567 = vle32_v_f32m4(i18 + 4, 4);
      i18 += 8;

      const vfloat32m4_t vk18x0123 = vle32_v_f32m4(w + 152, 4);
      const vfloat32m4_t vk18x4567 = vle32_v_f32m4(w + 156, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi18x0123, vk18x0123, 4), 4);
      vacc4567p0 = vfadd_vv_f32m4(vacc4567p0, vfmul_vv_f32m4(vi18x4567, vk18x4567, 4), 4);

      const vfloat32m4_t vi19x0123 = vle32_v_f32m4(i19, 4);
      const vfloat32m4_t vi19x4567 = vle32_v_f32m4(i19 + 4, 4);
      i19 += 8;

      const vfloat32m4_t vk19x0123 = vle32_v_f32m4(w + 160, 4);
      const vfloat32m4_t vk19x4567 = vle32_v_f32m4(w + 164, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi19x0123, vk19x0123, 4), 4);
      vacc4567p0 = vfadd_vv_f32m4(vacc4567p0, vfmul_vv_f32m4(vi19x4567, vk19x4567, 4), 4);

      const vfloat32m4_t vi20x0123 = vle32_v_f32m4(i20, 4);
      const vfloat32m4_t vi20x4567 = vle32_v_f32m4(i20 + 4, 4);
      i20 += 8;

      const vfloat32m4_t vk20x0123 = vle32_v_f32m4(w + 168, 4);
      const vfloat32m4_t vk20x4567 = vle32_v_f32m4(w + 172, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi20x0123, vk20x0123, 4), 4);
      vacc4567p0 = vfadd_vv_f32m4(vacc4567p0, vfmul_vv_f32m4(vi20x4567, vk20x4567, 4), 4);

      const vfloat32m4_t vi21x0123 = vle32_v_f32m4(i21, 4);
      const vfloat32m4_t vi21x4567 = vle32_v_f32m4(i21 + 4, 4);
      i21 += 8;

      const vfloat32m4_t vk21x0123 = vle32_v_f32m4(w + 176, 4);
      const vfloat32m4_t vk21x4567 = vle32_v_f32m4(w + 180, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi21x0123, vk21x0123, 4), 4);
      vacc4567p0 = vfadd_vv_f32m4(vacc4567p0, vfmul_vv_f32m4(vi21x4567, vk21x4567, 4), 4);

      const vfloat32m4_t vi22x0123 = vle32_v_f32m4(i22, 4);
      const vfloat32m4_t vi22x4567 = vle32_v_f32m4(i22 + 4, 4);
      i22 += 8;

      const vfloat32m4_t vk22x0123 = vle32_v_f32m4(w + 184, 4);
      const vfloat32m4_t vk22x4567 = vle32_v_f32m4(w + 188, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi22x0123, vk22x0123, 4), 4);
      vacc4567p0 = vfadd_vv_f32m4(vacc4567p0, vfmul_vv_f32m4(vi22x4567, vk22x4567, 4), 4);

      const vfloat32m4_t vi23x0123 = vle32_v_f32m4(i23, 4);
      const vfloat32m4_t vi23x4567 = vle32_v_f32m4(i23 + 4, 4);
      i23 += 8;

      const vfloat32m4_t vk23x0123 = vle32_v_f32m4(w + 192, 4);
      const vfloat32m4_t vk23x4567 = vle32_v_f32m4(w + 196, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi23x0123, vk23x0123, 4), 4);
      vacc4567p0 = vfadd_vv_f32m4(vacc4567p0, vfmul_vv_f32m4(vi23x4567, vk23x4567, 4), 4);

      const vfloat32m4_t vi24x0123 = vle32_v_f32m4(i24, 4);
      const vfloat32m4_t vi24x4567 = vle32_v_f32m4(i24 + 4, 4);
      i24 += 8;

      const vfloat32m4_t vk24x0123 = vle32_v_f32m4(w + 200, 4);
      const vfloat32m4_t vk24x4567 = vle32_v_f32m4(w + 204, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi24x0123, vk24x0123, 4), 4);
      vacc4567p0 = vfadd_vv_f32m4(vacc4567p0, vfmul_vv_f32m4(vi24x4567, vk24x4567, 4), 4);

      w += 208;


      const vfloat32m4_t vacc0123 = vacc0123p0;
      const vfloat32m4_t vacc4567 = vacc4567p0;

      vse32_v_f32m4(output, vacc0123, 4);
      vse32_v_f32m4(output + 4, vacc4567, 4);
      output += 8;
    }
    for (; c >= 4; c -= 4) {
      vfloat32m4_t vacc0123p0 = vle32_v_f32m4(w, 4);

      const vfloat32m4_t vi0x0123 = vle32_v_f32m4(i0, 4);
      i0 += 4;

      const vfloat32m4_t vk0x0123 = vle32_v_f32m4(w + 8, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi0x0123, vk0x0123, 4), 4);

      const vfloat32m4_t vi1x0123 = vle32_v_f32m4(i1, 4);
      i1 += 4;

      const vfloat32m4_t vk1x0123 = vle32_v_f32m4(w + 16, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi1x0123, vk1x0123, 4), 4);

      const vfloat32m4_t vi2x0123 = vle32_v_f32m4(i2, 4);
      i2 += 4;

      const vfloat32m4_t vk2x0123 = vle32_v_f32m4(w + 24, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi2x0123, vk2x0123, 4), 4);

      const vfloat32m4_t vi3x0123 = vle32_v_f32m4(i3, 4);
      i3 += 4;

      const vfloat32m4_t vk3x0123 = vle32_v_f32m4(w + 32, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi3x0123, vk3x0123, 4), 4);

      const vfloat32m4_t vi4x0123 = vle32_v_f32m4(i4, 4);
      i4 += 4;

      const vfloat32m4_t vk4x0123 = vle32_v_f32m4(w + 40, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi4x0123, vk4x0123, 4), 4);

      const vfloat32m4_t vi5x0123 = vle32_v_f32m4(i5, 4);
      i5 += 4;

      const vfloat32m4_t vk5x0123 = vle32_v_f32m4(w + 48, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi5x0123, vk5x0123, 4), 4);

      const vfloat32m4_t vi6x0123 = vle32_v_f32m4(i6, 4);
      i6 += 4;

      const vfloat32m4_t vk6x0123 = vle32_v_f32m4(w + 56, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi6x0123, vk6x0123, 4), 4);

      const vfloat32m4_t vi7x0123 = vle32_v_f32m4(i7, 4);
      i7 += 4;

      const vfloat32m4_t vk7x0123 = vle32_v_f32m4(w + 64, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi7x0123, vk7x0123, 4), 4);

      const vfloat32m4_t vi8x0123 = vle32_v_f32m4(i8, 4);
      i8 += 4;

      const vfloat32m4_t vk8x0123 = vle32_v_f32m4(w + 72, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi8x0123, vk8x0123, 4), 4);

      const vfloat32m4_t vi9x0123 = vle32_v_f32m4(i9, 4);
      i9 += 4;

      const vfloat32m4_t vk9x0123 = vle32_v_f32m4(w + 80, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi9x0123, vk9x0123, 4), 4);

      const vfloat32m4_t vi10x0123 = vle32_v_f32m4(i10, 4);
      i10 += 4;

      const vfloat32m4_t vk10x0123 = vle32_v_f32m4(w + 88, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi10x0123, vk10x0123, 4), 4);

      const vfloat32m4_t vi11x0123 = vle32_v_f32m4(i11, 4);
      i11 += 4;

      const vfloat32m4_t vk11x0123 = vle32_v_f32m4(w + 96, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi11x0123, vk11x0123, 4), 4);

      const vfloat32m4_t vi12x0123 = vle32_v_f32m4(i12, 4);
      i12 += 4;

      const vfloat32m4_t vk12x0123 = vle32_v_f32m4(w + 104, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi12x0123, vk12x0123, 4), 4);

      const vfloat32m4_t vi13x0123 = vle32_v_f32m4(i13, 4);
      i13 += 4;

      const vfloat32m4_t vk13x0123 = vle32_v_f32m4(w + 112, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi13x0123, vk13x0123, 4), 4);

      const vfloat32m4_t vi14x0123 = vle32_v_f32m4(i14, 4);
      i14 += 4;

      const vfloat32m4_t vk14x0123 = vle32_v_f32m4(w + 120, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi14x0123, vk14x0123, 4), 4);

      const vfloat32m4_t vi15x0123 = vle32_v_f32m4(i15, 4);
      i15 += 4;

      const vfloat32m4_t vk15x0123 = vle32_v_f32m4(w + 128, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi15x0123, vk15x0123, 4), 4);

      const vfloat32m4_t vi16x0123 = vle32_v_f32m4(i16, 4);
      i16 += 4;

      const vfloat32m4_t vk16x0123 = vle32_v_f32m4(w + 136, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi16x0123, vk16x0123, 4), 4);

      const vfloat32m4_t vi17x0123 = vle32_v_f32m4(i17, 4);
      i17 += 4;

      const vfloat32m4_t vk17x0123 = vle32_v_f32m4(w + 144, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi17x0123, vk17x0123, 4), 4);

      const vfloat32m4_t vi18x0123 = vle32_v_f32m4(i18, 4);
      i18 += 4;

      const vfloat32m4_t vk18x0123 = vle32_v_f32m4(w + 152, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi18x0123, vk18x0123, 4), 4);

      const vfloat32m4_t vi19x0123 = vle32_v_f32m4(i19, 4);
      i19 += 4;

      const vfloat32m4_t vk19x0123 = vle32_v_f32m4(w + 160, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi19x0123, vk19x0123, 4), 4);

      const vfloat32m4_t vi20x0123 = vle32_v_f32m4(i20, 4);
      i20 += 4;

      const vfloat32m4_t vk20x0123 = vle32_v_f32m4(w + 168, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi20x0123, vk20x0123, 4), 4);

      const vfloat32m4_t vi21x0123 = vle32_v_f32m4(i21, 4);
      i21 += 4;

      const vfloat32m4_t vk21x0123 = vle32_v_f32m4(w + 176, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi21x0123, vk21x0123, 4), 4);

      const vfloat32m4_t vi22x0123 = vle32_v_f32m4(i22, 4);
      i22 += 4;

      const vfloat32m4_t vk22x0123 = vle32_v_f32m4(w + 184, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi22x0123, vk22x0123, 4), 4);

      const vfloat32m4_t vi23x0123 = vle32_v_f32m4(i23, 4);
      i23 += 4;

      const vfloat32m4_t vk23x0123 = vle32_v_f32m4(w + 192, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi23x0123, vk23x0123, 4), 4);

      const vfloat32m4_t vi24x0123 = vle32_v_f32m4(i24, 4);
      i24 += 4;

      const vfloat32m4_t vk24x0123 = vle32_v_f32m4(w + 200, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi24x0123, vk24x0123, 4), 4);

      w += 4;


      const vfloat32m4_t vacc0123 = vacc0123p0;

      vse32_v_f32m4(output, vacc0123, 4);
      output += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      vfloat32m4_t vacc0123p0 = vle32_v_f32m4(w, 4);

      const vfloat32m4_t vi0x0123 = vle32_v_f32m4(i0, 4);
      const vfloat32m4_t vk0x0123 = vle32_v_f32m4(w + 8, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi0x0123, vk0x0123, 4), 4);

      const vfloat32m4_t vi1x0123 = vle32_v_f32m4(i1, 4);
      const vfloat32m4_t vk1x0123 = vle32_v_f32m4(w + 16, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi1x0123, vk1x0123, 4), 4);

      const vfloat32m4_t vi2x0123 = vle32_v_f32m4(i2, 4);
      const vfloat32m4_t vk2x0123 = vle32_v_f32m4(w + 24, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi2x0123, vk2x0123, 4), 4);

      const vfloat32m4_t vi3x0123 = vle32_v_f32m4(i3, 4);
      const vfloat32m4_t vk3x0123 = vle32_v_f32m4(w + 32, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi3x0123, vk3x0123, 4), 4);

      const vfloat32m4_t vi4x0123 = vle32_v_f32m4(i4, 4);
      const vfloat32m4_t vk4x0123 = vle32_v_f32m4(w + 40, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi4x0123, vk4x0123, 4), 4);

      const vfloat32m4_t vi5x0123 = vle32_v_f32m4(i5, 4);
      const vfloat32m4_t vk5x0123 = vle32_v_f32m4(w + 48, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi5x0123, vk5x0123, 4), 4);

      const vfloat32m4_t vi6x0123 = vle32_v_f32m4(i6, 4);
      const vfloat32m4_t vk6x0123 = vle32_v_f32m4(w + 56, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi6x0123, vk6x0123, 4), 4);

      const vfloat32m4_t vi7x0123 = vle32_v_f32m4(i7, 4);
      const vfloat32m4_t vk7x0123 = vle32_v_f32m4(w + 64, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi7x0123, vk7x0123, 4), 4);

      const vfloat32m4_t vi8x0123 = vle32_v_f32m4(i8, 4);
      const vfloat32m4_t vk8x0123 = vle32_v_f32m4(w + 72, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi8x0123, vk8x0123, 4), 4);

      const vfloat32m4_t vi9x0123 = vle32_v_f32m4(i9, 4);
      const vfloat32m4_t vk9x0123 = vle32_v_f32m4(w + 80, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi9x0123, vk9x0123, 4), 4);

      const vfloat32m4_t vi10x0123 = vle32_v_f32m4(i10, 4);
      const vfloat32m4_t vk10x0123 = vle32_v_f32m4(w + 88, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi10x0123, vk10x0123, 4), 4);

      const vfloat32m4_t vi11x0123 = vle32_v_f32m4(i11, 4);
      const vfloat32m4_t vk11x0123 = vle32_v_f32m4(w + 96, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi11x0123, vk11x0123, 4), 4);

      const vfloat32m4_t vi12x0123 = vle32_v_f32m4(i12, 4);
      const vfloat32m4_t vk12x0123 = vle32_v_f32m4(w + 104, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi12x0123, vk12x0123, 4), 4);

      const vfloat32m4_t vi13x0123 = vle32_v_f32m4(i13, 4);
      const vfloat32m4_t vk13x0123 = vle32_v_f32m4(w + 112, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi13x0123, vk13x0123, 4), 4);

      const vfloat32m4_t vi14x0123 = vle32_v_f32m4(i14, 4);
      const vfloat32m4_t vk14x0123 = vle32_v_f32m4(w + 120, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi14x0123, vk14x0123, 4), 4);

      const vfloat32m4_t vi15x0123 = vle32_v_f32m4(i15, 4);
      const vfloat32m4_t vk15x0123 = vle32_v_f32m4(w + 128, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi15x0123, vk15x0123, 4), 4);

      const vfloat32m4_t vi16x0123 = vle32_v_f32m4(i16, 4);
      const vfloat32m4_t vk16x0123 = vle32_v_f32m4(w + 136, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi16x0123, vk16x0123, 4), 4);

      const vfloat32m4_t vi17x0123 = vle32_v_f32m4(i17, 4);
      const vfloat32m4_t vk17x0123 = vle32_v_f32m4(w + 144, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi17x0123, vk17x0123, 4), 4);

      const vfloat32m4_t vi18x0123 = vle32_v_f32m4(i18, 4);
      const vfloat32m4_t vk18x0123 = vle32_v_f32m4(w + 152, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi18x0123, vk18x0123, 4), 4);

      const vfloat32m4_t vi19x0123 = vle32_v_f32m4(i19, 4);
      const vfloat32m4_t vk19x0123 = vle32_v_f32m4(w + 160, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi19x0123, vk19x0123, 4), 4);

      const vfloat32m4_t vi20x0123 = vle32_v_f32m4(i20, 4);
      const vfloat32m4_t vk20x0123 = vle32_v_f32m4(w + 168, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi20x0123, vk20x0123, 4), 4);

      const vfloat32m4_t vi21x0123 = vle32_v_f32m4(i21, 4);
      const vfloat32m4_t vk21x0123 = vle32_v_f32m4(w + 176, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi21x0123, vk21x0123, 4), 4);

      const vfloat32m4_t vi22x0123 = vle32_v_f32m4(i22, 4);
      const vfloat32m4_t vk22x0123 = vle32_v_f32m4(w + 184, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi22x0123, vk22x0123, 4), 4);

      const vfloat32m4_t vi23x0123 = vle32_v_f32m4(i23, 4);
      const vfloat32m4_t vk23x0123 = vle32_v_f32m4(w + 192, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi23x0123, vk23x0123, 4), 4);

      const vfloat32m4_t vi24x0123 = vle32_v_f32m4(i24, 4);
      const vfloat32m4_t vk24x0123 = vle32_v_f32m4(w + 200, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi24x0123, vk24x0123, 4), 4);


      vfloat32m4_t vacc0123 = vacc0123p0;

      if (c & 2) {
        vse32_v_f32m4(output, vacc0123, 2);
        output += 2;
        vacc0123 = vrgather_vx_f32m4(vacc0123, 2, 1);
      }
      if (c & 1) {
        vse32_v_f32m4(output, vacc0123, 1);
        output += 1;
      }
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_dwconv_minmax_ukernel_25p8c__rvv(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  do {
    const float* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
    }
    const float* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
    }
    const float* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
    }
    const float* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
    }
    const float* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
    }
    const float* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
    }
    const float* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
    }
    const float* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const float*) ((uintptr_t) i7 + input_offset);
    }
    const float* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const float*) ((uintptr_t) i8 + input_offset);
    }
    const float* i9 = input[9];
    assert(i9 != NULL);
    if XNN_UNPREDICTABLE(i9 != zero) {
      i9 = (const float*) ((uintptr_t) i9 + input_offset);
    }
    const float* i10 = input[10];
    assert(i10 != NULL);
    if XNN_UNPREDICTABLE(i10 != zero) {
      i10 = (const float*) ((uintptr_t) i10 + input_offset);
    }
    const float* i11 = input[11];
    assert(i11 != NULL);
    if XNN_UNPREDICTABLE(i11 != zero) {
      i11 = (const float*) ((uintptr_t) i11 + input_offset);
    }
    const float* i12 = input[12];
    assert(i12 != NULL);
    if XNN_UNPREDICTABLE(i12 != zero) {
      i12 = (const float*) ((uintptr_t) i12 + input_offset);
    }
    const float* i13 = input[13];
    assert(i13 != NULL);
    if XNN_UNPREDICTABLE(i13 != zero) {
      i13 = (const float*) ((uintptr_t) i13 + input_offset);
    }
    const float* i14 = input[14];
    assert(i14 != NULL);
    if XNN_UNPREDICTABLE(i14 != zero) {
      i14 = (const float*) ((uintptr_t) i14 + input_offset);
    }
    const float* i15 = input[15];
    assert(i15 != NULL);
    if XNN_UNPREDICTABLE(i15 != zero) {
      i15 = (const float*) ((uintptr_t) i15 + input_offset);
    }
    const float* i16 = input[16];
    assert(i16 != NULL);
    if XNN_UNPREDICTABLE(i16 != zero) {
      i16 = (const float*) ((uintptr_t) i16 + input_offset);
    }
    const float* i17 = input[17];
    assert(i17 != NULL);
    if XNN_UNPREDICTABLE(i17 != zero) {
      i17 = (const float*) ((uintptr_t) i17 + input_offset);
    }
    const float* i18 = input[18];
    assert(i18 != NULL);
    if XNN_UNPREDICTABLE(i18 != zero) {
      i18 = (const float*) ((uintptr_t) i18 + input_offset);
    }
    const float* i19 = input[19];
    assert(i19 != NULL);
    if XNN_UNPREDICTABLE(i19 != zero) {
      i19 = (const float*) ((uintptr_t) i19 + input_offset);
    }
    const float* i20 = input[20];
    assert(i20 != NULL);
    if XNN_UNPREDICTABLE(i20 != zero) {
      i20 = (const float*) ((uintptr_t) i20 + input_offset);
    }
    const float* i21 = input[21];
    assert(i21 != NULL);
    if XNN_UNPREDICTABLE(i21 != zero) {
      i21 = (const float*) ((uintptr_t) i21 + input_offset);
    }
    const float* i22 = input[22];
    assert(i22 != NULL);
    if XNN_UNPREDICTABLE(i22 != zero) {
      i22 = (const float*) ((uintptr_t) i22 + input_offset);
    }
    const float* i23 = input[23];
    assert(i23 != NULL);
    if XNN_UNPREDICTABLE(i23 != zero) {
      i23 = (const float*) ((uintptr_t) i23 + input_offset);
    }
    const float* i24 = input[24];
    assert(i24 != NULL);
    if XNN_UNPREDICTABLE(i24 != zero) {
      i24 = (const float*) ((uintptr_t) i24 + input_offset);
    }
    input = (const float**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* w = weights;
    for (; c >= 8; c -= 8) {
      vfloat32m4_t vacc0123p0 = vle32_v_f32m4(w, 4);
      vfloat32m4_t vacc4567p0 = vle32_v_f32m4(w + 4, 4);

      const vfloat32m4_t vi0x0123 = vle32_v_f32m4(i0, 4);
      const vfloat32m4_t vi0x4567 = vle32_v_f32m4(i0 + 4, 4);
      i0 += 8;

      const vfloat32m4_t vk0x0123 = vle32_v_f32m4(w + 8, 4);
      const vfloat32m4_t vk0x4567 = vle32_v_f32m4(w + 12, 4);

      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi0x0123, vk0x0123, 4), 4);
      vacc4567p0 = vfadd_vv_f32m4(vacc4567p0, vfmul_vv_f32m4(vi0x4567, vk0x4567, 4), 4);

      const vfloat32m4_t vi1x0123 = vle32_v_f32m4(i1, 4);
      const vfloat32m4_t vi1x4567 = vle32_v_f32m4(i1 + 4, 4);
      i1 += 8;

      const vfloat32m4_t vk1x0123 = vle32_v_f32m4(w + 16, 4);
      const vfloat32m4_t vk1x4567 = vle32_v_f32m4(w + 20, 4);

      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi1x0123, vk1x0123, 4), 4);
      vacc4567p0 = vfadd_vv_f32m4(vacc4567p0, vfmul_vv_f32m4(vi1x4567, vk1x4567, 4), 4);

      const vfloat32m4_t vi2x0123 = vle32_v_f32m4(i2, 4);
      const vfloat32m4_t vi2x4567 = vle32_v_f32m4(i2 + 4, 4);
      i2 += 8;

      const vfloat32m4_t vk2x0123 = vle32_v_f32m4(w + 24, 4);
      const vfloat32m4_t vk2x4567 = vle32_v_f32m4(w + 28, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi2x0123, vk2x0123, 4), 4);
      vacc4567p0 = vfadd_vv_f32m4(vacc4567p0, vfmul_vv_f32m4(vi2x4567, vk2x4567, 4), 4);

      const vfloat32m4_t vi3x0123 = vle32_v_f32m4(i3, 4);
      const vfloat32m4_t vi3x4567 = vle32_v_f32m4(i3 + 4, 4);
      i3 += 8;

      const vfloat32m4_t vk3x0123 = vle32_v_f32m4(w + 32, 4);
      const vfloat32m4_t vk3x4567 = vle32_v_f32m4(w + 36, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi3x0123, vk3x0123, 4), 4);
      vacc4567p0 = vfadd_vv_f32m4(vacc4567p0, vfmul_vv_f32m4(vi3x4567, vk3x4567, 4), 4);

      const vfloat32m4_t vi4x0123 = vle32_v_f32m4(i4, 4);
      const vfloat32m4_t vi4x4567 = vle32_v_f32m4(i4 + 4, 4);
      i4 += 8;
  
      const vfloat32m4_t vk4x0123 = vle32_v_f32m4(w + 40, 4);
      const vfloat32m4_t vk4x4567 = vle32_v_f32m4(w + 44, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi4x0123, vk4x0123, 4), 4);
      vacc4567p0 = vfadd_vv_f32m4(vacc4567p0, vfmul_vv_f32m4(vi4x4567, vk4x4567, 4), 4);

      const vfloat32m4_t vi5x0123 = vle32_v_f32m4(i5, 4);
      const vfloat32m4_t vi5x4567 = vle32_v_f32m4(i5 + 4, 4);
      i5 += 8;

      const vfloat32m4_t vk5x0123 = vle32_v_f32m4(w + 48, 4);
      const vfloat32m4_t vk5x4567 = vle32_v_f32m4(w + 52, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi5x0123, vk5x0123, 4), 4);
      vacc4567p0 = vfadd_vv_f32m4(vacc4567p0, vfmul_vv_f32m4(vi5x4567, vk5x4567, 4), 4);

      const vfloat32m4_t vi6x0123 = vle32_v_f32m4(i6, 4);
      const vfloat32m4_t vi6x4567 = vle32_v_f32m4(i6 + 4, 4);
      i6 += 8;

      const vfloat32m4_t vk6x0123 = vle32_v_f32m4(w + 56, 4);
      const vfloat32m4_t vk6x4567 = vle32_v_f32m4(w + 60, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi6x0123, vk6x0123, 4), 4);
      vacc4567p0 = vfadd_vv_f32m4(vacc4567p0, vfmul_vv_f32m4(vi6x4567, vk6x4567, 4), 4);

      const vfloat32m4_t vi7x0123 = vle32_v_f32m4(i7, 4);
      const vfloat32m4_t vi7x4567 = vle32_v_f32m4(i7 + 4, 4);
      i7 += 8;

      const vfloat32m4_t vk7x0123 = vle32_v_f32m4(w + 64, 4);
      const vfloat32m4_t vk7x4567 = vle32_v_f32m4(w + 68, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi7x0123, vk7x0123, 4), 4);
      vacc4567p0 = vfadd_vv_f32m4(vacc4567p0, vfmul_vv_f32m4(vi7x4567, vk7x4567, 4), 4);

      const vfloat32m4_t vi8x0123 = vle32_v_f32m4(i8, 4);
      const vfloat32m4_t vi8x4567 = vle32_v_f32m4(i8 + 4, 4);
      i8 += 8;

      const vfloat32m4_t vk8x0123 = vle32_v_f32m4(w + 72, 4);
      const vfloat32m4_t vk8x4567 = vle32_v_f32m4(w + 76, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi8x0123, vk8x0123, 4), 4);
      vacc4567p0 = vfadd_vv_f32m4(vacc4567p0, vfmul_vv_f32m4(vi8x4567, vk8x4567, 4), 4);

      const vfloat32m4_t vi9x0123 = vle32_v_f32m4(i9, 4);
      const vfloat32m4_t vi9x4567 = vle32_v_f32m4(i9 + 4, 4);
      i9 += 8;

      const vfloat32m4_t vk9x0123 = vle32_v_f32m4(w + 80, 4);
      const vfloat32m4_t vk9x4567 = vle32_v_f32m4(w + 84, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi9x0123, vk9x0123, 4), 4);
      vacc4567p0 = vfadd_vv_f32m4(vacc4567p0, vfmul_vv_f32m4(vi9x4567, vk9x4567, 4), 4);

      const vfloat32m4_t vi10x0123 = vle32_v_f32m4(i10, 4);
      const vfloat32m4_t vi10x4567 = vle32_v_f32m4(i10 + 4, 4);
      i10 += 8;

      const vfloat32m4_t vk10x0123 = vle32_v_f32m4(w + 88, 4);
      const vfloat32m4_t vk10x4567 = vle32_v_f32m4(w + 92, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi10x0123, vk10x0123, 4), 4);
      vacc4567p0 = vfadd_vv_f32m4(vacc4567p0, vfmul_vv_f32m4(vi10x4567, vk10x4567, 4), 4);

      const vfloat32m4_t vi11x0123 = vle32_v_f32m4(i11, 4);
      const vfloat32m4_t vi11x4567 = vle32_v_f32m4(i11 + 4, 4);
      i11 += 8;

      const vfloat32m4_t vk11x0123 = vle32_v_f32m4(w + 96, 4);
      const vfloat32m4_t vk11x4567 = vle32_v_f32m4(w + 100, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi11x0123, vk11x0123, 4), 4);
      vacc4567p0 = vfadd_vv_f32m4(vacc4567p0, vfmul_vv_f32m4(vi11x4567, vk11x4567, 4), 4);

      const vfloat32m4_t vi12x0123 = vle32_v_f32m4(i12, 4);
      const vfloat32m4_t vi12x4567 = vle32_v_f32m4(i12 + 4, 4);
      i12 += 8;

      const vfloat32m4_t vk12x0123 = vle32_v_f32m4(w + 104, 4);
      const vfloat32m4_t vk12x4567 = vle32_v_f32m4(w + 108, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi12x0123, vk12x0123, 4), 4);
      vacc4567p0 = vfadd_vv_f32m4(vacc4567p0, vfmul_vv_f32m4(vi12x4567, vk12x4567, 4), 4);

      const vfloat32m4_t vi13x0123 = vle32_v_f32m4(i13, 4);
      const vfloat32m4_t vi13x4567 = vle32_v_f32m4(i13 + 4, 4);
      i13 += 8;

      const vfloat32m4_t vk13x0123 = vle32_v_f32m4(w + 112, 4);
      const vfloat32m4_t vk13x4567 = vle32_v_f32m4(w + 116, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi13x0123, vk13x0123, 4), 4);
      vacc4567p0 = vfadd_vv_f32m4(vacc4567p0, vfmul_vv_f32m4(vi13x4567, vk13x4567, 4), 4);

      const vfloat32m4_t vi14x0123 = vle32_v_f32m4(i14, 4);
      const vfloat32m4_t vi14x4567 = vle32_v_f32m4(i14 + 4, 4);
      i14 += 8;

      const vfloat32m4_t vk14x0123 = vle32_v_f32m4(w + 120, 4);
      const vfloat32m4_t vk14x4567 = vle32_v_f32m4(w + 124, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi14x0123, vk14x0123, 4), 4);
      vacc4567p0 = vfadd_vv_f32m4(vacc4567p0, vfmul_vv_f32m4(vi14x4567, vk14x4567, 4), 4);

      const vfloat32m4_t vi15x0123 = vle32_v_f32m4(i15, 4);
      const vfloat32m4_t vi15x4567 = vle32_v_f32m4(i15 + 4, 4);
      i15 += 8;

      const vfloat32m4_t vk15x0123 = vle32_v_f32m4(w + 128, 4);
      const vfloat32m4_t vk15x4567 = vle32_v_f32m4(w + 132, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi15x0123, vk15x0123, 4), 4);
      vacc4567p0 = vfadd_vv_f32m4(vacc4567p0, vfmul_vv_f32m4(vi15x4567, vk15x4567, 4), 4);

      const vfloat32m4_t vi16x0123 = vle32_v_f32m4(i16, 4);
      const vfloat32m4_t vi16x4567 = vle32_v_f32m4(i16 + 4, 4);
      i16 += 8;

      const vfloat32m4_t vk16x0123 = vle32_v_f32m4(w + 136, 4);
      const vfloat32m4_t vk16x4567 = vle32_v_f32m4(w + 140, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi16x0123, vk16x0123, 4), 4);
      vacc4567p0 = vfadd_vv_f32m4(vacc4567p0, vfmul_vv_f32m4(vi16x4567, vk16x4567, 4), 4);

      const vfloat32m4_t vi17x0123 = vle32_v_f32m4(i17, 4);
      const vfloat32m4_t vi17x4567 = vle32_v_f32m4(i17 + 4, 4);
      i17 += 8;

      const vfloat32m4_t vk17x0123 = vle32_v_f32m4(w + 144, 4);
      const vfloat32m4_t vk17x4567 = vle32_v_f32m4(w + 148, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi17x0123, vk17x0123, 4), 4);
      vacc4567p0 = vfadd_vv_f32m4(vacc4567p0, vfmul_vv_f32m4(vi17x4567, vk17x4567, 4), 4);

      const vfloat32m4_t vi18x0123 = vle32_v_f32m4(i18, 4);
      const vfloat32m4_t vi18x4567 = vle32_v_f32m4(i18 + 4, 4);
      i18 += 8;

      const vfloat32m4_t vk18x0123 = vle32_v_f32m4(w + 152, 4);
      const vfloat32m4_t vk18x4567 = vle32_v_f32m4(w + 156, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi18x0123, vk18x0123, 4), 4);
      vacc4567p0 = vfadd_vv_f32m4(vacc4567p0, vfmul_vv_f32m4(vi18x4567, vk18x4567, 4), 4);

      const vfloat32m4_t vi19x0123 = vle32_v_f32m4(i19, 4);
      const vfloat32m4_t vi19x4567 = vle32_v_f32m4(i19 + 4, 4);
      i19 += 8;

      const vfloat32m4_t vk19x0123 = vle32_v_f32m4(w + 160, 4);
      const vfloat32m4_t vk19x4567 = vle32_v_f32m4(w + 164, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi19x0123, vk19x0123, 4), 4);
      vacc4567p0 = vfadd_vv_f32m4(vacc4567p0, vfmul_vv_f32m4(vi19x4567, vk19x4567, 4), 4);

      const vfloat32m4_t vi20x0123 = vle32_v_f32m4(i20, 4);
      const vfloat32m4_t vi20x4567 = vle32_v_f32m4(i20 + 4, 4);
      i20 += 8;

      const vfloat32m4_t vk20x0123 = vle32_v_f32m4(w + 168, 4);
      const vfloat32m4_t vk20x4567 = vle32_v_f32m4(w + 172, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi20x0123, vk20x0123, 4), 4);
      vacc4567p0 = vfadd_vv_f32m4(vacc4567p0, vfmul_vv_f32m4(vi20x4567, vk20x4567, 4), 4);

      const vfloat32m4_t vi21x0123 = vle32_v_f32m4(i21, 4);
      const vfloat32m4_t vi21x4567 = vle32_v_f32m4(i21 + 4, 4);
      i21 += 8;

      const vfloat32m4_t vk21x0123 = vle32_v_f32m4(w + 176, 4);
      const vfloat32m4_t vk21x4567 = vle32_v_f32m4(w + 180, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi21x0123, vk21x0123, 4), 4);
      vacc4567p0 = vfadd_vv_f32m4(vacc4567p0, vfmul_vv_f32m4(vi21x4567, vk21x4567, 4), 4);

      const vfloat32m4_t vi22x0123 = vle32_v_f32m4(i22, 4);
      const vfloat32m4_t vi22x4567 = vle32_v_f32m4(i22 + 4, 4);
      i22 += 8;

      const vfloat32m4_t vk22x0123 = vle32_v_f32m4(w + 184, 4);
      const vfloat32m4_t vk22x4567 = vle32_v_f32m4(w + 188, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi22x0123, vk22x0123, 4), 4);
      vacc4567p0 = vfadd_vv_f32m4(vacc4567p0, vfmul_vv_f32m4(vi22x4567, vk22x4567, 4), 4);

      const vfloat32m4_t vi23x0123 = vle32_v_f32m4(i23, 4);
      const vfloat32m4_t vi23x4567 = vle32_v_f32m4(i23 + 4, 4);
      i23 += 8;

      const vfloat32m4_t vk23x0123 = vle32_v_f32m4(w + 192, 4);
      const vfloat32m4_t vk23x4567 = vle32_v_f32m4(w + 196, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi23x0123, vk23x0123, 4), 4);
      vacc4567p0 = vfadd_vv_f32m4(vacc4567p0, vfmul_vv_f32m4(vi23x4567, vk23x4567, 4), 4);

      const vfloat32m4_t vi24x0123 = vle32_v_f32m4(i24, 4);
      const vfloat32m4_t vi24x4567 = vle32_v_f32m4(i24 + 4, 4);
      i24 += 8;

      const vfloat32m4_t vk24x0123 = vle32_v_f32m4(w + 200, 4);
      const vfloat32m4_t vk24x4567 = vle32_v_f32m4(w + 204, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi24x0123, vk24x0123, 4), 4);
      vacc4567p0 = vfadd_vv_f32m4(vacc4567p0, vfmul_vv_f32m4(vi24x4567, vk24x4567, 4), 4);

      w += 208;


      vfloat32m4_t vacc0123 = vfmax_vf_f32m4(vacc0123p0, vmin, 4);
      vfloat32m4_t vacc4567 = vfmax_vf_f32m4(vacc4567p0, vmin, 4);

      vacc0123 = vfmin_vf_f32m4(vacc0123, vmax, 4);
      vacc4567 = vfmin_vf_f32m4(vacc4567, vmax, 4);

      vse32_v_f32m4(output, vacc0123, 4);
      vse32_v_f32m4(output + 4, vacc4567, 4);
      output += 8;
    }
    for (; c >= 4; c -= 4) {
      vfloat32m4_t vacc0123p0 = vle32_v_f32m4(w, 4);

      const vfloat32m4_t vi0x0123 = vle32_v_f32m4(i0, 4);
      i0 += 4;

      const vfloat32m4_t vk0x0123 = vle32_v_f32m4(w + 8, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi0x0123, vk0x0123, 4), 4);

      const vfloat32m4_t vi1x0123 = vle32_v_f32m4(i1, 4);
      i1 += 4;

      const vfloat32m4_t vk1x0123 = vle32_v_f32m4(w + 16, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi1x0123, vk1x0123, 4), 4);

      const vfloat32m4_t vi2x0123 = vle32_v_f32m4(i2, 4);
      i2 += 4;

      const vfloat32m4_t vk2x0123 = vle32_v_f32m4(w + 24, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi2x0123, vk2x0123, 4), 4);

      const vfloat32m4_t vi3x0123 = vle32_v_f32m4(i3, 4);
      i3 += 4;

      const vfloat32m4_t vk3x0123 = vle32_v_f32m4(w + 32, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi3x0123, vk3x0123, 4), 4);

      const vfloat32m4_t vi4x0123 = vle32_v_f32m4(i4, 4);
      i4 += 4;

      const vfloat32m4_t vk4x0123 = vle32_v_f32m4(w + 40, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi4x0123, vk4x0123, 4), 4);

      const vfloat32m4_t vi5x0123 = vle32_v_f32m4(i5, 4);
      i5 += 4;

      const vfloat32m4_t vk5x0123 = vle32_v_f32m4(w + 48, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi5x0123, vk5x0123, 4), 4);

      const vfloat32m4_t vi6x0123 = vle32_v_f32m4(i6, 4);
      i6 += 4;

      const vfloat32m4_t vk6x0123 = vle32_v_f32m4(w + 56, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi6x0123, vk6x0123, 4), 4);

      const vfloat32m4_t vi7x0123 = vle32_v_f32m4(i7, 4);
      i7 += 4;

      const vfloat32m4_t vk7x0123 = vle32_v_f32m4(w + 64, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi7x0123, vk7x0123, 4), 4);

      const vfloat32m4_t vi8x0123 = vle32_v_f32m4(i8, 4);
      i8 += 4;

      const vfloat32m4_t vk8x0123 = vle32_v_f32m4(w + 72, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi8x0123, vk8x0123, 4), 4);

      const vfloat32m4_t vi9x0123 = vle32_v_f32m4(i9, 4);
      i9 += 4;

      const vfloat32m4_t vk9x0123 = vle32_v_f32m4(w + 80, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi9x0123, vk9x0123, 4), 4);

      const vfloat32m4_t vi10x0123 = vle32_v_f32m4(i10, 4);
      i10 += 4;

      const vfloat32m4_t vk10x0123 = vle32_v_f32m4(w + 88, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi10x0123, vk10x0123, 4), 4);

      const vfloat32m4_t vi11x0123 = vle32_v_f32m4(i11, 4);
      i11 += 4;

      const vfloat32m4_t vk11x0123 = vle32_v_f32m4(w + 96, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi11x0123, vk11x0123, 4), 4);

      const vfloat32m4_t vi12x0123 = vle32_v_f32m4(i12, 4);
      i12 += 4;

      const vfloat32m4_t vk12x0123 = vle32_v_f32m4(w + 104, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi12x0123, vk12x0123, 4), 4);

      const vfloat32m4_t vi13x0123 = vle32_v_f32m4(i13, 4);
      i13 += 4;

      const vfloat32m4_t vk13x0123 = vle32_v_f32m4(w + 112, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi13x0123, vk13x0123, 4), 4);

      const vfloat32m4_t vi14x0123 = vle32_v_f32m4(i14, 4);
      i14 += 4;

      const vfloat32m4_t vk14x0123 = vle32_v_f32m4(w + 120, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi14x0123, vk14x0123, 4), 4);

      const vfloat32m4_t vi15x0123 = vle32_v_f32m4(i15, 4);
      i15 += 4;

      const vfloat32m4_t vk15x0123 = vle32_v_f32m4(w + 128, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi15x0123, vk15x0123, 4), 4);

      const vfloat32m4_t vi16x0123 = vle32_v_f32m4(i16, 4);
      i16 += 4;

      const vfloat32m4_t vk16x0123 = vle32_v_f32m4(w + 136, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi16x0123, vk16x0123, 4), 4);

      const vfloat32m4_t vi17x0123 = vle32_v_f32m4(i17, 4);
      i17 += 4;

      const vfloat32m4_t vk17x0123 = vle32_v_f32m4(w + 144, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi17x0123, vk17x0123, 4), 4);

      const vfloat32m4_t vi18x0123 = vle32_v_f32m4(i18, 4);
      i18 += 4;

      const vfloat32m4_t vk18x0123 = vle32_v_f32m4(w + 152, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi18x0123, vk18x0123, 4), 4);

      const vfloat32m4_t vi19x0123 = vle32_v_f32m4(i19, 4);
      i19 += 4;

      const vfloat32m4_t vk19x0123 = vle32_v_f32m4(w + 160, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi19x0123, vk19x0123, 4), 4);

      const vfloat32m4_t vi20x0123 = vle32_v_f32m4(i20, 4);
      i20 += 4;

      const vfloat32m4_t vk20x0123 = vle32_v_f32m4(w + 168, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi20x0123, vk20x0123, 4), 4);

      const vfloat32m4_t vi21x0123 = vle32_v_f32m4(i21, 4);
      i21 += 4;

      const vfloat32m4_t vk21x0123 = vle32_v_f32m4(w + 176, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi21x0123, vk21x0123, 4), 4);

      const vfloat32m4_t vi22x0123 = vle32_v_f32m4(i22, 4);
      i22 += 4;

      const vfloat32m4_t vk22x0123 = vle32_v_f32m4(w + 184, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi22x0123, vk22x0123, 4), 4);

      const vfloat32m4_t vi23x0123 = vle32_v_f32m4(i23, 4);
      i23 += 4;

      const vfloat32m4_t vk23x0123 = vle32_v_f32m4(w + 192, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi23x0123, vk23x0123, 4), 4);

      const vfloat32m4_t vi24x0123 = vle32_v_f32m4(i24, 4);
      i24 += 4;

      const vfloat32m4_t vk24x0123 = vle32_v_f32m4(w + 200, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi24x0123, vk24x0123, 4), 4);

      w += 4;


      vfloat32m4_t vacc0123 = vfmax_vf_f32m4(vacc0123p0, vmin, 4);
      vacc0123 = vfmin_vf_f32m4(vacc0123, vmax, 4);

      vse32_v_f32m4(output, vacc0123, 4);
      output += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      vfloat32m4_t vacc0123p0 = vle32_v_f32m4(w, 4);

      const vfloat32m4_t vi0x0123 = vle32_v_f32m4(i0, 4);
      const vfloat32m4_t vk0x0123 = vle32_v_f32m4(w + 8, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi0x0123, vk0x0123, 4), 4);

      const vfloat32m4_t vi1x0123 = vle32_v_f32m4(i1, 4);
      const vfloat32m4_t vk1x0123 = vle32_v_f32m4(w + 16, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi1x0123, vk1x0123, 4), 4);

      const vfloat32m4_t vi2x0123 = vle32_v_f32m4(i2, 4);
      const vfloat32m4_t vk2x0123 = vle32_v_f32m4(w + 24, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi2x0123, vk2x0123, 4), 4);

      const vfloat32m4_t vi3x0123 = vle32_v_f32m4(i3, 4);
      const vfloat32m4_t vk3x0123 = vle32_v_f32m4(w + 32, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi3x0123, vk3x0123, 4), 4);

      const vfloat32m4_t vi4x0123 = vle32_v_f32m4(i4, 4);
      const vfloat32m4_t vk4x0123 = vle32_v_f32m4(w + 40, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi4x0123, vk4x0123, 4), 4);

      const vfloat32m4_t vi5x0123 = vle32_v_f32m4(i5, 4);
      const vfloat32m4_t vk5x0123 = vle32_v_f32m4(w + 48, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi5x0123, vk5x0123, 4), 4);

      const vfloat32m4_t vi6x0123 = vle32_v_f32m4(i6, 4);
      const vfloat32m4_t vk6x0123 = vle32_v_f32m4(w + 56, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi6x0123, vk6x0123, 4), 4);

      const vfloat32m4_t vi7x0123 = vle32_v_f32m4(i7, 4);
      const vfloat32m4_t vk7x0123 = vle32_v_f32m4(w + 64, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi7x0123, vk7x0123, 4), 4);

      const vfloat32m4_t vi8x0123 = vle32_v_f32m4(i8, 4);
      const vfloat32m4_t vk8x0123 = vle32_v_f32m4(w + 72, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi8x0123, vk8x0123, 4), 4);

      const vfloat32m4_t vi9x0123 = vle32_v_f32m4(i9, 4);
      const vfloat32m4_t vk9x0123 = vle32_v_f32m4(w + 80, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi9x0123, vk9x0123, 4), 4);

      const vfloat32m4_t vi10x0123 = vle32_v_f32m4(i10, 4);
      const vfloat32m4_t vk10x0123 = vle32_v_f32m4(w + 88, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi10x0123, vk10x0123, 4), 4);

      const vfloat32m4_t vi11x0123 = vle32_v_f32m4(i11, 4);
      const vfloat32m4_t vk11x0123 = vle32_v_f32m4(w + 96, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi11x0123, vk11x0123, 4), 4);

      const vfloat32m4_t vi12x0123 = vle32_v_f32m4(i12, 4);
      const vfloat32m4_t vk12x0123 = vle32_v_f32m4(w + 104, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi12x0123, vk12x0123, 4), 4);

      const vfloat32m4_t vi13x0123 = vle32_v_f32m4(i13, 4);
      const vfloat32m4_t vk13x0123 = vle32_v_f32m4(w + 112, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi13x0123, vk13x0123, 4), 4);

      const vfloat32m4_t vi14x0123 = vle32_v_f32m4(i14, 4);
      const vfloat32m4_t vk14x0123 = vle32_v_f32m4(w + 120, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi14x0123, vk14x0123, 4), 4);

      const vfloat32m4_t vi15x0123 = vle32_v_f32m4(i15, 4);
      const vfloat32m4_t vk15x0123 = vle32_v_f32m4(w + 128, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi15x0123, vk15x0123, 4), 4);

      const vfloat32m4_t vi16x0123 = vle32_v_f32m4(i16, 4);
      const vfloat32m4_t vk16x0123 = vle32_v_f32m4(w + 136, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi16x0123, vk16x0123, 4), 4);

      const vfloat32m4_t vi17x0123 = vle32_v_f32m4(i17, 4);
      const vfloat32m4_t vk17x0123 = vle32_v_f32m4(w + 144, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi17x0123, vk17x0123, 4), 4);

      const vfloat32m4_t vi18x0123 = vle32_v_f32m4(i18, 4);
      const vfloat32m4_t vk18x0123 = vle32_v_f32m4(w + 152, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi18x0123, vk18x0123, 4), 4);

      const vfloat32m4_t vi19x0123 = vle32_v_f32m4(i19, 4);
      const vfloat32m4_t vk19x0123 = vle32_v_f32m4(w + 160, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi19x0123, vk19x0123, 4), 4);

      const vfloat32m4_t vi20x0123 = vle32_v_f32m4(i20, 4);
      const vfloat32m4_t vk20x0123 = vle32_v_f32m4(w + 168, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi20x0123, vk20x0123, 4), 4);

      const vfloat32m4_t vi21x0123 = vle32_v_f32m4(i21, 4);
      const vfloat32m4_t vk21x0123 = vle32_v_f32m4(w + 176, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi21x0123, vk21x0123, 4), 4);

      const vfloat32m4_t vi22x0123 = vle32_v_f32m4(i22, 4);
      const vfloat32m4_t vk22x0123 = vle32_v_f32m4(w + 184, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi22x0123, vk22x0123, 4), 4);

      const vfloat32m4_t vi23x0123 = vle32_v_f32m4(i23, 4);
      const vfloat32m4_t vk23x0123 = vle32_v_f32m4(w + 192, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi23x0123, vk23x0123, 4), 4);

      const vfloat32m4_t vi24x0123 = vle32_v_f32m4(i24, 4);
      const vfloat32m4_t vk24x0123 = vle32_v_f32m4(w + 200, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi24x0123, vk24x0123, 4), 4);


      vfloat32m4_t vacc0123 = vfmax_vf_f32m4(vacc0123p0, vmin, 4);
      vacc0123 = vfmin_vf_f32m4(vacc0123, vmax, 4);

      if (c & 2) {
        vse32_v_f32m4(output, vacc0123, 2);
        output += 2;
        vacc0123 = vrgather_vx_f32m4(vacc0123, 2, 1);
      }
      if (c & 1) {
        vse32_v_f32m4(output, vacc0123, 1);
        output += 1;
      }
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_dwconv_minmax_ukernel_25p4c__rvv(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  do {
    const float* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
    }
    const float* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
    }
    const float* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
    }
    const float* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
    }
    const float* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
    }
    const float* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
    }
    const float* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
    }
    const float* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const float*) ((uintptr_t) i7 + input_offset);
    }
    const float* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const float*) ((uintptr_t) i8 + input_offset);
    }
    const float* i9 = input[9];
    assert(i9 != NULL);
    if XNN_UNPREDICTABLE(i9 != zero) {
      i9 = (const float*) ((uintptr_t) i9 + input_offset);
    }
    const float* i10 = input[10];
    assert(i10 != NULL);
    if XNN_UNPREDICTABLE(i10 != zero) {
      i10 = (const float*) ((uintptr_t) i10 + input_offset);
    }
    const float* i11 = input[11];
    assert(i11 != NULL);
    if XNN_UNPREDICTABLE(i11 != zero) {
      i11 = (const float*) ((uintptr_t) i11 + input_offset);
    }
    const float* i12 = input[12];
    assert(i12 != NULL);
    if XNN_UNPREDICTABLE(i12 != zero) {
      i12 = (const float*) ((uintptr_t) i12 + input_offset);
    }
    const float* i13 = input[13];
    assert(i13 != NULL);
    if XNN_UNPREDICTABLE(i13 != zero) {
      i13 = (const float*) ((uintptr_t) i13 + input_offset);
    }
    const float* i14 = input[14];
    assert(i14 != NULL);
    if XNN_UNPREDICTABLE(i14 != zero) {
      i14 = (const float*) ((uintptr_t) i14 + input_offset);
    }
    const float* i15 = input[15];
    assert(i15 != NULL);
    if XNN_UNPREDICTABLE(i15 != zero) {
      i15 = (const float*) ((uintptr_t) i15 + input_offset);
    }
    const float* i16 = input[16];
    assert(i16 != NULL);
    if XNN_UNPREDICTABLE(i16 != zero) {
      i16 = (const float*) ((uintptr_t) i16 + input_offset);
    }
    const float* i17 = input[17];
    assert(i17 != NULL);
    if XNN_UNPREDICTABLE(i17 != zero) {
      i17 = (const float*) ((uintptr_t) i17 + input_offset);
    }
    const float* i18 = input[18];
    assert(i18 != NULL);
    if XNN_UNPREDICTABLE(i18 != zero) {
      i18 = (const float*) ((uintptr_t) i18 + input_offset);
    }
    const float* i19 = input[19];
    assert(i19 != NULL);
    if XNN_UNPREDICTABLE(i19 != zero) {
      i19 = (const float*) ((uintptr_t) i19 + input_offset);
    }
    const float* i20 = input[20];
    assert(i20 != NULL);
    if XNN_UNPREDICTABLE(i20 != zero) {
      i20 = (const float*) ((uintptr_t) i20 + input_offset);
    }
    const float* i21 = input[21];
    assert(i21 != NULL);
    if XNN_UNPREDICTABLE(i21 != zero) {
      i21 = (const float*) ((uintptr_t) i21 + input_offset);
    }
    const float* i22 = input[22];
    assert(i22 != NULL);
    if XNN_UNPREDICTABLE(i22 != zero) {
      i22 = (const float*) ((uintptr_t) i22 + input_offset);
    }
    const float* i23 = input[23];
    assert(i23 != NULL);
    if XNN_UNPREDICTABLE(i23 != zero) {
      i23 = (const float*) ((uintptr_t) i23 + input_offset);
    }
    const float* i24 = input[24];
    assert(i24 != NULL);
    if XNN_UNPREDICTABLE(i24 != zero) {
      i24 = (const float*) ((uintptr_t) i24 + input_offset);
    }
    input = (const float**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* w = weights;
    for (; c >= 4; c -= 4) {
      vfloat32m4_t vacc0123p0 = vle32_v_f32m4(w, 4);


      const vfloat32m4_t vi0x0123 = vle32_v_f32m4(i0, 4);
      i0 += 4;

      const vfloat32m4_t vk0x0123 = vle32_v_f32m4(w + 4, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi0x0123, vk0x0123, 4), 4);

      const vfloat32m4_t vi1x0123 = vle32_v_f32m4(i1, 4);
      i1 += 4;

      const vfloat32m4_t vk1x0123 = vle32_v_f32m4(w + 8, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi1x0123, vk1x0123, 4), 4);

      const vfloat32m4_t vi2x0123 = vle32_v_f32m4(i2, 4);
      i2 += 4;

      const vfloat32m4_t vk2x0123 = vle32_v_f32m4(w + 12, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi2x0123, vk2x0123, 4), 4);

      const vfloat32m4_t vi3x0123 = vle32_v_f32m4(i3, 4);
      i3 += 4;

      const vfloat32m4_t vk3x0123 = vle32_v_f32m4(w + 16, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi3x0123, vk3x0123, 4), 4);

      const vfloat32m4_t vi4x0123 = vle32_v_f32m4(i4, 4);
      i4 += 4;

      const vfloat32m4_t vk4x0123 = vle32_v_f32m4(w + 20, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi4x0123, vk4x0123, 4), 4);

      const vfloat32m4_t vi5x0123 = vle32_v_f32m4(i5, 4);
      i5 += 4;

      const vfloat32m4_t vk5x0123 = vle32_v_f32m4(w + 24, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi5x0123, vk5x0123, 4), 4);

      const vfloat32m4_t vi6x0123 = vle32_v_f32m4(i6, 4);
      i6 += 4;

      const vfloat32m4_t vk6x0123 = vle32_v_f32m4(w + 28, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi6x0123, vk6x0123, 4), 4);

      const vfloat32m4_t vi7x0123 = vle32_v_f32m4(i7, 4);
      i7 += 4;

      const vfloat32m4_t vk7x0123 = vle32_v_f32m4(w + 32, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi7x0123, vk7x0123, 4), 4);

      const vfloat32m4_t vi8x0123 = vle32_v_f32m4(i8, 4);
      i8 += 4;

      const vfloat32m4_t vk8x0123 = vle32_v_f32m4(w + 36, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi8x0123, vk8x0123, 4), 4);

      const vfloat32m4_t vi9x0123 = vle32_v_f32m4(i9, 4);
      i9 += 4;

      const vfloat32m4_t vk9x0123 = vle32_v_f32m4(w + 40, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi9x0123, vk9x0123, 4), 4);

      const vfloat32m4_t vi10x0123 = vle32_v_f32m4(i10, 4);
      i10 += 4;

      const vfloat32m4_t vk10x0123 = vle32_v_f32m4(w + 44, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi10x0123, vk10x0123, 4), 4);

      const vfloat32m4_t vi11x0123 = vle32_v_f32m4(i11, 4);
      i11 += 4;

      const vfloat32m4_t vk11x0123 = vle32_v_f32m4(w + 48, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi11x0123, vk11x0123, 4), 4);

      const vfloat32m4_t vi12x0123 = vle32_v_f32m4(i12, 4);
      i12 += 4;

      const vfloat32m4_t vk12x0123 = vle32_v_f32m4(w + 52, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi12x0123, vk12x0123, 4), 4);

      const vfloat32m4_t vi13x0123 = vle32_v_f32m4(i13, 4);
      i13 += 4;

      const vfloat32m4_t vk13x0123 = vle32_v_f32m4(w + 56, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi13x0123, vk13x0123, 4), 4);

      const vfloat32m4_t vi14x0123 = vle32_v_f32m4(i14, 4);
      i14 += 4;

      const vfloat32m4_t vk14x0123 = vle32_v_f32m4(w + 60, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi14x0123, vk14x0123, 4), 4);

      const vfloat32m4_t vi15x0123 = vle32_v_f32m4(i15, 4);
      i15 += 4;

      const vfloat32m4_t vk15x0123 = vle32_v_f32m4(w + 64, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi15x0123, vk15x0123, 4), 4);

      const vfloat32m4_t vi16x0123 = vle32_v_f32m4(i16, 4);
      i16 += 4;

      const vfloat32m4_t vk16x0123 = vle32_v_f32m4(w + 68, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi16x0123, vk16x0123, 4), 4);

      const vfloat32m4_t vi17x0123 = vle32_v_f32m4(i17, 4);
      i17 += 4;

      const vfloat32m4_t vk17x0123 = vle32_v_f32m4(w + 72, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi17x0123, vk17x0123, 4), 4);

      const vfloat32m4_t vi18x0123 = vle32_v_f32m4(i18, 4);
      i18 += 4;

      const vfloat32m4_t vk18x0123 = vle32_v_f32m4(w + 76, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi18x0123, vk18x0123, 4), 4);

      const vfloat32m4_t vi19x0123 = vle32_v_f32m4(i19, 4);
      i19 += 4;

      const vfloat32m4_t vk19x0123 = vle32_v_f32m4(w + 80, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi19x0123, vk19x0123, 4), 4);

      const vfloat32m4_t vi20x0123 = vle32_v_f32m4(i20, 4);
      i20 += 4;

      const vfloat32m4_t vk20x0123 = vle32_v_f32m4(w + 84, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi20x0123, vk20x0123, 4), 4);

      const vfloat32m4_t vi21x0123 = vle32_v_f32m4(i21, 4);
      i21 += 4;

      const vfloat32m4_t vk21x0123 = vle32_v_f32m4(w + 88, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi21x0123, vk21x0123, 4), 4);

      const vfloat32m4_t vi22x0123 = vle32_v_f32m4(i22, 4);
      i22 += 4;

      const vfloat32m4_t vk22x0123 = vle32_v_f32m4(w + 92, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi22x0123, vk22x0123, 4), 4);

      const vfloat32m4_t vi23x0123 = vle32_v_f32m4(i23, 4);
      i23 += 4;

      const vfloat32m4_t vk23x0123 = vle32_v_f32m4(w + 96, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi23x0123, vk23x0123, 4), 4);

      const vfloat32m4_t vi24x0123 = vle32_v_f32m4(i24, 4);
      i24 += 4;

      const vfloat32m4_t vk24x0123 = vle32_v_f32m4(w + 100, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi24x0123, vk24x0123, 4), 4);

      w += 104;

      vfloat32m4_t vacc0123 = vfmax_vf_f32m4(vacc0123p0, vmin, 4);

      vacc0123 = vfmin_vf_f32m4(vacc0123, vmax, 4);

      vse32_v_f32m4(output, vacc0123, 4);
      output += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      vfloat32m4_t vacc0123p0 = vle32_v_f32m4(w, 4);

      const vfloat32m4_t vi0x0123 = vle32_v_f32m4(i0, 4);
      const vfloat32m4_t vk0x0123 = vle32_v_f32m4(w + 4, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi0x0123, vk0x0123, 4), 4);

      const vfloat32m4_t vi1x0123 = vle32_v_f32m4(i1, 4);
      const vfloat32m4_t vk1x0123 = vle32_v_f32m4(w + 8, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi1x0123, vk1x0123, 4), 4);

      const vfloat32m4_t vi2x0123 = vle32_v_f32m4(i2, 4);
      const vfloat32m4_t vk2x0123 = vle32_v_f32m4(w + 12, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi2x0123, vk2x0123, 4), 4);

      const vfloat32m4_t vi3x0123 = vle32_v_f32m4(i3, 4);
      const vfloat32m4_t vk3x0123 = vle32_v_f32m4(w + 16, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi3x0123, vk3x0123, 4), 4);

      const vfloat32m4_t vi4x0123 = vle32_v_f32m4(i4, 4);
      const vfloat32m4_t vk4x0123 = vle32_v_f32m4(w + 20, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi4x0123, vk4x0123, 4), 4);

      const vfloat32m4_t vi5x0123 = vle32_v_f32m4(i5, 4);
      const vfloat32m4_t vk5x0123 = vle32_v_f32m4(w + 24, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi5x0123, vk5x0123, 4), 4);

      const vfloat32m4_t vi6x0123 = vle32_v_f32m4(i6, 4);
      const vfloat32m4_t vk6x0123 = vle32_v_f32m4(w + 28, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi6x0123, vk6x0123, 4), 4);

      const vfloat32m4_t vi7x0123 = vle32_v_f32m4(i7, 4);
      const vfloat32m4_t vk7x0123 = vle32_v_f32m4(w + 32, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi7x0123, vk7x0123, 4), 4);

      const vfloat32m4_t vi8x0123 = vle32_v_f32m4(i8, 4);
      const vfloat32m4_t vk8x0123 = vle32_v_f32m4(w + 36, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi8x0123, vk8x0123, 4), 4);

      const vfloat32m4_t vi9x0123 = vle32_v_f32m4(i9, 4);
      const vfloat32m4_t vk9x0123 = vle32_v_f32m4(w + 40, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi9x0123, vk9x0123, 4), 4);

      const vfloat32m4_t vi10x0123 = vle32_v_f32m4(i10, 4);
      const vfloat32m4_t vk10x0123 = vle32_v_f32m4(w + 44, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi10x0123, vk10x0123, 4), 4);

      const vfloat32m4_t vi11x0123 = vle32_v_f32m4(i11, 4);
      const vfloat32m4_t vk11x0123 = vle32_v_f32m4(w + 48, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi11x0123, vk11x0123, 4), 4);

      const vfloat32m4_t vi12x0123 = vle32_v_f32m4(i12, 4);
      const vfloat32m4_t vk12x0123 = vle32_v_f32m4(w + 52, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi12x0123, vk12x0123, 4), 4);

      const vfloat32m4_t vi13x0123 = vle32_v_f32m4(i13, 4);
      const vfloat32m4_t vk13x0123 = vle32_v_f32m4(w + 56, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi13x0123, vk13x0123, 4), 4);

      const vfloat32m4_t vi14x0123 = vle32_v_f32m4(i14, 4);
      const vfloat32m4_t vk14x0123 = vle32_v_f32m4(w + 60, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi14x0123, vk14x0123, 4), 4);

      const vfloat32m4_t vi15x0123 = vle32_v_f32m4(i15, 4);
      const vfloat32m4_t vk15x0123 = vle32_v_f32m4(w + 64, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi15x0123, vk15x0123, 4), 4);

      const vfloat32m4_t vi16x0123 = vle32_v_f32m4(i16, 4);
      const vfloat32m4_t vk16x0123 = vle32_v_f32m4(w + 68, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi16x0123, vk16x0123, 4), 4);

      const vfloat32m4_t vi17x0123 = vle32_v_f32m4(i17, 4);
      const vfloat32m4_t vk17x0123 = vle32_v_f32m4(w + 72, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi17x0123, vk17x0123, 4), 4);

      const vfloat32m4_t vi18x0123 = vle32_v_f32m4(i18, 4);
      const vfloat32m4_t vk18x0123 = vle32_v_f32m4(w + 76, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi18x0123, vk18x0123, 4), 4);

      const vfloat32m4_t vi19x0123 = vle32_v_f32m4(i19, 4);
      const vfloat32m4_t vk19x0123 = vle32_v_f32m4(w + 80, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi19x0123, vk19x0123, 4), 4);

      const vfloat32m4_t vi20x0123 = vle32_v_f32m4(i20, 4);
      const vfloat32m4_t vk20x0123 = vle32_v_f32m4(w + 84, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi20x0123, vk20x0123, 4), 4);

      const vfloat32m4_t vi21x0123 = vle32_v_f32m4(i21, 4);
      const vfloat32m4_t vk21x0123 = vle32_v_f32m4(w + 88, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi21x0123, vk21x0123, 4), 4);

      const vfloat32m4_t vi22x0123 = vle32_v_f32m4(i22, 4);
      const vfloat32m4_t vk22x0123 = vle32_v_f32m4(w + 92, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi22x0123, vk22x0123, 4), 4);

      const vfloat32m4_t vi23x0123 = vle32_v_f32m4(i23, 4);
      const vfloat32m4_t vk23x0123 = vle32_v_f32m4(w + 96, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi23x0123, vk23x0123, 4), 4);

      const vfloat32m4_t vi24x0123 = vle32_v_f32m4(i24, 4);
      const vfloat32m4_t vk24x0123 = vle32_v_f32m4(w + 100, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi24x0123, vk24x0123, 4), 4);


      vfloat32m4_t vacc0123 = vfmax_vf_f32m4(vacc0123p0, vmin, 4);
      vacc0123 = vfmin_vf_f32m4(vacc0123, vmax, 4);

      if (c & 2) {
        vse32_v_f32m4(output, vacc0123, 2);
        output += 2;
        vacc0123 = vrgather_vx_f32m4(vacc0123, 2, 1);
      }
      if (c & 1) {
        vse32_v_f32m4(output, vacc0123, 1);
        output += 1;
      }
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_dwconv_ukernel_25p4c__rvv(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  do {
    const float* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
    }
    const float* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
    }
    const float* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
    }
    const float* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
    }
    const float* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
    }
    const float* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
    }
    const float* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
    }
    const float* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const float*) ((uintptr_t) i7 + input_offset);
    }
    const float* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const float*) ((uintptr_t) i8 + input_offset);
    }
    const float* i9 = input[9];
    assert(i9 != NULL);
    if XNN_UNPREDICTABLE(i9 != zero) {
      i9 = (const float*) ((uintptr_t) i9 + input_offset);
    }
    const float* i10 = input[10];
    assert(i10 != NULL);
    if XNN_UNPREDICTABLE(i10 != zero) {
      i10 = (const float*) ((uintptr_t) i10 + input_offset);
    }
    const float* i11 = input[11];
    assert(i11 != NULL);
    if XNN_UNPREDICTABLE(i11 != zero) {
      i11 = (const float*) ((uintptr_t) i11 + input_offset);
    }
    const float* i12 = input[12];
    assert(i12 != NULL);
    if XNN_UNPREDICTABLE(i12 != zero) {
      i12 = (const float*) ((uintptr_t) i12 + input_offset);
    }
    const float* i13 = input[13];
    assert(i13 != NULL);
    if XNN_UNPREDICTABLE(i13 != zero) {
      i13 = (const float*) ((uintptr_t) i13 + input_offset);
    }
    const float* i14 = input[14];
    assert(i14 != NULL);
    if XNN_UNPREDICTABLE(i14 != zero) {
      i14 = (const float*) ((uintptr_t) i14 + input_offset);
    }
    const float* i15 = input[15];
    assert(i15 != NULL);
    if XNN_UNPREDICTABLE(i15 != zero) {
      i15 = (const float*) ((uintptr_t) i15 + input_offset);
    }
    const float* i16 = input[16];
    assert(i16 != NULL);
    if XNN_UNPREDICTABLE(i16 != zero) {
      i16 = (const float*) ((uintptr_t) i16 + input_offset);
    }
    const float* i17 = input[17];
    assert(i17 != NULL);
    if XNN_UNPREDICTABLE(i17 != zero) {
      i17 = (const float*) ((uintptr_t) i17 + input_offset);
    }
    const float* i18 = input[18];
    assert(i18 != NULL);
    if XNN_UNPREDICTABLE(i18 != zero) {
      i18 = (const float*) ((uintptr_t) i18 + input_offset);
    }
    const float* i19 = input[19];
    assert(i19 != NULL);
    if XNN_UNPREDICTABLE(i19 != zero) {
      i19 = (const float*) ((uintptr_t) i19 + input_offset);
    }
    const float* i20 = input[20];
    assert(i20 != NULL);
    if XNN_UNPREDICTABLE(i20 != zero) {
      i20 = (const float*) ((uintptr_t) i20 + input_offset);
    }
    const float* i21 = input[21];
    assert(i21 != NULL);
    if XNN_UNPREDICTABLE(i21 != zero) {
      i21 = (const float*) ((uintptr_t) i21 + input_offset);
    }
    const float* i22 = input[22];
    assert(i22 != NULL);
    if XNN_UNPREDICTABLE(i22 != zero) {
      i22 = (const float*) ((uintptr_t) i22 + input_offset);
    }
    const float* i23 = input[23];
    assert(i23 != NULL);
    if XNN_UNPREDICTABLE(i23 != zero) {
      i23 = (const float*) ((uintptr_t) i23 + input_offset);
    }
    const float* i24 = input[24];
    assert(i24 != NULL);
    if XNN_UNPREDICTABLE(i24 != zero) {
      i24 = (const float*) ((uintptr_t) i24 + input_offset);
    }
    input = (const float**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* w = weights;
    for (; c >= 4; c -= 4) {
      vfloat32m4_t vacc0123p0 = vle32_v_f32m4(w, 4);


      const vfloat32m4_t vi0x0123 = vle32_v_f32m4(i0, 4);
      i0 += 4;

      const vfloat32m4_t vk0x0123 = vle32_v_f32m4(w + 4, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi0x0123, vk0x0123, 4), 4);

      const vfloat32m4_t vi1x0123 = vle32_v_f32m4(i1, 4);
      i1 += 4;

      const vfloat32m4_t vk1x0123 = vle32_v_f32m4(w + 8, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi1x0123, vk1x0123, 4), 4);

      const vfloat32m4_t vi2x0123 = vle32_v_f32m4(i2, 4);
      i2 += 4;

      const vfloat32m4_t vk2x0123 = vle32_v_f32m4(w + 12, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi2x0123, vk2x0123, 4), 4);

      const vfloat32m4_t vi3x0123 = vle32_v_f32m4(i3, 4);
      i3 += 4;

      const vfloat32m4_t vk3x0123 = vle32_v_f32m4(w + 16, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi3x0123, vk3x0123, 4), 4);

      const vfloat32m4_t vi4x0123 = vle32_v_f32m4(i4, 4);
      i4 += 4;

      const vfloat32m4_t vk4x0123 = vle32_v_f32m4(w + 20, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi4x0123, vk4x0123, 4), 4);

      const vfloat32m4_t vi5x0123 = vle32_v_f32m4(i5, 4);
      i5 += 4;

      const vfloat32m4_t vk5x0123 = vle32_v_f32m4(w + 24, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi5x0123, vk5x0123, 4), 4);

      const vfloat32m4_t vi6x0123 = vle32_v_f32m4(i6, 4);
      i6 += 4;

      const vfloat32m4_t vk6x0123 = vle32_v_f32m4(w + 28, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi6x0123, vk6x0123, 4), 4);

      const vfloat32m4_t vi7x0123 = vle32_v_f32m4(i7, 4);
      i7 += 4;

      const vfloat32m4_t vk7x0123 = vle32_v_f32m4(w + 32, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi7x0123, vk7x0123, 4), 4);

      const vfloat32m4_t vi8x0123 = vle32_v_f32m4(i8, 4);
      i8 += 4;

      const vfloat32m4_t vk8x0123 = vle32_v_f32m4(w + 36, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi8x0123, vk8x0123, 4), 4);

      const vfloat32m4_t vi9x0123 = vle32_v_f32m4(i9, 4);
      i9 += 4;

      const vfloat32m4_t vk9x0123 = vle32_v_f32m4(w + 40, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi9x0123, vk9x0123, 4), 4);

      const vfloat32m4_t vi10x0123 = vle32_v_f32m4(i10, 4);
      i10 += 4;

      const vfloat32m4_t vk10x0123 = vle32_v_f32m4(w + 44, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi10x0123, vk10x0123, 4), 4);

      const vfloat32m4_t vi11x0123 = vle32_v_f32m4(i11, 4);
      i11 += 4;

      const vfloat32m4_t vk11x0123 = vle32_v_f32m4(w + 48, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi11x0123, vk11x0123, 4), 4);

      const vfloat32m4_t vi12x0123 = vle32_v_f32m4(i12, 4);
      i12 += 4;

      const vfloat32m4_t vk12x0123 = vle32_v_f32m4(w + 52, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi12x0123, vk12x0123, 4), 4);

      const vfloat32m4_t vi13x0123 = vle32_v_f32m4(i13, 4);
      i13 += 4;

      const vfloat32m4_t vk13x0123 = vle32_v_f32m4(w + 56, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi13x0123, vk13x0123, 4), 4);

      const vfloat32m4_t vi14x0123 = vle32_v_f32m4(i14, 4);
      i14 += 4;

      const vfloat32m4_t vk14x0123 = vle32_v_f32m4(w + 60, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi14x0123, vk14x0123, 4), 4);

      const vfloat32m4_t vi15x0123 = vle32_v_f32m4(i15, 4);
      i15 += 4;

      const vfloat32m4_t vk15x0123 = vle32_v_f32m4(w + 64, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi15x0123, vk15x0123, 4), 4);

      const vfloat32m4_t vi16x0123 = vle32_v_f32m4(i16, 4);
      i16 += 4;

      const vfloat32m4_t vk16x0123 = vle32_v_f32m4(w + 68, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi16x0123, vk16x0123, 4), 4);

      const vfloat32m4_t vi17x0123 = vle32_v_f32m4(i17, 4);
      i17 += 4;

      const vfloat32m4_t vk17x0123 = vle32_v_f32m4(w + 72, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi17x0123, vk17x0123, 4), 4);

      const vfloat32m4_t vi18x0123 = vle32_v_f32m4(i18, 4);
      i18 += 4;

      const vfloat32m4_t vk18x0123 = vle32_v_f32m4(w + 76, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi18x0123, vk18x0123, 4), 4);

      const vfloat32m4_t vi19x0123 = vle32_v_f32m4(i19, 4);
      i19 += 4;

      const vfloat32m4_t vk19x0123 = vle32_v_f32m4(w + 80, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi19x0123, vk19x0123, 4), 4);

      const vfloat32m4_t vi20x0123 = vle32_v_f32m4(i20, 4);
      i20 += 4;

      const vfloat32m4_t vk20x0123 = vle32_v_f32m4(w + 84, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi20x0123, vk20x0123, 4), 4);

      const vfloat32m4_t vi21x0123 = vle32_v_f32m4(i21, 4);
      i21 += 4;

      const vfloat32m4_t vk21x0123 = vle32_v_f32m4(w + 88, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi21x0123, vk21x0123, 4), 4);

      const vfloat32m4_t vi22x0123 = vle32_v_f32m4(i22, 4);
      i22 += 4;

      const vfloat32m4_t vk22x0123 = vle32_v_f32m4(w + 92, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi22x0123, vk22x0123, 4), 4);

      const vfloat32m4_t vi23x0123 = vle32_v_f32m4(i23, 4);
      i23 += 4;

      const vfloat32m4_t vk23x0123 = vle32_v_f32m4(w + 96, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi23x0123, vk23x0123, 4), 4);

      const vfloat32m4_t vi24x0123 = vle32_v_f32m4(i24, 4);
      i24 += 4;

      const vfloat32m4_t vk24x0123 = vle32_v_f32m4(w + 100, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi24x0123, vk24x0123, 4), 4);

      w += 104;

      const vfloat32m4_t vacc0123 = vacc0123p0;

      vse32_v_f32m4(output, vacc0123, 4);
      output += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      vfloat32m4_t vacc0123p0 = vle32_v_f32m4(w, 4);

      const vfloat32m4_t vi0x0123 = vle32_v_f32m4(i0, 4);
      const vfloat32m4_t vk0x0123 = vle32_v_f32m4(w + 4, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi0x0123, vk0x0123, 4), 4);

      const vfloat32m4_t vi1x0123 = vle32_v_f32m4(i1, 4);
      const vfloat32m4_t vk1x0123 = vle32_v_f32m4(w + 8, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi1x0123, vk1x0123, 4), 4);

      const vfloat32m4_t vi2x0123 = vle32_v_f32m4(i2, 4);
      const vfloat32m4_t vk2x0123 = vle32_v_f32m4(w + 12, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi2x0123, vk2x0123, 4), 4);

      const vfloat32m4_t vi3x0123 = vle32_v_f32m4(i3, 4);
      const vfloat32m4_t vk3x0123 = vle32_v_f32m4(w + 16, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi3x0123, vk3x0123, 4), 4);

      const vfloat32m4_t vi4x0123 = vle32_v_f32m4(i4, 4);
      const vfloat32m4_t vk4x0123 = vle32_v_f32m4(w + 20, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi4x0123, vk4x0123, 4), 4);

      const vfloat32m4_t vi5x0123 = vle32_v_f32m4(i5, 4);
      const vfloat32m4_t vk5x0123 = vle32_v_f32m4(w + 24, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi5x0123, vk5x0123, 4), 4);

      const vfloat32m4_t vi6x0123 = vle32_v_f32m4(i6, 4);
      const vfloat32m4_t vk6x0123 = vle32_v_f32m4(w + 28, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi6x0123, vk6x0123, 4), 4);

      const vfloat32m4_t vi7x0123 = vle32_v_f32m4(i7, 4);
      const vfloat32m4_t vk7x0123 = vle32_v_f32m4(w + 32, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi7x0123, vk7x0123, 4), 4);

      const vfloat32m4_t vi8x0123 = vle32_v_f32m4(i8, 4);
      const vfloat32m4_t vk8x0123 = vle32_v_f32m4(w + 36, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi8x0123, vk8x0123, 4), 4);

      const vfloat32m4_t vi9x0123 = vle32_v_f32m4(i9, 4);
      const vfloat32m4_t vk9x0123 = vle32_v_f32m4(w + 40, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi9x0123, vk9x0123, 4), 4);

      const vfloat32m4_t vi10x0123 = vle32_v_f32m4(i10, 4);
      const vfloat32m4_t vk10x0123 = vle32_v_f32m4(w + 44, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi10x0123, vk10x0123, 4), 4);

      const vfloat32m4_t vi11x0123 = vle32_v_f32m4(i11, 4);
      const vfloat32m4_t vk11x0123 = vle32_v_f32m4(w + 48, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi11x0123, vk11x0123, 4), 4);

      const vfloat32m4_t vi12x0123 = vle32_v_f32m4(i12, 4);
      const vfloat32m4_t vk12x0123 = vle32_v_f32m4(w + 52, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi12x0123, vk12x0123, 4), 4);

      const vfloat32m4_t vi13x0123 = vle32_v_f32m4(i13, 4);
      const vfloat32m4_t vk13x0123 = vle32_v_f32m4(w + 56, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi13x0123, vk13x0123, 4), 4);

      const vfloat32m4_t vi14x0123 = vle32_v_f32m4(i14, 4);
      const vfloat32m4_t vk14x0123 = vle32_v_f32m4(w + 60, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi14x0123, vk14x0123, 4), 4);

      const vfloat32m4_t vi15x0123 = vle32_v_f32m4(i15, 4);
      const vfloat32m4_t vk15x0123 = vle32_v_f32m4(w + 64, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi15x0123, vk15x0123, 4), 4);

      const vfloat32m4_t vi16x0123 = vle32_v_f32m4(i16, 4);
      const vfloat32m4_t vk16x0123 = vle32_v_f32m4(w + 68, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi16x0123, vk16x0123, 4), 4);

      const vfloat32m4_t vi17x0123 = vle32_v_f32m4(i17, 4);
      const vfloat32m4_t vk17x0123 = vle32_v_f32m4(w + 72, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi17x0123, vk17x0123, 4), 4);

      const vfloat32m4_t vi18x0123 = vle32_v_f32m4(i18, 4);
      const vfloat32m4_t vk18x0123 = vle32_v_f32m4(w + 76, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi18x0123, vk18x0123, 4), 4);

      const vfloat32m4_t vi19x0123 = vle32_v_f32m4(i19, 4);
      const vfloat32m4_t vk19x0123 = vle32_v_f32m4(w + 80, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi19x0123, vk19x0123, 4), 4);

      const vfloat32m4_t vi20x0123 = vle32_v_f32m4(i20, 4);
      const vfloat32m4_t vk20x0123 = vle32_v_f32m4(w + 84, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi20x0123, vk20x0123, 4), 4);

      const vfloat32m4_t vi21x0123 = vle32_v_f32m4(i21, 4);
      const vfloat32m4_t vk21x0123 = vle32_v_f32m4(w + 88, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi21x0123, vk21x0123, 4), 4);

      const vfloat32m4_t vi22x0123 = vle32_v_f32m4(i22, 4);
      const vfloat32m4_t vk22x0123 = vle32_v_f32m4(w + 92, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi22x0123, vk22x0123, 4), 4);

      const vfloat32m4_t vi23x0123 = vle32_v_f32m4(i23, 4);
      const vfloat32m4_t vk23x0123 = vle32_v_f32m4(w + 96, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi23x0123, vk23x0123, 4), 4);

      const vfloat32m4_t vi24x0123 = vle32_v_f32m4(i24, 4);
      const vfloat32m4_t vk24x0123 = vle32_v_f32m4(w + 100, 4);
      vacc0123p0 = vfadd_vv_f32m4(vacc0123p0, vfmul_vv_f32m4(vi24x0123, vk24x0123, 4), 4);


      vfloat32m4_t vacc0123 = vacc0123p0;

      if (c & 2) {
        vse32_v_f32m4(output, vacc0123, 2);
        output += 2;
        vacc0123 = vrgather_vx_f32m4(vacc0123, 2, 1);
      }
      if (c & 1) {
        vse32_v_f32m4(output, vacc0123, 1);
        output += 1;
      }
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
