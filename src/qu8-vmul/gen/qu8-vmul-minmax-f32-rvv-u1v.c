// Auto-generated file. Do not edit!
//   Template: src/qs8-vmul/rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <riscv_vector.h>
#include <xnnpack/vbinary.h>


void xnn_qu8_vmul_minmax_fp32_ukernel__rvv_u1v(
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
    const size_t n = vsetvl_e8m1(batch);

    vuint8m1_t in_a_u8v = vle8_v_u8m1(input_a, n); input_a += n;
    vuint8m1_t in_b_u8v = vle8_v_u8m1(input_b, n); input_b += n;
    vuint16m2_t a_u16v = vwsubu_vx_u16m2(in_a_u8v, a_zero_point, n);
    vuint16m2_t b_u16v = vwsubu_vx_u16m2(in_b_u8v, b_zero_point, n);
    vint16m2_t a_i16v = vreinterpret_v_u16m2_i16m2(a_u16v);
    vint16m2_t b_i16v = vreinterpret_v_u16m2_i16m2(b_u16v);

    vint32m4_t acc_i32v = vwmul_vv_i32m4(a_i16v, b_i16v, n);
    vfloat32m4_t acc_f32v = vfcvt_f_x_v_f32m4(acc_i32v, n);
    acc_f32v = vfmul_vf_f32m4(acc_f32v, scale, n);
    acc_f32v = vfmin_vf_f32m4(vfmax_vf_f32m4(acc_f32v, output_min_less_zero_point, n), output_max_less_zero_point, n);
    acc_f32v = vfadd_vf_f32m4(acc_f32v, magic_bias, n);

    vuint32m4_t out_u32v = vfcvt_xu_f_v_u32m4(acc_f32v, n);
    out_u32v = vsub_vx_u32m4(out_u32v, magic_bias_less_output_zero_point, n);
    vuint16m2_t out_u16v = vncvt_x_x_w_u16m2(out_u32v, n);
    vuint8m1_t out_u8v = vncvt_x_x_w_u8m1(out_u16v, n);
    vse8_v_u8m1(output, out_u8v, n); output += n;

    batch -= n;
  } while (batch != 0);
}
