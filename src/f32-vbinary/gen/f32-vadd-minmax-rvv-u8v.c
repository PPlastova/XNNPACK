// Auto-generated file. Do not edit!
//   Template: src/f32-vbinary/vop-rvv.c.in
//   Generator: tools/xngen
//
// Copyright 2024 SiFive, Inc.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <riscv_vector.h>

#include <xnnpack/common.h>
#include <xnnpack/vbinary.h>


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
