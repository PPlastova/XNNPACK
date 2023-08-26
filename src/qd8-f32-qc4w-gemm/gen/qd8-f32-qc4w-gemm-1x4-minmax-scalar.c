// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/gemm.h>
#include <xnnpack/math.h>

void xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__scalar(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_qc4w_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);

  const int8_t* a0 = a;
  float* c0 = c;

  const int32_t vminus_kernel_zero_point = params->scalar.minus_kernel_zero_point;
  assert(vminus_kernel_zero_point >= -15);
  assert(vminus_kernel_zero_point <= 0);

  do {
    const int32_t vksum0 = ((const int32_t*) w)[0];
    const int32_t vksum1 = ((const int32_t*) w)[1];
    const int32_t vksum2 = ((const int32_t*) w)[2];
    const int32_t vksum3 = ((const int32_t*) w)[3];
    const int32_t vinput_zero_point0 = quantization_params[0].zero_point;
    int32_t vacc0x0 = vksum0 * vinput_zero_point0;
    int32_t vacc0x1 = vksum1 * vinput_zero_point0;
    int32_t vacc0x2 = vksum2 * vinput_zero_point0;
    int32_t vacc0x3 = vksum3 * vinput_zero_point0;
    w = (const int32_t*) w + 4;

    size_t k = kc;
    for (; k >= 2 * sizeof(int8_t); k -= 2 * sizeof(int8_t)) {
      const int32_t va00 = (int32_t) a0[0];
      const int32_t va01 = (int32_t) a0[1];
      a0 += 2;

      const uint32_t vbi00 = (uint32_t) ((const uint8_t*) w)[0];
      const uint32_t vbi10 = (uint32_t) ((const uint8_t*) w)[1];
      const uint32_t vbi20 = (uint32_t) ((const uint8_t*) w)[2];
      const uint32_t vbi30 = (uint32_t) ((const uint8_t*) w)[3];
      const int32_t vb00 = (int32_t) (vbi00 & 0xF) + vminus_kernel_zero_point;
      const int32_t vb10 = (int32_t) (vbi10 & 0xF) + vminus_kernel_zero_point;
      const int32_t vb20 = (int32_t) (vbi20 & 0xF) + vminus_kernel_zero_point;
      const int32_t vb30 = (int32_t) (vbi30 & 0xF) + vminus_kernel_zero_point;
      const int32_t vb01 = (int32_t) (vbi00 >> 4) + vminus_kernel_zero_point;
      const int32_t vb11 = (int32_t) (vbi10 >> 4) + vminus_kernel_zero_point;
      const int32_t vb21 = (int32_t) (vbi20 >> 4) + vminus_kernel_zero_point;
      const int32_t vb31 = (int32_t) (vbi30 >> 4) + vminus_kernel_zero_point;
      w = (const int8_t*) w + 4;

      vacc0x0 += va00 * vb00;
      vacc0x1 += va00 * vb10;
      vacc0x2 += va00 * vb20;
      vacc0x3 += va00 * vb30;
      vacc0x0 += va01 * vb01;
      vacc0x1 += va01 * vb11;
      vacc0x2 += va01 * vb21;
      vacc0x3 += va01 * vb31;
    }
    if XNN_UNLIKELY(k != 0) {
      const int32_t va0 = (int32_t) *a0++;

      const int32_t vb0 = (int32_t) ((const int8_t*) w)[0] + vminus_kernel_zero_point;
      const int32_t vb1 = (int32_t) ((const int8_t*) w)[1] + vminus_kernel_zero_point;
      const int32_t vb2 = (int32_t) ((const int8_t*) w)[2] + vminus_kernel_zero_point;
      const int32_t vb3 = (int32_t) ((const int8_t*) w)[3] + vminus_kernel_zero_point;
      w = (const int8_t*) w + 4;

      vacc0x0 += va0 * vb0;
      vacc0x1 += va0 * vb1;
      vacc0x2 += va0 * vb2;
      vacc0x3 += va0 * vb3;
    }

    float vout0x0 = (float) vacc0x0;
    float vout0x1 = (float) vacc0x1;
    float vout0x2 = (float) vacc0x2;
    float vout0x3 = (float) vacc0x3;

    const float vinput_scale0 = quantization_params[0].inv_scale;
    vout0x0 *= vinput_scale0;
    vout0x1 *= vinput_scale0;
    vout0x2 *= vinput_scale0;
    vout0x3 *= vinput_scale0;

    const float vfilter_output_scale0 = ((const float*) w)[0];
    vout0x0 *= vfilter_output_scale0;
    const float vfilter_output_scale1 = ((const float*) w)[1];
    vout0x1 *= vfilter_output_scale1;
    const float vfilter_output_scale2 = ((const float*) w)[2];
    vout0x2 *= vfilter_output_scale2;
    const float vfilter_output_scale3 = ((const float*) w)[3];
    vout0x3 *= vfilter_output_scale3;

    const float vbias0 = ((const float*) w)[4];
    vout0x0 += vbias0;
    const float vbias1 = ((const float*) w)[5];
    vout0x1 += vbias1;
    const float vbias2 = ((const float*) w)[6];
    vout0x2 += vbias2;
    const float vbias3 = ((const float*) w)[7];
    vout0x3 += vbias3;

    w = (const float*) w + 8;

    const float voutput_min = params->scalar.min;
    vout0x0 = math_max_f32(vout0x0, voutput_min);
    vout0x1 = math_max_f32(vout0x1, voutput_min);
    vout0x2 = math_max_f32(vout0x2, voutput_min);
    vout0x3 = math_max_f32(vout0x3, voutput_min);

    const float voutput_max = params->scalar.max;
    vout0x0 = math_min_f32(vout0x0, voutput_max);
    vout0x1 = math_min_f32(vout0x1, voutput_max);
    vout0x2 = math_min_f32(vout0x2, voutput_max);
    vout0x3 = math_min_f32(vout0x3, voutput_max);

    if XNN_LIKELY(nc >= 4) {
      c0[0] = vout0x0;
      c0[1] = vout0x1;
      c0[2] = vout0x2;
      c0[3] = vout0x3;

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      nc -= 4;
    } else {
      if (nc & 2) {
        c0[0] = vout0x0;
        c0[1] = vout0x1;
        vout0x0 = vout0x2;
        c0 += 2;
      }
      if (nc & 1) {
        c0[0] = vout0x0;
      }

      nc = 0;
    }
  } while (nc != 0);
}
