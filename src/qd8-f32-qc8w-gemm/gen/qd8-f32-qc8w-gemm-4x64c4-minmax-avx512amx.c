// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/c4-avx512amx.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/gemm.h>
#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/math.h>
#include <xnnpack/unaligned.h>


void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x64c4__avx512amx(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

// TODO: amxintrin.h only provide intrinsics for __x86_64__
// Update if amxintrin changes
#if defined(__x86_64__)
  __attribute__((aligned(64))) int32_t res0[4 * 16];
  __attribute__((aligned(64))) int32_t res1[4 * 16];
  __attribute__((aligned(64))) int32_t res2[4 * 16];
  __attribute__((aligned(64))) int32_t res3[4 * 16];

  kc = round_up_po2(kc, 4 * sizeof(int8_t));
  const size_t kremainder = (kc & 63) ? (kc & 63) : 64;

  // Define tile config data structure
  struct __tile_config {
    uint8_t palette_id;
    uint8_t start_row;
    uint8_t reserved_0[14];
    uint16_t colsb[8];
    uint16_t reserved_1[8];
    uint8_t rows[8];
    uint8_t reserved_2[8];
  };

  // Load tile configuration
  __attribute__((aligned(64))) struct __tile_config tile_data = {0};
  tile_data.palette_id = 1;
  tile_data.rows[0] = mr;              // tmm0 = res 0
  tile_data.rows[1] = mr;              // tmm1 = res 1
  tile_data.rows[2] = mr;              // tmm2 = res 2
  tile_data.rows[3] = mr;              // tmm3 = res 3
  tile_data.rows[4] = mr;              // tmm4 = input
  tile_data.rows[5] = 16;              // tmm5 = weights
  tile_data.rows[6] = mr;              // tmm6 = input remainder
  tile_data.rows[7] = kremainder >> 2; // tmm7 = weights remainder

  tile_data.colsb[0] = 64;          // tmm0 = res 0
  tile_data.colsb[1] = 64;          // tmm1 = res 1
  tile_data.colsb[2] = 64;          // tmm2 = res 1
  tile_data.colsb[3] = 64;          // tmm3 = res 1
  tile_data.colsb[4] = 64;          // tmm4 = input
  tile_data.colsb[5] = 64;          // tmm5 = weights
  tile_data.colsb[6] = kremainder;  // tmm6 = input remainder
  tile_data.colsb[7] = 64;          // tmm7 = weights remainder

  //_tile_loadconfig(&tile_data);
  __asm__ volatile ("ldtilecfg %0" :: "m" (tile_data));

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
  if XNN_UNPREDICTABLE(mr != 4) {
    c3 = c2;
  }

  const __m512 voutput_min = _mm512_set1_ps(params->scalar.min);
  const __m512 voutput_max = _mm512_set1_ps(params->scalar.max);

  do {
    const __m512i vksum0123456789ABCDEF = _mm512_load_epi32((const int32_t*) w + 0);
    const __m512i vksumGHIJKLMNOPQRSTUV = _mm512_load_epi32((const int32_t*) w + 16);
    const __m512i vksumWXYZabcdefghijkl = _mm512_load_epi32((const int32_t*) w + 32);
    const __m512i vksummnopqrstuvwxyz01 = _mm512_load_epi32((const int32_t*) w + 48);
    w = (const int32_t*) w + 64;

    // Zero tile accumulator
    __asm__ volatile (
      "tilezero %%tmm0\n"
      "tilezero %%tmm1\n"
      "tilezero %%tmm2\n"
      "tilezero %%tmm3\n"
      ::);

    size_t k = kc;
    while (k >= 64 * sizeof(int8_t)) {
      _tile_loadd(4, a, a_stride);
      a += 64;
      _tile_loadd(5, (const int8_t*) w + 0, 256);
      _tile_dpbssd(0, 4, 5);
      _tile_loadd(5, (const int8_t*) w + 64, 256);
      _tile_dpbssd(1, 4, 5);
      _tile_loadd(5, (const int8_t*) w + 128, 256);
      _tile_dpbssd(2, 4, 5);
      _tile_loadd(5, (const int8_t*) w + 192, 256);
      _tile_dpbssd(3, 4, 5);

      w = (const int8_t*) w + 4096;
      k -= 64 * sizeof(int8_t);
    }

    if XNN_UNLIKELY(k != 0) {
      _tile_loadd(6, a, a_stride);
      a += kremainder;
      _tile_loadd(7, (const int8_t*) w + 0, 256);
      _tile_dpbssd(0, 6, 7);
      _tile_loadd(7, (const int8_t*) w + 64, 256);
      _tile_dpbssd(1, 6, 7);
      _tile_loadd(7, (const int8_t*) w + 128, 256);
      _tile_dpbssd(2, 6, 7);
      _tile_loadd(7, (const int8_t*) w + 192, 256);
      _tile_dpbssd(3, 6, 7);

      w = (const int8_t*) w + kremainder * 64;
      k -= kremainder * sizeof(int8_t);
    }

    // Add tile to bias
    _tile_stored(0, res0, 64);
    _tile_stored(1, res1, 64);
    _tile_stored(2, res2, 64);
    _tile_stored(3, res3, 64);

    __m512i vacc0x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, _mm512_set1_epi32((int) quantization_params[0].zero_point));
    __m512i vacc0xGHIJKLMNOPQRSTUV = _mm512_mullo_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_set1_epi32((int) quantization_params[0].zero_point));
    __m512i vacc0xWXYZabcdefghijkl = _mm512_mullo_epi32(vksumWXYZabcdefghijkl, _mm512_set1_epi32((int) quantization_params[0].zero_point));
    __m512i vacc0xmnopqrstuvwxyz01 = _mm512_mullo_epi32(vksummnopqrstuvwxyz01, _mm512_set1_epi32((int) quantization_params[0].zero_point));
    __m512i vacc1x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, _mm512_set1_epi32((int) quantization_params[1].zero_point));
    __m512i vacc1xGHIJKLMNOPQRSTUV = _mm512_mullo_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_set1_epi32((int) quantization_params[1].zero_point));
    __m512i vacc1xWXYZabcdefghijkl = _mm512_mullo_epi32(vksumWXYZabcdefghijkl, _mm512_set1_epi32((int) quantization_params[1].zero_point));
    __m512i vacc1xmnopqrstuvwxyz01 = _mm512_mullo_epi32(vksummnopqrstuvwxyz01, _mm512_set1_epi32((int) quantization_params[1].zero_point));
    __m512i vacc2x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, _mm512_set1_epi32((int) quantization_params[2].zero_point));
    __m512i vacc2xGHIJKLMNOPQRSTUV = _mm512_mullo_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_set1_epi32((int) quantization_params[2].zero_point));
    __m512i vacc2xWXYZabcdefghijkl = _mm512_mullo_epi32(vksumWXYZabcdefghijkl, _mm512_set1_epi32((int) quantization_params[2].zero_point));
    __m512i vacc2xmnopqrstuvwxyz01 = _mm512_mullo_epi32(vksummnopqrstuvwxyz01, _mm512_set1_epi32((int) quantization_params[2].zero_point));
    __m512i vacc3x0123456789ABCDEF = _mm512_mullo_epi32(vksum0123456789ABCDEF, _mm512_set1_epi32((int) quantization_params[3].zero_point));
    __m512i vacc3xGHIJKLMNOPQRSTUV = _mm512_mullo_epi32(vksumGHIJKLMNOPQRSTUV, _mm512_set1_epi32((int) quantization_params[3].zero_point));
    __m512i vacc3xWXYZabcdefghijkl = _mm512_mullo_epi32(vksumWXYZabcdefghijkl, _mm512_set1_epi32((int) quantization_params[3].zero_point));
    __m512i vacc3xmnopqrstuvwxyz01 = _mm512_mullo_epi32(vksummnopqrstuvwxyz01, _mm512_set1_epi32((int) quantization_params[3].zero_point));
    vacc0x0123456789ABCDEF = _mm512_add_epi32(vacc0x0123456789ABCDEF, _mm512_load_epi32(res0 + 0));
    vacc0xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc0xGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 0));
    vacc0xWXYZabcdefghijkl = _mm512_add_epi32(vacc0xWXYZabcdefghijkl, _mm512_load_epi32(res2 + 0));
    vacc0xmnopqrstuvwxyz01 = _mm512_add_epi32(vacc0xmnopqrstuvwxyz01, _mm512_load_epi32(res3 + 0));
    vacc1x0123456789ABCDEF = _mm512_add_epi32(vacc1x0123456789ABCDEF, _mm512_load_epi32(res0 + 16));
    vacc1xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc1xGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 16));
    vacc1xWXYZabcdefghijkl = _mm512_add_epi32(vacc1xWXYZabcdefghijkl, _mm512_load_epi32(res2 + 16));
    vacc1xmnopqrstuvwxyz01 = _mm512_add_epi32(vacc1xmnopqrstuvwxyz01, _mm512_load_epi32(res3 + 16));
    vacc2x0123456789ABCDEF = _mm512_add_epi32(vacc2x0123456789ABCDEF, _mm512_load_epi32(res0 + 32));
    vacc2xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc2xGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 32));
    vacc2xWXYZabcdefghijkl = _mm512_add_epi32(vacc2xWXYZabcdefghijkl, _mm512_load_epi32(res2 + 32));
    vacc2xmnopqrstuvwxyz01 = _mm512_add_epi32(vacc2xmnopqrstuvwxyz01, _mm512_load_epi32(res3 + 32));
    vacc3x0123456789ABCDEF = _mm512_add_epi32(vacc3x0123456789ABCDEF, _mm512_load_epi32(res0 + 48));
    vacc3xGHIJKLMNOPQRSTUV = _mm512_add_epi32(vacc3xGHIJKLMNOPQRSTUV, _mm512_load_epi32(res1 + 48));
    vacc3xWXYZabcdefghijkl = _mm512_add_epi32(vacc3xWXYZabcdefghijkl, _mm512_load_epi32(res2 + 48));
    vacc3xmnopqrstuvwxyz01 = _mm512_add_epi32(vacc3xmnopqrstuvwxyz01, _mm512_load_epi32(res3 + 48));

    __m512 vscaled0x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc0x0123456789ABCDEF);
    __m512 vscaled0xGHIJKLMNOPQRSTUV = _mm512_cvtepi32_ps(vacc0xGHIJKLMNOPQRSTUV);
    __m512 vscaled0xWXYZabcdefghijkl = _mm512_cvtepi32_ps(vacc0xWXYZabcdefghijkl);
    __m512 vscaled0xmnopqrstuvwxyz01 = _mm512_cvtepi32_ps(vacc0xmnopqrstuvwxyz01);
    __m512 vscaled1x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc1x0123456789ABCDEF);
    __m512 vscaled1xGHIJKLMNOPQRSTUV = _mm512_cvtepi32_ps(vacc1xGHIJKLMNOPQRSTUV);
    __m512 vscaled1xWXYZabcdefghijkl = _mm512_cvtepi32_ps(vacc1xWXYZabcdefghijkl);
    __m512 vscaled1xmnopqrstuvwxyz01 = _mm512_cvtepi32_ps(vacc1xmnopqrstuvwxyz01);
    __m512 vscaled2x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc2x0123456789ABCDEF);
    __m512 vscaled2xGHIJKLMNOPQRSTUV = _mm512_cvtepi32_ps(vacc2xGHIJKLMNOPQRSTUV);
    __m512 vscaled2xWXYZabcdefghijkl = _mm512_cvtepi32_ps(vacc2xWXYZabcdefghijkl);
    __m512 vscaled2xmnopqrstuvwxyz01 = _mm512_cvtepi32_ps(vacc2xmnopqrstuvwxyz01);
    __m512 vscaled3x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc3x0123456789ABCDEF);
    __m512 vscaled3xGHIJKLMNOPQRSTUV = _mm512_cvtepi32_ps(vacc3xGHIJKLMNOPQRSTUV);
    __m512 vscaled3xWXYZabcdefghijkl = _mm512_cvtepi32_ps(vacc3xWXYZabcdefghijkl);
    __m512 vscaled3xmnopqrstuvwxyz01 = _mm512_cvtepi32_ps(vacc3xmnopqrstuvwxyz01);

    vscaled0x0123456789ABCDEF = _mm512_mul_ps(vscaled0x0123456789ABCDEF, _mm512_set1_ps(quantization_params[0].inv_scale));
    vscaled0xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled0xGHIJKLMNOPQRSTUV, _mm512_set1_ps(quantization_params[0].inv_scale));
    vscaled0xWXYZabcdefghijkl = _mm512_mul_ps(vscaled0xWXYZabcdefghijkl, _mm512_set1_ps(quantization_params[0].inv_scale));
    vscaled0xmnopqrstuvwxyz01 = _mm512_mul_ps(vscaled0xmnopqrstuvwxyz01, _mm512_set1_ps(quantization_params[0].inv_scale));
    vscaled1x0123456789ABCDEF = _mm512_mul_ps(vscaled1x0123456789ABCDEF, _mm512_set1_ps(quantization_params[1].inv_scale));
    vscaled1xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled1xGHIJKLMNOPQRSTUV, _mm512_set1_ps(quantization_params[1].inv_scale));
    vscaled1xWXYZabcdefghijkl = _mm512_mul_ps(vscaled1xWXYZabcdefghijkl, _mm512_set1_ps(quantization_params[1].inv_scale));
    vscaled1xmnopqrstuvwxyz01 = _mm512_mul_ps(vscaled1xmnopqrstuvwxyz01, _mm512_set1_ps(quantization_params[1].inv_scale));
    vscaled2x0123456789ABCDEF = _mm512_mul_ps(vscaled2x0123456789ABCDEF, _mm512_set1_ps(quantization_params[2].inv_scale));
    vscaled2xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled2xGHIJKLMNOPQRSTUV, _mm512_set1_ps(quantization_params[2].inv_scale));
    vscaled2xWXYZabcdefghijkl = _mm512_mul_ps(vscaled2xWXYZabcdefghijkl, _mm512_set1_ps(quantization_params[2].inv_scale));
    vscaled2xmnopqrstuvwxyz01 = _mm512_mul_ps(vscaled2xmnopqrstuvwxyz01, _mm512_set1_ps(quantization_params[2].inv_scale));
    vscaled3x0123456789ABCDEF = _mm512_mul_ps(vscaled3x0123456789ABCDEF, _mm512_set1_ps(quantization_params[3].inv_scale));
    vscaled3xGHIJKLMNOPQRSTUV = _mm512_mul_ps(vscaled3xGHIJKLMNOPQRSTUV, _mm512_set1_ps(quantization_params[3].inv_scale));
    vscaled3xWXYZabcdefghijkl = _mm512_mul_ps(vscaled3xWXYZabcdefghijkl, _mm512_set1_ps(quantization_params[3].inv_scale));
    vscaled3xmnopqrstuvwxyz01 = _mm512_mul_ps(vscaled3xmnopqrstuvwxyz01, _mm512_set1_ps(quantization_params[3].inv_scale));

    const __m512 vfilter_output_scale0123456789ABCDEF = _mm512_load_ps((const float*) w + 0);
    const __m512 vfilter_output_scaleGHIJKLMNOPQRSTUV = _mm512_load_ps((const float*) w + 16);
    const __m512 vfilter_output_scaleWXYZabcdefghijkl = _mm512_load_ps((const float*) w + 32);
    const __m512 vfilter_output_scalemnopqrstuvwxyz01 = _mm512_load_ps((const float*) w + 48);
    w = (const int32_t*) w + 64;
    const __m512 vbias0123456789ABCDEF = _mm512_load_ps((const float*) w + 0);
    const __m512 vbiasGHIJKLMNOPQRSTUV = _mm512_load_ps((const float*) w + 16);
    const __m512 vbiasWXYZabcdefghijkl = _mm512_load_ps((const float*) w + 32);
    const __m512 vbiasmnopqrstuvwxyz01 = _mm512_load_ps((const float*) w + 48);
    w = (const int32_t*) w + 64;

    vscaled0x0123456789ABCDEF = _mm512_fmadd_ps(vscaled0x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled0xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(vscaled0xGHIJKLMNOPQRSTUV, vfilter_output_scaleGHIJKLMNOPQRSTUV, vbiasGHIJKLMNOPQRSTUV);
    vscaled0xWXYZabcdefghijkl = _mm512_fmadd_ps(vscaled0xWXYZabcdefghijkl, vfilter_output_scaleWXYZabcdefghijkl, vbiasWXYZabcdefghijkl);
    vscaled0xmnopqrstuvwxyz01 = _mm512_fmadd_ps(vscaled0xmnopqrstuvwxyz01, vfilter_output_scalemnopqrstuvwxyz01, vbiasmnopqrstuvwxyz01);
    vscaled1x0123456789ABCDEF = _mm512_fmadd_ps(vscaled1x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled1xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(vscaled1xGHIJKLMNOPQRSTUV, vfilter_output_scaleGHIJKLMNOPQRSTUV, vbiasGHIJKLMNOPQRSTUV);
    vscaled1xWXYZabcdefghijkl = _mm512_fmadd_ps(vscaled1xWXYZabcdefghijkl, vfilter_output_scaleWXYZabcdefghijkl, vbiasWXYZabcdefghijkl);
    vscaled1xmnopqrstuvwxyz01 = _mm512_fmadd_ps(vscaled1xmnopqrstuvwxyz01, vfilter_output_scalemnopqrstuvwxyz01, vbiasmnopqrstuvwxyz01);
    vscaled2x0123456789ABCDEF = _mm512_fmadd_ps(vscaled2x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled2xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(vscaled2xGHIJKLMNOPQRSTUV, vfilter_output_scaleGHIJKLMNOPQRSTUV, vbiasGHIJKLMNOPQRSTUV);
    vscaled2xWXYZabcdefghijkl = _mm512_fmadd_ps(vscaled2xWXYZabcdefghijkl, vfilter_output_scaleWXYZabcdefghijkl, vbiasWXYZabcdefghijkl);
    vscaled2xmnopqrstuvwxyz01 = _mm512_fmadd_ps(vscaled2xmnopqrstuvwxyz01, vfilter_output_scalemnopqrstuvwxyz01, vbiasmnopqrstuvwxyz01);
    vscaled3x0123456789ABCDEF = _mm512_fmadd_ps(vscaled3x0123456789ABCDEF, vfilter_output_scale0123456789ABCDEF, vbias0123456789ABCDEF);
    vscaled3xGHIJKLMNOPQRSTUV = _mm512_fmadd_ps(vscaled3xGHIJKLMNOPQRSTUV, vfilter_output_scaleGHIJKLMNOPQRSTUV, vbiasGHIJKLMNOPQRSTUV);
    vscaled3xWXYZabcdefghijkl = _mm512_fmadd_ps(vscaled3xWXYZabcdefghijkl, vfilter_output_scaleWXYZabcdefghijkl, vbiasWXYZabcdefghijkl);
    vscaled3xmnopqrstuvwxyz01 = _mm512_fmadd_ps(vscaled3xmnopqrstuvwxyz01, vfilter_output_scalemnopqrstuvwxyz01, vbiasmnopqrstuvwxyz01);

    vscaled0x0123456789ABCDEF = _mm512_max_ps(vscaled0x0123456789ABCDEF, voutput_min);
    vscaled0xGHIJKLMNOPQRSTUV = _mm512_max_ps(vscaled0xGHIJKLMNOPQRSTUV, voutput_min);
    vscaled0xWXYZabcdefghijkl = _mm512_max_ps(vscaled0xWXYZabcdefghijkl, voutput_min);
    vscaled0xmnopqrstuvwxyz01 = _mm512_max_ps(vscaled0xmnopqrstuvwxyz01, voutput_min);
    vscaled1x0123456789ABCDEF = _mm512_max_ps(vscaled1x0123456789ABCDEF, voutput_min);
    vscaled1xGHIJKLMNOPQRSTUV = _mm512_max_ps(vscaled1xGHIJKLMNOPQRSTUV, voutput_min);
    vscaled1xWXYZabcdefghijkl = _mm512_max_ps(vscaled1xWXYZabcdefghijkl, voutput_min);
    vscaled1xmnopqrstuvwxyz01 = _mm512_max_ps(vscaled1xmnopqrstuvwxyz01, voutput_min);
    vscaled2x0123456789ABCDEF = _mm512_max_ps(vscaled2x0123456789ABCDEF, voutput_min);
    vscaled2xGHIJKLMNOPQRSTUV = _mm512_max_ps(vscaled2xGHIJKLMNOPQRSTUV, voutput_min);
    vscaled2xWXYZabcdefghijkl = _mm512_max_ps(vscaled2xWXYZabcdefghijkl, voutput_min);
    vscaled2xmnopqrstuvwxyz01 = _mm512_max_ps(vscaled2xmnopqrstuvwxyz01, voutput_min);
    vscaled3x0123456789ABCDEF = _mm512_max_ps(vscaled3x0123456789ABCDEF, voutput_min);
    vscaled3xGHIJKLMNOPQRSTUV = _mm512_max_ps(vscaled3xGHIJKLMNOPQRSTUV, voutput_min);
    vscaled3xWXYZabcdefghijkl = _mm512_max_ps(vscaled3xWXYZabcdefghijkl, voutput_min);
    vscaled3xmnopqrstuvwxyz01 = _mm512_max_ps(vscaled3xmnopqrstuvwxyz01, voutput_min);

    vscaled0x0123456789ABCDEF = _mm512_min_ps(vscaled0x0123456789ABCDEF, voutput_max);
    vscaled0xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled0xGHIJKLMNOPQRSTUV, voutput_max);
    vscaled0xWXYZabcdefghijkl = _mm512_min_ps(vscaled0xWXYZabcdefghijkl, voutput_max);
    vscaled0xmnopqrstuvwxyz01 = _mm512_min_ps(vscaled0xmnopqrstuvwxyz01, voutput_max);
    vscaled1x0123456789ABCDEF = _mm512_min_ps(vscaled1x0123456789ABCDEF, voutput_max);
    vscaled1xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled1xGHIJKLMNOPQRSTUV, voutput_max);
    vscaled1xWXYZabcdefghijkl = _mm512_min_ps(vscaled1xWXYZabcdefghijkl, voutput_max);
    vscaled1xmnopqrstuvwxyz01 = _mm512_min_ps(vscaled1xmnopqrstuvwxyz01, voutput_max);
    vscaled2x0123456789ABCDEF = _mm512_min_ps(vscaled2x0123456789ABCDEF, voutput_max);
    vscaled2xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled2xGHIJKLMNOPQRSTUV, voutput_max);
    vscaled2xWXYZabcdefghijkl = _mm512_min_ps(vscaled2xWXYZabcdefghijkl, voutput_max);
    vscaled2xmnopqrstuvwxyz01 = _mm512_min_ps(vscaled2xmnopqrstuvwxyz01, voutput_max);
    vscaled3x0123456789ABCDEF = _mm512_min_ps(vscaled3x0123456789ABCDEF, voutput_max);
    vscaled3xGHIJKLMNOPQRSTUV = _mm512_min_ps(vscaled3xGHIJKLMNOPQRSTUV, voutput_max);
    vscaled3xWXYZabcdefghijkl = _mm512_min_ps(vscaled3xWXYZabcdefghijkl, voutput_max);
    vscaled3xmnopqrstuvwxyz01 = _mm512_min_ps(vscaled3xmnopqrstuvwxyz01, voutput_max);

    if XNN_LIKELY(nc >= 64) {
      _mm512_storeu_ps(c3 + 0, vscaled3x0123456789ABCDEF);
      _mm512_storeu_ps(c3 + 16, vscaled3xGHIJKLMNOPQRSTUV);
      _mm512_storeu_ps(c3 + 32, vscaled3xWXYZabcdefghijkl);
      _mm512_storeu_ps(c3 + 48, vscaled3xmnopqrstuvwxyz01);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      _mm512_storeu_ps(c2 + 0, vscaled2x0123456789ABCDEF);
      _mm512_storeu_ps(c2 + 16, vscaled2xGHIJKLMNOPQRSTUV);
      _mm512_storeu_ps(c2 + 32, vscaled2xWXYZabcdefghijkl);
      _mm512_storeu_ps(c2 + 48, vscaled2xmnopqrstuvwxyz01);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      _mm512_storeu_ps(c1 + 0, vscaled1x0123456789ABCDEF);
      _mm512_storeu_ps(c1 + 16, vscaled1xGHIJKLMNOPQRSTUV);
      _mm512_storeu_ps(c1 + 32, vscaled1xWXYZabcdefghijkl);
      _mm512_storeu_ps(c1 + 48, vscaled1xmnopqrstuvwxyz01);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      _mm512_storeu_ps(c0 + 0, vscaled0x0123456789ABCDEF);
      _mm512_storeu_ps(c0 + 16, vscaled0xGHIJKLMNOPQRSTUV);
      _mm512_storeu_ps(c0 + 32, vscaled0xWXYZabcdefghijkl);
      _mm512_storeu_ps(c0 + 48, vscaled0xmnopqrstuvwxyz01);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a -= kc;
      nc -= 64;
    } else {
      // Prepare mask for valid 32-bit elements (depends on nc).
      const __mmask16 vmask0 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 0) & 0xFFFF));
      const __mmask16 vmask1 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 16) & 0xFFFF));
      const __mmask16 vmask2 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 32) & 0xFFFF));
      const __mmask16 vmask3 = _cvtu32_mask16((uint32_t) ((((UINT64_C(1) << nc) - 1) >> 48) & 0xFFFF));
      _mm512_mask_storeu_ps(c3 + 0, vmask0, vscaled3x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c3 + 16, vmask1, vscaled3xGHIJKLMNOPQRSTUV);
      _mm512_mask_storeu_ps(c3 + 32, vmask2, vscaled3xWXYZabcdefghijkl);
      _mm512_mask_storeu_ps(c3 + 48, vmask3, vscaled3xmnopqrstuvwxyz01);
      _mm512_mask_storeu_ps(c2 + 0, vmask0, vscaled2x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c2 + 16, vmask1, vscaled2xGHIJKLMNOPQRSTUV);
      _mm512_mask_storeu_ps(c2 + 32, vmask2, vscaled2xWXYZabcdefghijkl);
      _mm512_mask_storeu_ps(c2 + 48, vmask3, vscaled2xmnopqrstuvwxyz01);
      _mm512_mask_storeu_ps(c1 + 0, vmask0, vscaled1x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c1 + 16, vmask1, vscaled1xGHIJKLMNOPQRSTUV);
      _mm512_mask_storeu_ps(c1 + 32, vmask2, vscaled1xWXYZabcdefghijkl);
      _mm512_mask_storeu_ps(c1 + 48, vmask3, vscaled1xmnopqrstuvwxyz01);
      _mm512_mask_storeu_ps(c0 + 0, vmask0, vscaled0x0123456789ABCDEF);
      _mm512_mask_storeu_ps(c0 + 16, vmask1, vscaled0xGHIJKLMNOPQRSTUV);
      _mm512_mask_storeu_ps(c0 + 32, vmask2, vscaled0xWXYZabcdefghijkl);
      _mm512_mask_storeu_ps(c0 + 48, vmask3, vscaled0xmnopqrstuvwxyz01);
      nc = 0;
    }
  } while (nc != 0);
  // Release tile config
  //  _tile_release();
  __asm__ volatile ("tilerelease" ::);
  #endif  // defined(__x86_64__)
}
