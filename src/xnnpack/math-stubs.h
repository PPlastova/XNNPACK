// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*xnn_f16_unary_math_function)(
  size_t n,
  const void* input,
  void* output);

typedef void (*xnn_f32_unary_math_function)(
  size_t n,
  const float* input,
  float* output);

typedef void (*xnn_f32_ext_unary_math_function)(
  size_t n,
  const float* input,
  float* output_mantissa,
  float* output_exponent);

#define DECLARE_F16_UNARY_MATH_FUNCTION(fn_name) \
  void fn_name(                                  \
    size_t n,                                    \
    const void* input,                           \
    void* output);

#define DECLARE_F32_UNARY_MATH_FUNCTION(fn_name) \
  void fn_name(                                  \
    size_t n,                                    \
    const float* input,                          \
    float* output);

#define DECLARE_F16_F32_CVT_MATH_FUNCTION(fn_name) \
  void fn_name(                                    \
    size_t n,                                      \
    const void* input,                             \
    float* output);

#define DECLARE_F32_F16_CVT_MATH_FUNCTION(fn_name) \
  void fn_name(                                    \
    size_t n,                                      \
    const float* input,                            \
    void* output);

#define DECLARE_F32_QS8_CVT_MATH_FUNCTION(fn_name) \
  void fn_name(                                    \
    size_t n,                                      \
    const float* input,                            \
    int8_t* output,                                \
    int8_t output_zero_point);

#define DECLARE_F32_QU8_CVT_MATH_FUNCTION(fn_name) \
  void fn_name(                                    \
    size_t n,                                      \
    const float* input,                            \
    uint8_t* output,                               \
    uint8_t output_zero_point);

#define DECLARE_F32_EXT_UNARY_MATH_FUNCTION(fn_name) \
  void fn_name(                                      \
    size_t n,                                        \
    const float* input,                              \
    float* output_mantissa,                          \
    float* output_exponent);

DECLARE_F16_F32_CVT_MATH_FUNCTION(xnn_math_f16_f32_cvt__f16c)
DECLARE_F16_F32_CVT_MATH_FUNCTION(xnn_math_f16_f32_cvt__neon_int16)
DECLARE_F16_F32_CVT_MATH_FUNCTION(xnn_math_f16_f32_cvt__neon_int32)
DECLARE_F16_F32_CVT_MATH_FUNCTION(xnn_math_f16_f32_cvt__neonfp16)
DECLARE_F16_F32_CVT_MATH_FUNCTION(xnn_math_f16_f32_cvt__sse2_int16)
DECLARE_F16_F32_CVT_MATH_FUNCTION(xnn_math_f16_f32_cvt__sse2_int32)
DECLARE_F16_F32_CVT_MATH_FUNCTION(xnn_math_f16_f32_cvt__sse41_int16)
DECLARE_F16_F32_CVT_MATH_FUNCTION(xnn_math_f16_f32_cvt__sse41_int32)
DECLARE_F16_F32_CVT_MATH_FUNCTION(xnn_math_f16_f32_cvt__wasmsimd_int16)
DECLARE_F16_F32_CVT_MATH_FUNCTION(xnn_math_f16_f32_cvt__wasmsimd_int32)

DECLARE_F32_F16_CVT_MATH_FUNCTION(xnn_math_f32_f16_cvt__f16c)
DECLARE_F32_F16_CVT_MATH_FUNCTION(xnn_math_f32_f16_cvt__neon)
DECLARE_F32_F16_CVT_MATH_FUNCTION(xnn_math_f32_f16_cvt__neonfp16)
DECLARE_F32_F16_CVT_MATH_FUNCTION(xnn_math_f32_f16_cvt__scalar_bitcast)
DECLARE_F32_F16_CVT_MATH_FUNCTION(xnn_math_f32_f16_cvt__scalar_fabsf)
DECLARE_F32_F16_CVT_MATH_FUNCTION(xnn_math_f32_f16_cvt__sse2)
DECLARE_F32_F16_CVT_MATH_FUNCTION(xnn_math_f32_f16_cvt__sse41)
DECLARE_F32_F16_CVT_MATH_FUNCTION(xnn_math_f32_f16_cvt__wasmsimd)

DECLARE_F32_QS8_CVT_MATH_FUNCTION(xnn_math_f32_qs8_cvt__neon)
DECLARE_F32_QS8_CVT_MATH_FUNCTION(xnn_math_f32_qs8_cvt__neonv8)
DECLARE_F32_QS8_CVT_MATH_FUNCTION(xnn_math_f32_qs8_cvt__wasmsimd)

DECLARE_F32_QU8_CVT_MATH_FUNCTION(xnn_math_f32_qu8_cvt__neon)
DECLARE_F32_QU8_CVT_MATH_FUNCTION(xnn_math_f32_qu8_cvt__neonv8)
DECLARE_F32_QU8_CVT_MATH_FUNCTION(xnn_math_f32_qu8_cvt__wasmsimd)

DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_roundne__neon_addsub)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_roundne__neonv8)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_roundne__scalar_addsub)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_roundne__scalar_nearbyint)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_roundne__scalar_rint)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_roundne__sse_addsub)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_roundne__sse2_cvt)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_roundne__sse41)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_roundne__wasmsimd_addsub)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_roundne__wasmsimd_native)

DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_roundd__neon_addsub)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_roundd__neon_cvt)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_roundd__neonv8)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_roundd__scalar_addsub)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_roundd__scalar_cvt)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_roundd__scalar_floor)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_roundd__sse_addsub)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_roundd__sse2_cvt)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_roundd__sse41)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_roundd__wasmsimd_addsub)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_roundd__wasmsimd_cvt)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_roundd__wasmsimd_native)

DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_roundu__neon_addsub)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_roundu__neon_cvt)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_roundu__neonv8)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_roundu__scalar_addsub)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_roundu__scalar_ceil)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_roundu__scalar_cvt)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_roundu__sse_addsub)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_roundu__sse2_cvt)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_roundu__sse41)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_roundu__wasmsimd_addsub)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_roundu__wasmsimd_cvt)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_roundu__wasmsimd_native)

DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_roundz__neon_addsub)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_roundz__neon_cvt)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_roundz__neonv8)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_roundz__scalar_addsub)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_roundz__scalar_cvt)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_roundz__scalar_trunc)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_roundz__sse_addsub)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_roundz__sse2_cvt)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_roundz__sse41)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_roundz__wasmsimd_addsub)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_roundz__wasmsimd_cvt)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_roundz__wasmsimd_native)

DECLARE_F16_UNARY_MATH_FUNCTION(xnn_math_f16_exp__neonfp16arith_rr2_p3)

DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_exp__avx_rr2_p5)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_exp__avx2_rr2_lut8_p3_perm)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_exp__avx2_rr2_lut8_p4_perm)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_exp__avx2_rr2_p5)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_exp__avx512f_rr2_lut16_p3_perm)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_exp__avx512f_rr2_lut16_p3_perm_scalef)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_exp__avx512f_rr2_lut32_p2_perm2)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_exp__avx512f_rr2_lut32_p2_perm2_scalef)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_exp__avx512f_rr2_p5)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_exp__avx512f_rr2_p5_scalef)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_exp__neonfma_rr2_lut64_p2)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_exp__neonfma_rr2_p5)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_exp__sse2_rr2_lut64_p2)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_exp__sse2_rr2_p5)

DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_expminus__avx2_rr1_p5)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_expminus__avx2_rr2_p5)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_expminus__neonfma_rr2_lut64_p2)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_expminus__neonfma_rr2_lut2048_p1)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_expminus__neonfma_rr2_p5)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_expminus__scalar_rr2_lut64_p2)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_expminus__scalar_rr2_lut2048_p1)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_expminus__scalar_rr2_p5)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_expminus__sse2_rr2_p5)

DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_expm1minus__avx_rr2_lut4_p4_perm)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_expm1minus__avx_rr2_lut16_p3)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_expm1minus__avx_rr2_p6)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_expm1minus__avx2_rr1_lut4_p4_perm)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_expm1minus__avx2_rr1_lut8_p4_perm)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_expm1minus__avx2_rr1_lut16_p3_gather)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_expm1minus__avx2_rr1_p6)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_expm1minus__avx512f_rr1_lut16_p3_perm)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_expm1minus__avx512f_rr1_p6)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_expm1minus__neon_rr2_lut16_p3)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_expm1minus__neon_rr2_p6)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_expm1minus__neonfma_rr1_lut16_p3)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_expm1minus__neonfma_rr1_p6)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_expm1minus__scalar_rr2_lut4_p4)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_expm1minus__scalar_rr2_lut8_p3)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_expm1minus__scalar_rr2_lut8_p4)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_expm1minus__scalar_rr2_lut16_p3)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_expm1minus__scalar_rr2_lut16_p4)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_expm1minus__scalar_rr2_p5)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_expm1minus__scalar_rr2_p6)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_expm1minus__sse2_rr2_lut16_p3)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_expm1minus__sse2_rr2_p6)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_expm1minus__wasmsimd_rr2_lut16_p3_andnot)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_expm1minus__wasmsimd_rr2_lut16_p3_max)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_expm1minus__wasmsimd_rr2_p6_andnot)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_expm1minus__wasmsimd_rr2_p6_max)

DECLARE_F32_EXT_UNARY_MATH_FUNCTION(xnn_math_f32_extexp__avx2_p5)
DECLARE_F32_EXT_UNARY_MATH_FUNCTION(xnn_math_f32_extexp__avx512f_p5)

DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__avx_rr2_lut64_p2_div)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__avx_rr2_p5_div)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__avx_rr2_p5_nr1)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__avx_rr2_p5_nr2)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__avx2_rr1_lut64_p2_gather_div)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__avx2_rr1_lut64_p2_gather_nr1fma)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__avx2_rr1_lut64_p2_gather_nr2fma)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__avx2_rr1_lut64_p2_gather_nr2fma1adj)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__avx2_rr1_p5_div)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__avx2_rr1_p5_nr1fma)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__avx2_rr1_p5_nr2fma)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__avx2_rr2_lut64_p2_gather_div)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__avx2_rr2_lut64_p2_gather_nr1fma)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__avx2_rr2_lut64_p2_gather_nr2fma)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__avx2_rr2_lut64_p2_gather_nr2fma1adj)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__avx2_rr2_p5_div)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__avx2_rr2_p5_nr1fma)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__avx2_rr2_p5_nr2fma)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__avx512f_rr1_lut16_p3_perm_scalef_div)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__avx512f_rr1_lut16_p3_perm_scalef_nr1fma)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__avx512f_rr1_lut16_p3_perm_scalef_nr1fma1adj)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__avx512f_rr1_lut32_p2_perm2_scalef_div)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__avx512f_rr1_lut32_p2_perm2_scalef_nr1fma)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__avx512f_rr1_lut32_p2_perm2_scalef_nr1fma1adj)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__avx512f_rr1_lut64_p2_gather_scalef_div)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__avx512f_rr1_lut64_p2_gather_scalef_nr1fma)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__avx512f_rr1_lut64_p2_gather_scalef_nr1fma1adj)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__avx512f_rr1_p5_scalef_div)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__avx512f_rr1_p5_scalef_nr1fma)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__avx512f_rr1_p5_scalef_nr1fma1adj)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__avx512f_rr2_lut16_p3_perm_scalef_div)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__avx512f_rr2_lut16_p3_perm_scalef_nr1fma)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__avx512f_rr2_lut16_p3_perm_scalef_nr1fma1adj)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__avx512f_rr2_lut32_p2_perm2_scalef_div)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma1adj)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__avx512f_rr2_lut64_p2_gather_scalef_div)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__avx512f_rr2_lut64_p2_gather_scalef_nr1fma)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__avx512f_rr2_lut64_p2_gather_scalef_nr1fma1adj)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__avx512f_rr2_p5_scalef_div)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__avx512f_rr2_p5_scalef_nr1fma)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__avx512f_rr2_p5_scalef_nr1fma1adj)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__neon_frac_p9_p10_nr1recps)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__neon_rr1_lut64_p2_nr2recps)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__neon_rr1_lut2048_p1_nr2recps)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__neon_rr1_p5_nr2recps)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__neon_rr2_lut64_p2_nr2recps)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__neon_rr2_lut2048_p1_nr2recps)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__neon_rr2_p5_nr2recps)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__neonfma_rr1_lut64_p2_div)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__neonfma_rr1_lut64_p2_nr1recps1fma)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__neonfma_rr1_lut64_p2_nr2fma)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__neonfma_rr1_lut64_p2_nr2recps)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__neonfma_rr1_lut2048_p1_div)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__neonfma_rr1_lut2048_p1_nr1recps1fma)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__neonfma_rr1_lut2048_p1_nr2fma)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__neonfma_rr1_lut2048_p1_nr2recps)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__neonfma_rr1_p5_div)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__neonfma_rr1_p5_nr1recps1fma)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__neonfma_rr1_p5_nr2fma)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__neonfma_rr1_p5_nr2recps)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__neonfma_rr2_lut64_p2_div)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__neonfma_rr2_lut64_p2_nr1recps1fma)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__neonfma_rr2_lut64_p2_nr2fma)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__neonfma_rr2_lut64_p2_nr2recps)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__neonfma_rr2_lut2048_p1_div)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__neonfma_rr2_lut2048_p1_nr1recps1fma)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__neonfma_rr2_lut2048_p1_nr2fma)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__neonfma_rr2_lut2048_p1_nr2recps)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__neonfma_rr2_p5_div)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__neonfma_rr2_p5_nr1recps1fma)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__neonfma_rr2_p5_nr2fma)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__neonfma_rr2_p5_nr2recps)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__scalar_rr2_lut64_p2_div)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__scalar_rr2_lut2048_p1_div)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__scalar_rr2_p5_div)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__sse2_rr2_lut64_p2_div)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__sse2_rr2_lut64_p2_nr1)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__sse2_rr2_lut64_p2_nr2)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__sse2_rr2_p5_div)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__sse2_rr2_p5_nr1)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__sse2_rr2_p5_nr2)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__wasmsimd_rr2_lut64_p2_div)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sigmoid__wasmsimd_rr2_p5_div)

DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sqrt__avx512f_nr1fma)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sqrt__avx512f_nr1fma1adj)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sqrt__avx512f_nr2fma)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sqrt__fma3_nr1fma)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sqrt__fma3_nr1fma1adj)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sqrt__fma3_nr2fma)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sqrt__neon_nr1rsqrts)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sqrt__neon_nr2rsqrts)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sqrt__neon_nr3rsqrts)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sqrt__neonfma_nr1fma)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sqrt__neonfma_nr1rsqrts1fma1adj)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sqrt__neonfma_nr2fma)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sqrt__neonfma_nr2fma1adj)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sqrt__neonfma_nr3fma)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sqrt__sse_hh1mac)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sqrt__sse_nr1mac)
DECLARE_F32_UNARY_MATH_FUNCTION(xnn_math_f32_sqrt__sse_nr2mac)

#ifdef __cplusplus
} /* extern "C" */
#endif
