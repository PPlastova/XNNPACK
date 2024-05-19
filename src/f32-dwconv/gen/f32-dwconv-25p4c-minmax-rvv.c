#include <assert.h>

#include <riscv_vector.h>

#include <xnnpack/dwconv.h>


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
  size_t vl_max = vsetvlmax_e32m1();
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

    const float* w = weights;
    size_t vl = vl_max;
    for (size_t c = channels; c != 0; ) {
      if XNN_UNLIKELY(c < vl_max) {
        vl = vsetvl_e32m1(c);
      }
      vfloat32m1_t vacc0123p0 = vle32_v_f32m1(w, vl);
      w += vl_max;

      const vfloat32m1_t vi0x0123 = vle32_v_f32m1(i0, vl);
      i0 += vl;

      const vfloat32m1_t vk0x0123 = vle32_v_f32m1(w, vl);
      w += vl_max;
      vacc0123p0 = vfadd_vv_f32m1(vacc0123p0, vfmul_vv_f32m1(vi0x0123, vk0x0123, vl), vl);

      const vfloat32m1_t vi1x0123 = vle32_v_f32m1(i1, vl);
      i1 += vl;

      const vfloat32m1_t vk1x0123 = vle32_v_f32m1(w, vl);
      w += vl_max;
      vacc0123p0 = vfadd_vv_f32m1(vacc0123p0, vfmul_vv_f32m1(vi1x0123, vk1x0123, vl), vl);

      const vfloat32m1_t vi2x0123 = vle32_v_f32m1(i2, vl);
      i2 += vl;

      const vfloat32m1_t vk2x0123 = vle32_v_f32m1(w, vl);
      w += vl_max;
      vacc0123p0 = vfadd_vv_f32m1(vacc0123p0, vfmul_vv_f32m1(vi2x0123, vk2x0123, vl), vl);

      const vfloat32m1_t vi3x0123 = vle32_v_f32m1(i3, vl);
      i3 += vl;

      const vfloat32m1_t vk3x0123 = vle32_v_f32m1(w, vl);
      w += vl_max;
      vacc0123p0 = vfadd_vv_f32m1(vacc0123p0, vfmul_vv_f32m1(vi3x0123, vk3x0123, vl), vl);

      const vfloat32m1_t vi4x0123 = vle32_v_f32m1(i4, vl);
      i4 += vl;

      const vfloat32m1_t vk4x0123 = vle32_v_f32m1(w, vl);
      w += vl_max;
      vacc0123p0 = vfadd_vv_f32m1(vacc0123p0, vfmul_vv_f32m1(vi4x0123, vk4x0123, vl), vl);

      const vfloat32m1_t vi5x0123 = vle32_v_f32m1(i5, vl);
      i5 += vl;

      const vfloat32m1_t vk5x0123 = vle32_v_f32m1(w, vl);
      w += vl_max;
      vacc0123p0 = vfadd_vv_f32m1(vacc0123p0, vfmul_vv_f32m1(vi5x0123, vk5x0123, vl), vl);

      const vfloat32m1_t vi6x0123 = vle32_v_f32m1(i6, vl);
      i6 += vl;

      const vfloat32m1_t vk6x0123 = vle32_v_f32m1(w, vl);
      w += vl_max;
      vacc0123p0 = vfadd_vv_f32m1(vacc0123p0, vfmul_vv_f32m1(vi6x0123, vk6x0123, vl), vl);

      const vfloat32m1_t vi7x0123 = vle32_v_f32m1(i7, vl);
      i7 += vl;

      const vfloat32m1_t vk7x0123 = vle32_v_f32m1(w, vl);
      w += vl_max;
      vacc0123p0 = vfadd_vv_f32m1(vacc0123p0, vfmul_vv_f32m1(vi7x0123, vk7x0123, vl), vl);

      const vfloat32m1_t vi8x0123 = vle32_v_f32m1(i8, vl);
      i8 += vl;

      const vfloat32m1_t vk8x0123 = vle32_v_f32m1(w, vl);
      w += vl_max;
      vacc0123p0 = vfadd_vv_f32m1(vacc0123p0, vfmul_vv_f32m1(vi8x0123, vk8x0123, vl), vl);

      const vfloat32m1_t vi9x0123 = vle32_v_f32m1(i9, vl);
      i9 += vl;

      const vfloat32m1_t vk9x0123 = vle32_v_f32m1(w, vl);
      w += vl_max;
      vacc0123p0 = vfadd_vv_f32m1(vacc0123p0, vfmul_vv_f32m1(vi9x0123, vk9x0123, vl), vl);

      const vfloat32m1_t vi10x0123 = vle32_v_f32m1(i10, vl);
      i10 += vl;

      const vfloat32m1_t vk10x0123 = vle32_v_f32m1(w, vl);
      w += vl_max;
      vacc0123p0 = vfadd_vv_f32m1(vacc0123p0, vfmul_vv_f32m1(vi10x0123, vk10x0123, vl), vl);

      const vfloat32m1_t vi11x0123 = vle32_v_f32m1(i11, vl);
      i11 += vl;

      const vfloat32m1_t vk11x0123 = vle32_v_f32m1(w, vl);
      w += vl_max;
      vacc0123p0 = vfadd_vv_f32m1(vacc0123p0, vfmul_vv_f32m1(vi11x0123, vk11x0123, vl), vl);

      const vfloat32m1_t vi12x0123 = vle32_v_f32m1(i12, vl);
      i12 += vl;

      const vfloat32m1_t vk12x0123 = vle32_v_f32m1(w, vl);
      w += vl_max;
      vacc0123p0 = vfadd_vv_f32m1(vacc0123p0, vfmul_vv_f32m1(vi12x0123, vk12x0123, vl), vl);

      const vfloat32m1_t vi13x0123 = vle32_v_f32m1(i13, vl);
      i13 += vl;

      const vfloat32m1_t vk13x0123 = vle32_v_f32m1(w, vl);
      w += vl_max;
      vacc0123p0 = vfadd_vv_f32m1(vacc0123p0, vfmul_vv_f32m1(vi13x0123, vk13x0123, vl), vl);

      const vfloat32m1_t vi14x0123 = vle32_v_f32m1(i14, vl);
      i14 += vl;

      const vfloat32m1_t vk14x0123 = vle32_v_f32m1(w, vl);
      w += vl_max;
      vacc0123p0 = vfadd_vv_f32m1(vacc0123p0, vfmul_vv_f32m1(vi14x0123, vk14x0123, vl), vl);

      const vfloat32m1_t vi15x0123 = vle32_v_f32m1(i15, vl);
      i15 += vl;

      const vfloat32m1_t vk15x0123 = vle32_v_f32m1(w, vl);
      w += vl_max;
      vacc0123p0 = vfadd_vv_f32m1(vacc0123p0, vfmul_vv_f32m1(vi15x0123, vk15x0123, vl), vl);

      const vfloat32m1_t vi16x0123 = vle32_v_f32m1(i16, vl);
      i16 += vl;

      const vfloat32m1_t vk16x0123 = vle32_v_f32m1(w, vl);
      w += vl_max;
      vacc0123p0 = vfadd_vv_f32m1(vacc0123p0, vfmul_vv_f32m1(vi16x0123, vk16x0123, vl), vl);

      const vfloat32m1_t vi17x0123 = vle32_v_f32m1(i17, vl);
      i17 += vl;

      const vfloat32m1_t vk17x0123 = vle32_v_f32m1(w, vl);
      w += vl_max;
      vacc0123p0 = vfadd_vv_f32m1(vacc0123p0, vfmul_vv_f32m1(vi17x0123, vk17x0123, vl), vl);

      const vfloat32m1_t vi18x0123 = vle32_v_f32m1(i18, vl);
      i18 += vl;

      const vfloat32m1_t vk18x0123 = vle32_v_f32m1(w, vl);
      w += vl_max;
      vacc0123p0 = vfadd_vv_f32m1(vacc0123p0, vfmul_vv_f32m1(vi18x0123, vk18x0123, vl), vl);

      const vfloat32m1_t vi19x0123 = vle32_v_f32m1(i19, vl);
      i19 += vl;

      const vfloat32m1_t vk19x0123 = vle32_v_f32m1(w, vl);
      w += vl_max;
      vacc0123p0 = vfadd_vv_f32m1(vacc0123p0, vfmul_vv_f32m1(vi19x0123, vk19x0123, vl), vl);

      const vfloat32m1_t vi20x0123 = vle32_v_f32m1(i20, vl);
      i20 += vl;

      const vfloat32m1_t vk20x0123 = vle32_v_f32m1(w, vl);
      w += vl_max;
      vacc0123p0 = vfadd_vv_f32m1(vacc0123p0, vfmul_vv_f32m1(vi20x0123, vk20x0123, vl), vl);

      const vfloat32m1_t vi21x0123 = vle32_v_f32m1(i21, vl);
      i21 += vl;

      const vfloat32m1_t vk21x0123 = vle32_v_f32m1(w, vl);
      w += vl_max;
      vacc0123p0 = vfadd_vv_f32m1(vacc0123p0, vfmul_vv_f32m1(vi21x0123, vk21x0123, vl), vl);

      const vfloat32m1_t vi22x0123 = vle32_v_f32m1(i22, vl);
      i22 += vl;

      const vfloat32m1_t vk22x0123 = vle32_v_f32m1(w, vl);
      w += vl_max;
      vacc0123p0 = vfadd_vv_f32m1(vacc0123p0, vfmul_vv_f32m1(vi22x0123, vk22x0123, vl), vl);

      const vfloat32m1_t vi23x0123 = vle32_v_f32m1(i23, vl);
      i23 += vl;

      const vfloat32m1_t vk23x0123 = vle32_v_f32m1(w, vl);
      w += vl_max;
      vacc0123p0 = vfadd_vv_f32m1(vacc0123p0, vfmul_vv_f32m1(vi23x0123, vk23x0123, vl), vl);

      const vfloat32m1_t vi24x0123 = vle32_v_f32m1(i24, vl);
      i24 += vl;

      const vfloat32m1_t vk24x0123 = vle32_v_f32m1(w, vl);
      w += vl_max;
      vacc0123p0 = vfadd_vv_f32m1(vacc0123p0, vfmul_vv_f32m1(vi24x0123, vk24x0123, vl), vl);

      vfloat32m1_t vacc0123 = vfmax_vf_f32m1(vacc0123p0, vmin, vl);
      vacc0123 = vfmin_vf_f32m1(vacc0123, vmax, vl);

      vse32_v_f32m1(output, vacc0123, vl);
      output += vl;
      c -= vl;
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
