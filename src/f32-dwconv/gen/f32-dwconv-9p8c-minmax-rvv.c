#include <assert.h>

#include <riscv_vector.h>

#include <xnnpack/dwconv.h>


void xnn_f32_dwconv_minmax_ukernel_9p8c__rvv(
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

      w += 80;


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
