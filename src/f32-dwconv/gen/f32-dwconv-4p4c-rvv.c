#include <assert.h>

#include <riscv_vector.h>

#include <xnnpack/dwconv.h>


void xnn_f32_dwconv_ukernel_4p4c__rvv(
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

      vfloat32m1_t vacc0123 = vacc0123p0;

      vse32_v_f32m1(output, vacc0123, vl);
      output += vl;
      c -= vl;
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}