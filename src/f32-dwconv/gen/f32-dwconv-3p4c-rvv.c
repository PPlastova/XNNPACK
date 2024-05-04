#include <assert.h>

#include <riscv_vector.h>

#include <xnnpack/dwconv.h>

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
