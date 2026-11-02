#version 450
// A test (xlionrender_test_gi_gpu.cpp): xgpu::compute's storage images: each texel its place and a value the frame gives, in a texture read back by another shader
layout(local_size_x = 8, local_size_y = 8) in;
layout(set = 2, binding = 0, rgba16f) uniform writeonly image2D Target;
layout(push_constant) uniform push_t { float m_Value; } Push;

void main()
{
    imageStore(Target, ivec2(gl_GlobalInvocationID.xy), vec4(float(gl_GlobalInvocationID.x + 1u) / 16.0, float(gl_GlobalInvocationID.y + 1u) / 16.0, Push.m_Value, 1.0));
}
