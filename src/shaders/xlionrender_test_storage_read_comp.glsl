#version 450
// A test (xlionrender_test_gi_gpu.cpp): a storage image's texture, sampled after the dispatch that wrote it, into a buffer
layout(local_size_x = 8, local_size_y = 8) in;
layout(std430, set = 0, binding = 0) buffer out_b { vec4 Out[]; };
layout(set = 1, binding = 0) uniform sampler2D Source;

void main()
{
    Out[gl_GlobalInvocationID.y * 16u + gl_GlobalInvocationID.x] = texelFetch(Source, ivec2(gl_GlobalInvocationID.xy), 0);
}
