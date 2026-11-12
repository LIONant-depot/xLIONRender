#version 450
// A test (xlionrender_test_gi_gpu.cpp): xGPU 3D textures - each texel its place, written by a compute into an image3D
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;
layout(set = 2, binding = 0, rgba16f) uniform writeonly image3D Target;

void main()
{
    const uvec3 p = gl_GlobalInvocationID;
    imageStore(Target, ivec3(p), vec4(float(p.x) / 8.0, float(p.y) / 8.0, float(p.z) / 8.0, 1.0));
}
