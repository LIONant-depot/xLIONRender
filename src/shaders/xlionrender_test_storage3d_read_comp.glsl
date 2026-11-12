#version 450
// A test (xlionrender_test_gi_gpu.cpp): an xGPU 3D texture sampled with the hardware trilinear filter between texel centres, into a buffer
layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;
layout(std430, set = 0, binding = 0) buffer out_b { vec4 Out[]; };
layout(set = 1, binding = 0) uniform sampler3D Source;

void main()
{
    // the point halfway between texel centres i and i+1 on every axis (8 texels a side): (i + 1) / 8 in texture coordinates
    const uvec3 p = gl_GlobalInvocationID;
    if (any(greaterThanEqual(p, uvec3(7u)))) return;
    Out[(p.z * 7u + p.y) * 7u + p.x] = textureLod(Source, (vec3(p) + 1.0) / 8.0, 0.0);
}
