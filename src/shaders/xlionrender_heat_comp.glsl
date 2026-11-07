#version 450
// XLION_HEAT (measurement, 2026-11-08: where a pass's cost lands on the screen): the GI gather's clock a pixel (xlionrender_gi_gather_frag.glsl
// built with XLION_HEAT: its subgroup's cycles, stalls included) as a colour - log2, 2^10 cycles blue to 2^18 red - into a texture the tone
// mapper shows over a grey image (xlionrender_tonemap_frag.glsl, Params.w); each pixel cleared for the next frame; the header's [0] the
// cycles / 256 summed, [1] the pixels (the HEAT line's mean)
#extension GL_KHR_shader_subgroup_arithmetic : require
layout(local_size_x = 8, local_size_y = 8) in;
layout(std430, set = 0, binding = 0) buffer heat_b { uint Heat[]; };
layout(set = 2, binding = 0, rgba16f) writeonly uniform image2D Dst;
layout(push_constant) uniform push_t { uint m_W, m_H, m_Pad0, m_Pad1; } Push;

vec3 Ramp(float t)          // blue, cyan, green, yellow, red
{
    const vec3 C[5] = vec3[5](vec3(0.0, 0.1, 1.0), vec3(0.0, 0.9, 1.0), vec3(0.0, 1.0, 0.1), vec3(1.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0));
    const float x = clamp(t, 0.0, 1.0) * 4.0;
    const int   i = min(int(x), 3);
    return mix(C[i], C[i + 1], x - float(i));
}

void main()
{
    const uvec2 p = gl_GlobalInvocationID.xy;
    const bool  bIn = p.x < Push.m_W && p.y < Push.m_H;
    const uint  k = 4u + p.y * Push.m_W + p.x;
    const uint  c = bIn ? Heat[k] : 0u;
    if (bIn)
    {
        Heat[k] = 0u;
        imageStore(Dst, ivec2(p), c == 0u ? vec4(0.0) : vec4(Ramp((log2(float(c)) - 10.0) / 8.0), 1.0));
    }
    const uint Sum = subgroupAdd(c >> 8u), n = subgroupAdd(c != 0u ? 1u : 0u);
    if (subgroupElect() && n != 0u) { atomicAdd(Heat[0], Sum); atomicAdd(Heat[1], n); }
}
