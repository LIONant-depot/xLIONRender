#version 450
// A5.3 (a5_compute_material_tiles_plan_2026-11-08.md): the local lights an 8 x 8 screen tile can see - the box of its visible points (the
// prepass's depth, camera-centred world) against each light's sphere. Exact for those pixels: a light whose sphere misses the box reaches
// none of them, every light that reaches one is listed (the pixel still tests its range). The tile pass reads this list instead of its 2 m
// world cell's (xlionrender_local_lights.glsl LocalLights; the cell's list kept a light whose sphere only touched the cell, ~half of them
// out of range of the pixel, Bistro Game 2026-11-08). A tile: 16 uints at 16 + 16 x tile - [0] the count (0xFFFF: more than fit, the
// cell path), then 30 light indices, two a uint, ascending (the sum's order is the cells'). [0] the tiles a row, [1] 1: written this frame
layout(local_size_x = 8, local_size_y = 8) in;
struct LocalLightData { vec4 PosRadius; vec4 Color; vec4 DirCosOuter; vec4 Shadow; };
layout(std430, set = 0, binding = 0) readonly buffer lights_b { LocalLightData Lights[]; };
layout(std430, set = 0, binding = 1) writeonly buffer tiles_b { uint Tiles[]; };
layout(set = 1, binding = 0) uniform sampler2D Depth;
layout(push_constant) uniform push_t { mat4 m_C2W; uint m_W, m_H, m_LightBase, m_nLights; } Push;

const uint kMaxLights = 30u;
shared uint sMin[3], sMax[3], sCount, sIdx[kMaxLights];
uint Ordered(in const float f) { const uint u = floatBitsToUint(f); return (u & 0x80000000u) != 0u ? ~u : u | 0x80000000u; }
float Unordered(in const uint u) { return uintBitsToFloat((u & 0x80000000u) != 0u ? u & 0x7FFFFFFFu : ~u); }

void main()
{
    const uint  lid = gl_LocalInvocationIndex;
    const uvec2 p   = gl_GlobalInvocationID.xy;
    if (lid == 0u) { for (int a = 0; a < 3; ++a) { sMin[a] = 0xFFFFFFFFu; sMax[a] = 0u; } sCount = 0u; }
    barrier();
    if (p.x < Push.m_W && p.y < Push.m_H)
    {
        const float z = texelFetch(Depth, ivec2(p), 0).r;
        if (z < 1.0)
        {
            const vec4 W = Push.m_C2W * vec4((vec2(p) + 0.5) / vec2(Push.m_W, Push.m_H) * 2.0 - 1.0, z, 1.0);
            const vec3 P = W.xyz / W.w;
            for (int a = 0; a < 3; ++a) { atomicMin(sMin[a], Ordered(P[a])); atomicMax(sMax[a], Ordered(P[a])); }
        }
    }
    barrier();
    const bool bAny = sMax[0] != 0u;
    if (bAny)
    {
        const vec3 Lo = vec3(Unordered(sMin[0]), Unordered(sMin[1]), Unordered(sMin[2]));
        const vec3 Hi = vec3(Unordered(sMax[0]), Unordered(sMax[1]), Unordered(sMax[2]));
        for (uint i = lid; i < Push.m_nLights; i += 64u)
        {
            const vec4 S = Lights[Push.m_LightBase + i].PosRadius;
            const vec3 d = max(max(Lo - S.xyz, S.xyz - Hi), vec3(0.0));
            if (dot(d, d) < S.w * S.w)
            {
                const uint k = atomicAdd(sCount, 1u);
                if (k < kMaxLights) sIdx[k] = i;
            }
        }
    }
    barrier();
    const uint TX = (Push.m_W + 7u) / 8u, t = 16u + 16u * (gl_WorkGroupID.y * TX + gl_WorkGroupID.x);
    if (lid == 0u)
    {
        const uint n = sCount;
        if (n > kMaxLights) Tiles[t] = 0xFFFFu;
        else
        {
            for (uint a = 1u; a < n; ++a) { const uint v = sIdx[a]; uint b = a; for (; b > 0u && sIdx[b - 1u] > v; --b) sIdx[b] = sIdx[b - 1u]; sIdx[b] = v; }
            Tiles[t] = n;
            for (uint k = 0u; k < 15u; ++k) Tiles[t + 1u + k] = (2u * k < n ? sIdx[2u * k] : 0u) | ((2u * k + 1u < n ? sIdx[2u * k + 1u] : 0u) << 16);
        }
        if (gl_WorkGroupID.x == 0u && gl_WorkGroupID.y == 0u) { Tiles[0] = TX; Tiles[1] = 1u; }
    }
}
