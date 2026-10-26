#version 450
// GPU cluster culling's Hi-Z pyramid in one dispatch (features/cull/xlionrender_gpu_cull.h), in the manner of AMD's single pass downsampler:
// a work group reads a 128 x 128 block of the scene's depth (sampled where the pass left it: no copy) and writes the pyramid's levels 0 to
// 6 of it (level 0: half the view's size), the last group to finish (a counter) the levels past 6 from the others' level 6. A texel is the
// farthest depth (0 near, 1 far) of the 2 x 2 under it; a level's size is the one below's halved, rounded up, a missing texel adds nothing,
// so a texel of level L covers the view's pixels [x 2^(L + 1), (x + 1) 2^(L + 1)). Shared memory and one atomic: no subgroup operations
layout(local_size_x = 256) in;

layout(std430, set = 0, binding = 0) coherent buffer hiz_b { float HiZ[]; };
layout(std430, set = 0, binding = 1) coherent buffer cnt_b { uint Counter[]; };       // 0 between dispatches (the last group resets it)
layout(set = 1, binding = 0) uniform sampler2D Depth;

layout(push_constant) uniform push_t { uint m_W, m_H, m_GroupsX, m_nGroups, m_nLevels, m_Pad0, m_Pad1, m_Pad2; } Push;     // the view's size

shared float S[2][256];
shared bool  sLast;

// Level L's size and first float (level 0: the view halved, rounded up)
void Level(uint L, out uint W, out uint H, out uint Offset)
{
    W = (Push.m_W + 1u) / 2u; H = (Push.m_H + 1u) / 2u; Offset = 0u;
    for (uint i = 0u; i < L; ++i) { Offset += W * H; W = (W + 1u) / 2u; H = (H + 1u) / 2u; }
}

void Store(uint L, uvec2 p, float v)
{
    uint W, H, Off;
    Level(L, W, H, Off);
    if (L < Push.m_nLevels && p.x < W && p.y < H) HiZ[Off + p.y * W + p.x] = v;
}

float DepthAt(ivec2 p)          // (outside the view: nothing)
{
    return p.x < int(Push.m_W) && p.y < int(Push.m_H) ? texelFetch(Depth, p, 0).r : 0.0;
}

void main()
{
    const uint  t = gl_LocalInvocationIndex;
    const uvec2 G = gl_WorkGroupID.xy;
    const uvec2 T = uvec2(t % 16u, t / 16u);

    // Level 0: a 4 x 4 block a thread (of the group's 64 x 64), each texel the farthest of its 2 x 2 pixels; level 1: its 2 x 2; level 2: its texel
    uint W0, H0, Off0;
    Level(0u, W0, H0, Off0);
    float L1[2][2];
    for (uint j = 0u; j < 2u; ++j)
        for (uint i = 0u; i < 2u; ++i)
        {
            float v1 = 0.0;
            for (uint b = 0u; b < 2u; ++b)
                for (uint a = 0u; a < 2u; ++a)
                {
                    const uvec2 p = G * 64u + T * 4u + uvec2(i * 2u + a, j * 2u + b);
                    float v0 = 0.0;
                    if (p.x < W0 && p.y < H0)
                    {
                        const ivec2 q = ivec2(p * 2u);
                        v0 = max(max(DepthAt(q), DepthAt(q + ivec2(1, 0))), max(DepthAt(q + ivec2(0, 1)), DepthAt(q + ivec2(1, 1))));
                        HiZ[Off0 + p.y * W0 + p.x] = v0;
                    }
                    v1 = max(v1, v0);
                }
            Store(1u, G * 32u + T * 2u + uvec2(i, j), v1);
            L1[j][i] = v1;
        }
    const float v2 = max(max(L1[0][0], L1[0][1]), max(L1[1][0], L1[1][1]));
    Store(2u, G * 16u + T, v2);
    S[0][t] = v2;

    // Levels 3 to 6 in shared memory (16 x 16 -> 8 x 8 -> 4 x 4 -> 2 x 2 -> 1)
    uint src = 0u, n = 16u;
    for (uint L = 3u; L <= 6u; ++L)
    {
        barrier();
        const uint m = n / 2u;
        if (t < m * m)
        {
            const uint x = t % m, y = t / m;
            const float v = max(max(S[src][(2u * y) * n + 2u * x], S[src][(2u * y) * n + 2u * x + 1u]), max(S[src][(2u * y + 1u) * n + 2u * x], S[src][(2u * y + 1u) * n + 2u * x + 1u]));
            S[1u - src][y * m + x] = v;
            Store(L, G * m + uvec2(x, y), v);
        }
        src = 1u - src; n = m;
    }
    if (Push.m_nLevels <= 7u) return;

    // The last group to finish: the levels past 6, from every group's level 6
    memoryBarrierBuffer();
    barrier();
    if (t == 0u) sLast = atomicAdd(Counter[0], 1u) == Push.m_nGroups - 1u;
    barrier();
    if (!sLast) return;
    if (t == 0u) Counter[0] = 0u;
    for (uint L = 7u; L < Push.m_nLevels; ++L)
    {
        uint SW, SH, SOff, DW, DH, DOff;
        Level(L - 1u, SW, SH, SOff);
        Level(L, DW, DH, DOff);
        for (uint i = t; i < DW * DH; i += 256u)
        {
            const uint x = i % DW, y = i / DW;
            float v = 0.0;
            for (uint b = 0u; b < 2u; ++b)
                for (uint a = 0u; a < 2u; ++a)
                    if (2u * x + a < SW && 2u * y + b < SH) v = max(v, HiZ[SOff + (2u * y + b) * SW + 2u * x + a]);
            HiZ[DOff + i] = v;
        }
        memoryBarrierBuffer();
        barrier();
    }
}
