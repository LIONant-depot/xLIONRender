#version 450
// The sun's virtual shadow pages: the request pass (features/shadows/xlionrender_shadow_vsm.h; documentation/lighting_strategy.md
// sec. 7). Per pixel of the main view's depth: its world position, its footprint in light space (the smaller of the forward and
// backward differences to its neighbours, per screen axis, so a silhouette does not ask for a coarse level), the level whose texel
// fits it (as the shading's ShadowVSM picks: shaders/xlionrender_shadow_vsm.glsl), and a bit for every page its filter's taps fall in
// (the dilation). Then what the shading will find in this frame's table (the same walk to coarser levels): counters for the report.
// The CPU reads both a ring late (instance::renderShadows).
// Two dispatches (m_Pad1): the full-resolution pass (m_Pad0 0) asks for each pixel's own level and counts; the coarse pass at a quarter
// of the pixels (a thread a 2 x 2 quad, its first pixel; m_Pad0 the contact-hardening's coarser levels) asks only for those levels'
// pages - a page of which covers thousands of pixels, the request's dilation (3 to 4 texels) covers the quad. In one pass they cost
// 0.23 of 0.54 ms on Bistro at 3093 x 1493: a wave's lanes run in lock-step, so skipping 3 of 4 pixels in it saved nothing
layout(local_size_x = 8, local_size_y = 8) in;

const int kMaxLevels   = 12;
const int kPageTexels  = 128;
const int kCounters    = 32;        // after the bits: 0 pixels, 1 fine, 2 coarser, 3 none, 4 boundary-crossing, 5-16 per level, 17-24 log2(footprint / texel) + 4
const float kDepthRange = 4.0;      // vsm::depth_range_v
// After the counters, per table entry: the largest z and the largest 1 - z of its pixels (z in 0..1 over the level's depth range, as
// float bits: ordered as uints) - a later epoch moves the request into its own basis with them (vsm::Remap)

struct frame_t
{
    mat4 C2W;                       // the main view's clip -> camera-centred world
    vec4 Eye;                       // the camera's world position
    vec4 LX, LY, LZ;                // world -> light rows
    vec4 P0;                        // x: level 0 texel, y: levels, z: window (pages), w: the resolution bias (levels)
    vec4 Levels[kMaxLevels];        // per level: xy the window's first page, z page size, w depth centre
};
layout(std430, set = 0, binding = 0) readonly buffer frame_b { frame_t F[]; };
layout(std430, set = 0, binding = 1) coherent buffer req_b  { uint Req[]; };
layout(std430, set = 0, binding = 2) readonly buffer cell_b { uint Cells[]; };     // the cell ring: this frame's page table at TableBase
layout(set = 1, binding = 0) uniform sampler2D Depth;
layout(push_constant) uniform push_t { uint m_Frame, m_ReqBase, m_TableBase, m_W, m_H, m_BitUints, m_Pad0, m_Pad1; } Push;     // m_Pad0: coarser levels, m_Pad1: 1 the coarse pass

shared uint s_Count[kCounters];

bool World(ivec2 p, out vec3 L)
{
    if (any(lessThan(p, ivec2(0))) || p.x >= int(Push.m_W) || p.y >= int(Push.m_H)) return false;
    const float d = texelFetch(Depth, p, 0).r;
    if (d >= 1.0) return false;                                                     // the sky
    const vec2 Ndc = (vec2(p) + 0.5) / vec2(Push.m_W, Push.m_H) * 2.0 - 1.0;
    vec4 W = F[Push.m_Frame].C2W * vec4(Ndc, d, 1.0);
    const vec3 P = W.xyz / W.w + F[Push.m_Frame].Eye.xyz;
    L = vec3(dot(P, F[Push.m_Frame].LX.xyz), dot(P, F[Push.m_Frame].LY.xyz), dot(P, F[Push.m_Frame].LZ.xyz));
    return true;
}

float AxisFootprint(ivec2 p, ivec2 Step, vec3 L)
{
    vec3 A, B;
    const bool bA = World(p + Step, A), bB = World(p - Step, B);
    const float a = bA ? length(A.xy - L.xy) : 1e30, b = bB ? length(L.xy - B.xy) : 1e30;
    return min(a, b);
}

uint Entry(int Level, ivec2 Page)
{
    const int W = int(F[Push.m_Frame].P0.z);
    const ivec2 T = ((Page % W) + W) % W;
    const uint i = uint((Level * W + T.y) * W + T.x);
    return (Cells[Push.m_TableBase + (i >> 1)] >> ((i & 1u) * 16u)) & 0xFFFFu;
}

bool InWindow(int Level, ivec2 Page)
{
    const vec4 Lv = F[Push.m_Frame].Levels[Level];
    const ivec2 Rel = Page - ivec2(Lv.xy);
    const int W = int(F[Push.m_Frame].P0.z);
    return all(greaterThanEqual(Rel, ivec2(0))) && all(lessThan(Rel, ivec2(W)));
}

void Request(int Level, ivec2 Page, uint Z, uint InvZ)
{
    if (!InWindow(Level, Page)) return;
    const int W = int(F[Push.m_Frame].P0.z);
    const ivec2 T = ((Page % W) + W) % W;
    const uint i = uint((Level * W + T.y) * W + T.x), w = Push.m_ReqBase + (i >> 5), b = 1u << (i & 31u);
    if ((Req[w] & b) == 0u) atomicOr(Req[w], b);                   // (read first: most pixels find their page asked for already)
    const uint d = Push.m_ReqBase + Push.m_BitUints + kCounters + 2u * i;
    if (Req[d] < Z)          atomicMax(Req[d], Z);
    if (Req[d + 1u] < InvZ)  atomicMax(Req[d + 1u], InvZ);
}

void main()
{
    if (gl_LocalInvocationIndex < kCounters) s_Count[gl_LocalInvocationIndex] = 0u;
    barrier();

    const bool  bCoarse = Push.m_Pad1 != 0u;
    const ivec2 p = ivec2(gl_GlobalInvocationID.xy) * (bCoarse ? 2 : 1);
    vec3 L;
    if (World(p, L))
    {
        const float t0 = F[Push.m_Frame].P0.x;
        const int   nLevels = int(F[Push.m_Frame].P0.y);
        const float fx = AxisFootprint(p, ivec2(1, 0), L), fy = AxisFootprint(p, ivec2(0, 1), L);
        float Foot = max(fx < 1e29 ? fx : 0.0, fy < 1e29 ? fy : 0.0);         // the shading's max(|dFdx|, |dFdy|) of the light-space xy
        if (Foot <= 0.0) Foot = t0;                                             // (an isolated pixel: no neighbour to measure with)
        int Level = clamp(int(floor(log2(Foot / t0) + F[Push.m_Frame].P0.w)), 0, nLevels - 1);
        if (!bCoarse)
        {
            atomicAdd(s_Count[0], 1u);
            atomicAdd(s_Count[17 + clamp(int(floor(log2(Foot / (t0 * float(1 << Level))))) + 4, 0, 7)], 1u);
        }

        // The level the shading starts at, its taps' pages requested (the filter: 4 x 4 texels around the point, xlionrender_shadow_vsm.glsl)
        int  First = -1, Found = -1;
        bool bCross = false;
        for (int l = Level; l < nLevels; ++l)
        {
            const float Texel = t0 * float(1 << l);
            const ivec2 Lo = ivec2(floor(L.xy / Texel - 0.5)) - 1, Hi = Lo + 3;
            const ivec2 PLo = ivec2(floor(vec2(Lo) / float(kPageTexels))), PHi = ivec2(floor(vec2(Hi) / float(kPageTexels)));
            if (!InWindow(l, PLo) || !InWindow(l, PHi)) continue;                     // past this level's window: a coarser one
            if (First < 0)
            {
                First = l;
                if (!bCoarse) atomicAdd(s_Count[5 + min(l, 11)], 1u);
                bCross = any(notEqual(PLo, PHi));
                {   // (the request also covers the shading's normal offset: 2 texels more around the taps)
            const ivec2 RLo = ivec2(floor(vec2(Lo - 2) / float(kPageTexels))), RHi = ivec2(floor(vec2(Hi + 2) / float(kPageTexels)));
            const vec4  Lv  = F[Push.m_Frame].Levels[l];
            const float Half = Lv.z * F[Push.m_Frame].P0.z * kDepthRange;
            const float z   = clamp((L.z - (Lv.w - Half)) / (2.0 * Half), 0.0, 1.0);
            // (rounded up to 1/256: still conservative, and after a few pixels most find the page's range holding theirs: no atomic)
            const uint  Z = floatBitsToUint(max(ceil(z * 256.0) / 256.0, 1e-7)), InvZ = floatBitsToUint(max(ceil((1.0 - z) * 256.0) / 256.0, 1e-7));
            if (!bCoarse) for (int y = RLo.y; y <= RHi.y; ++y) for (int x = RLo.x; x <= RHi.x; ++x) Request(l, ivec2(x, y), Z, InvZ);
            // Contact-hardening (a sun with a disc, xlionrender_shadow_vsm.glsl): the blocker search and the wide penumbrae read the coarser
            // levels' taps too
            for (int k = 1; k <= int(Push.m_Pad0) && l + k < nLevels; ++k)
            {
                const int   lk  = l + k;
                const float Tk  = t0 * float(1 << lk);
                const ivec2 Lk  = ivec2(floor(L.xy / Tk - 0.5)) - 3;         // (the 4 x 4 taps, the normal offset's 2 more)
                const ivec2 KLo = ivec2(floor(vec2(Lk) / float(kPageTexels))), KHi = ivec2(floor(vec2(Lk + 7) / float(kPageTexels)));
                const vec4  Lvk = F[Push.m_Frame].Levels[lk];
                const float Hk  = Lvk.z * F[Push.m_Frame].P0.z * kDepthRange;
                const float zk  = clamp((L.z - (Lvk.w - Hk)) / (2.0 * Hk), 0.0, 1.0);
                const uint  Zk = floatBitsToUint(max(ceil(zk * 256.0) / 256.0, 1e-7)), InvZk = floatBitsToUint(max(ceil((1.0 - zk) * 256.0) / 256.0, 1e-7));
                for (int y = KLo.y; y <= KHi.y; ++y) for (int x = KLo.x; x <= KHi.x; ++x) Request(lk, ivec2(x, y), Zk, InvZk);
            }
        }
            }
            if (bCoarse) break;
            // What the shading finds this frame: every tap's page in the table, or a coarser level
            bool bAll = true;
            for (int y = PLo.y; y <= PHi.y; ++y) for (int x = PLo.x; x <= PHi.x; ++x) bAll = bAll && Entry(l, ivec2(x, y)) != 0u;
            if (bAll) { Found = l; break; }
        }
        if (First >= 0 && !bCoarse) atomicAdd(s_Count[Found < 0 ? 3 : Found == First ? 1 : 2], 1u);    // (past every window: lit, beyond the shadows' reach)
        if (bCross && !bCoarse) atomicAdd(s_Count[4], 1u);
    }

    barrier();
    if (gl_LocalInvocationIndex < kCounters && s_Count[gl_LocalInvocationIndex] != 0u) atomicAdd(Req[Push.m_ReqBase + Push.m_BitUints + gl_LocalInvocationIndex], s_Count[gl_LocalInvocationIndex]);
}
