#version 450
// The sun's virtual shadow pages: the request pass (features/shadows/xlionrender_shadow_vsm.h; documentation/lighting_strategy.md
// sec. 7). Per pixel of the main view's depth: its world position, its footprint in light space (the smaller of the forward and
// backward differences to its neighbours, per screen axis, so a silhouette does not ask for a coarse level), the level whose texel
// fits it (as the shading's ShadowVSM picks: shaders/xlionrender_shadow_vsm.glsl), and a bit for every page its filter's taps fall in
// (the dilation). Then what the shading will find in this frame's table (the same walk to coarser levels): counters for the report.
// The CPU reads both a ring late (instance::renderShadows).
// The contact-hardening's coarser levels (m_Pad0 of them) shared by each 2 x 2 quad's lanes: a lane asks for one of them from its own
// pixel - a page of those covers thousands of pixels, the request's dilation (3 to 4 texels) covers the quad. (They were a second
// dispatch at a quarter of the pixels, each rebuilding its 4 neighbours' positions: 0.14 of Bistro's 0.42 ms at 3093 x 1493; in one
// pass with every lane asking for every level, 0.23: a wave's lanes run in lock-step)
layout(local_size_x = 8, local_size_y = 8) in;

const int kMaxLevels   = 12;
const int kPageTexels  = 128;
const int kCounters    = 32;        // after the bits: 0 pixels, 1 fine, 2 coarser, 3 none, 4 boundary-crossing, 5-16 per level, 17-24 log2(footprint / texel) + 4
const float kDepthRange = 4.0;      // vsm::depth_range_v
const float kLevelBand  = 0.1;      // xlionrender_shadow_vsm.glsl kVSMLevelBand: the levels' cross-fade (its pages asked for too)
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
layout(push_constant) uniform push_t { uint m_Frame, m_ReqBase, m_TableBase, m_W, m_H, m_BitUints, m_Pad0, m_Pad1; } Push;     // m_Pad0: coarser levels

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

// The light-space positions, the workgroup's 8 x 8 pixels and one more around (w 0: none - the sky, off the view):
// a pixel's and its 4 neighbours' read once for the group (each pixel rebuilt 5 of them: 0.14 of Bistro's 0.44 ms)
shared vec4 s_L[10 * 10];

float TileFootprint(ivec2 l, ivec2 Step, vec3 L)
{
    const vec4 A = s_L[(l.y + Step.y) * 10 + l.x + Step.x], B = s_L[(l.y - Step.y) * 10 + l.x - Step.x];
    const float a = A.w != 0.0 ? length(A.xy - L.xy) : 1e30, b = B.w != 0.0 ? length(L.xy - B.xy) : 1e30;
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
    for (uint t = gl_LocalInvocationIndex; t < 100u; t += 64u)
    {
        vec3 Lt;
        const bool b = World(ivec2(gl_WorkGroupID.xy) * 8 - 1 + ivec2(int(t) % 10, int(t) / 10), Lt);
        s_L[t] = vec4(Lt, b ? 1.0 : 0.0);
    }
    barrier();

    const ivec2 l = ivec2(gl_LocalInvocationID.xy) + 1;
    const vec3  L = s_L[l.y * 10 + l.x].xyz;
    const int   q = int(gl_LocalInvocationID.x & 1u) + 2 * int(gl_LocalInvocationID.y & 1u);     // (the lane in its 2 x 2 quad)
    if (s_L[l.y * 10 + l.x].w != 0.0)
    {
        const float t0 = F[Push.m_Frame].P0.x;
        const int   nLevels = int(F[Push.m_Frame].P0.y);
        // the shading's footprint (xlionrender_shadow_vsm.glsl VSMFootprint): the distance to the eye times the pixel's angle
        float Foot;
        {
            const ivec2 pp = ivec2(gl_WorkGroupID.xy) * 8 + ivec2(gl_LocalInvocationID.xy);
            const vec2  N0 = (vec2(pp) + 0.5) / vec2(Push.m_W, Push.m_H) * 2.0 - 1.0;
            const vec4  A  = F[Push.m_Frame].C2W * vec4(N0, 1.0, 1.0), B = F[Push.m_Frame].C2W * vec4(N0 + vec2(2.0 / float(Push.m_W), 0.0), 1.0, 1.0);
            const vec4  C  = F[Push.m_Frame].C2W * vec4(N0 + vec2(0.0, 2.0 / float(Push.m_H)), 1.0, 1.0);
            const vec3  Da = normalize(A.xyz / A.w), Db = normalize(B.xyz / B.w), Dc = normalize(C.xyz / C.w);
            const float d  = texelFetch(Depth, pp, 0).r;
            const vec4  W  = F[Push.m_Frame].C2W * vec4(N0, d, 1.0);
            const float fx = TileFootprint(l, ivec2(1, 0), L), fy = TileFootprint(l, ivec2(0, 1), L);
            float Fl = max(fx < 1e29 ? fx : 0.0, fy < 1e29 ? fy : 0.0);            // the light-space derivatives (as the shading's)
            if (Fl <= 0.0) Fl = t0;                                                 // (an isolated pixel: no neighbour to measure with)
            Foot = max(min(Fl, 2.0 * length(W.xyz / W.w) * max(length(Db - Da), length(Dc - Da))), 1e-9);
        }
        int Level = clamp(int(floor(log2(Foot / t0) + F[Push.m_Frame].P0.w)), 0, nLevels - 1);
        atomicAdd(s_Count[0], 1u);
        atomicAdd(s_Count[17 + clamp(int(floor(log2(Foot / (t0 * float(1 << Level))))) + 4, 0, 7)], 1u);

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
                atomicAdd(s_Count[5 + min(l, 11)], 1u);
                bCross = any(notEqual(PLo, PHi));
                {   // (the request also covers the shading's normal offset: 2 texels more around the taps)
            const ivec2 RLo = ivec2(floor(vec2(Lo - 2) / float(kPageTexels))), RHi = ivec2(floor(vec2(Hi + 2) / float(kPageTexels)));
            const vec4  Lv  = F[Push.m_Frame].Levels[l];
            const float Half = Lv.z * F[Push.m_Frame].P0.z * kDepthRange;
            const float z   = clamp((L.z - (Lv.w - Half)) / (2.0 * Half), 0.0, 1.0);
            // (rounded up to 1/256: still conservative, and after a few pixels most find the page's range holding theirs: no atomic)
            const uint  Z = floatBitsToUint(max(ceil(z * 256.0) / 256.0, 1e-7)), InvZ = floatBitsToUint(max(ceil((1.0 - z) * 256.0) / 256.0, 1e-7));
            for (int y = RLo.y; y <= RHi.y; ++y) for (int x = RLo.x; x <= RHi.x; ++x) Request(l, ivec2(x, y), Z, InvZ);
            // The level band (xlionrender_shadow_vsm.glsl kVSMLevelBand): the last share of a level cross-fades into the next - its taps too
            const float Lx = log2(Foot / t0) + F[Push.m_Frame].P0.w;
            // (the request's footprint is the depth tiles', the shading's its own derivatives': near a level's ends they differ - the shading
            // wanting a level whose pages nobody asked for fell back to a coarser one, a hard seam along the pages' edges while moving, Tomas.
            // A margin each way: the next level from 0.3 below its start, the finer one up to 0.3 above it - below)
            if (l == int(floor(Lx)) && l + 1 < nLevels && fract(Lx) > 1.0 - max(uintBitsToFloat(Push.m_Pad1) > 0.0 ? uintBitsToFloat(Push.m_Pad1) : kLevelBand, 0.3))
            {
                const float T1  = t0 * float(2 << l);
                const ivec2 L1  = ivec2(floor(L.xy / T1 - 0.5)) - 3;
                const ivec2 BLo = ivec2(floor(vec2(L1) / float(kPageTexels))), BHi = ivec2(floor(vec2(L1 + 7) / float(kPageTexels)));
                const vec4  Lv1 = F[Push.m_Frame].Levels[l + 1];
                const float H1  = Lv1.z * F[Push.m_Frame].P0.z * kDepthRange;
                const float z1  = clamp((L.z - (Lv1.w - H1)) / (2.0 * H1), 0.0, 1.0);
                const uint  Z1 = floatBitsToUint(max(ceil(z1 * 256.0) / 256.0, 1e-7)), InvZ1 = floatBitsToUint(max(ceil((1.0 - z1) * 256.0) / 256.0, 1e-7));
                for (int y = BLo.y; y <= BHi.y; ++y) for (int x = BLo.x; x <= BHi.x; ++x) Request(l + 1, ivec2(x, y), Z1, InvZ1);
            }
            if (l == int(floor(Lx)) && l > 0 && fract(Lx) < 0.3)
            {
                const float T0  = t0 * float(1 << (l - 1));
                const ivec2 L0  = ivec2(floor(L.xy / T0 - 0.5)) - 3;
                const ivec2 FLo = ivec2(floor(vec2(L0) / float(kPageTexels))), FHi = ivec2(floor(vec2(L0 + 7) / float(kPageTexels)));
                const vec4  Lv0 = F[Push.m_Frame].Levels[l - 1];
                const float H0  = Lv0.z * F[Push.m_Frame].P0.z * kDepthRange;
                const float z0  = clamp((L.z - (Lv0.w - H0)) / (2.0 * H0), 0.0, 1.0);
                const uint  Z0 = floatBitsToUint(max(ceil(z0 * 256.0) / 256.0, 1e-7)), InvZ0 = floatBitsToUint(max(ceil((1.0 - z0) * 256.0) / 256.0, 1e-7));
                if (InWindow(l - 1, FLo) && InWindow(l - 1, FHi))
                    for (int y = FLo.y; y <= FHi.y; ++y) for (int x = FLo.x; x <= FHi.x; ++x) Request(l - 1, ivec2(x, y), Z0, InvZ0);
            }
            // Contact-hardening (a sun with a disc, xlionrender_shadow_vsm.glsl): the blocker search and the wide penumbrae read the coarser
            // levels' taps too (the quad's lanes share them)
            for (int k = q + 1; k <= int(Push.m_Pad0) && l + k < nLevels; k += 4)
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
            // What the shading finds this frame: every tap's page in the table, or a coarser level
            bool bAll = true;
            for (int y = PLo.y; y <= PHi.y; ++y) for (int x = PLo.x; x <= PHi.x; ++x) bAll = bAll && Entry(l, ivec2(x, y)) != 0u;
            if (bAll) { Found = l; break; }
        }
        if (First >= 0) atomicAdd(s_Count[Found < 0 ? 3 : Found == First ? 1 : 2], 1u);    // (past every window: lit, beyond the shadows' reach)
        if (bCross) atomicAdd(s_Count[4], 1u);
    }

    barrier();
    if (gl_LocalInvocationIndex < kCounters && s_Count[gl_LocalInvocationIndex] != 0u) atomicAdd(Req[Push.m_ReqBase + Push.m_BitUints + gl_LocalInvocationIndex], s_Count[gl_LocalInvocationIndex]);
}
