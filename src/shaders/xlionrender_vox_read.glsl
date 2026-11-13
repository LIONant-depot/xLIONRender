// The voxels of light (experiment X1c, xlionrender_gi_ref_comp.glsl keeps them; the gather has its own read): the cells, the sparse store's lookup and a read at a world point. The includer has the store as GIRefAccum
const int vox_dim_v = 64, vox_levels_v = 4;
const uint vox_slots_v = 1u << 21;        // (gpu_trace::vox_slots_v)
const uint vox_probe_base_v = 3u * vox_slots_v + 4u;        // (Milestone A: the movers' probes after the faces - gpu_trace::vox_probe_base_v)
const uint vox_movers_v = 64u;        // (gpu_trace::mover_max_v; after the probes, 64 x 12 x 8 rays' scratch entries)
// A face's cell is finer along the face's own axis: surfaces that look the same way at different depths (a bench's front, a wall behind it) are told apart; a surface only fills the few layers it crosses
const float vox_layers_v = 4.0;
const float vox_read_min_v = 16.0;        // (C38: the samples a level needs before a hit's bounce read takes it alone)
const float vox_short_v = 0.02, vox_sigmas_v = 8.0;        // (single noisy rays, a test at every one: slower and stricter than estimator::Add's 0.08 and 3)
bool VoxCell(in const vec3 Pw, in const vec3 Anchor, in const float Cell0, in const int First, out int L, out ivec3 Org, out vec3 g)
{
    for (L = First; L < vox_levels_v; ++L)
    {
        const float Cell = Cell0 * float(1 << L);
        Org = ivec3(floor(Anchor / Cell)) - ivec3(vox_dim_v / 2);
        g   = Pw / Cell - vec3(Org) - 0.5;
        if (all(greaterThanEqual(g, vec3(0.0))) && all(lessThan(g, vec3(vox_dim_v - 1)))) return true;
    }
    return false;
}
// Where a face lives: a fixed pool, one probe; a slot knows whose it is (its tag). Another face in it reads as nothing (idTech 8's: a collision drops), so a miss is one load
// (C51, 2026-11-13: .y, the tag's hash, is a second hash of the key, independent of .x - the tag was (h * odd) >> 8 of the slot's own h: two keys in one slot
// differ in h's top 11 bits only, so another face matched the tag 1 time in 2048, not 1 in 16 M. VoxEntryOf / VoxTagOf take the pair)
uint VoxPcg(in uint v) { const uint s = v * 747796405u + 2891336453u; const uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u; return (w >> 22u) ^ w; }
uvec2 VoxHash(in const int L, in const ivec3 q, in const uint Face) { uint h = uint(q.x) * 73856093u ^ uint(q.y) * 19349663u ^ uint(q.z) * 83492791u ^ (uint(L) * 4096u + Face) * 2654435761u; h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
    return uvec2(h, VoxPcg(uint(q.x) ^ VoxPcg(uint(q.y) ^ VoxPcg(uint(q.z) ^ VoxPcg(uint(L) * 4096u + Face))))); }
// A record's three uvec4 by role (gi_store_read_sector_plan_2026-11-08.md): X0 and X1 what a read takes (a face: its value; its tag, leans), X2 the
// writer's. An even slot X0 X1 X2, an odd one X2 X0 X1: X0, X1 one 32-byte sector for every record (they are 48 bytes). VoxEntryOf: X0; X1 = X0 + 1;
// VoxX2Of. Every kind: the tag X1.x, its last write's time X2.w, the direction bins' marker 1.0 X1.y
uint VoxEntryOf(in const uint h) { const uint s = h % vox_slots_v; return 3u * s + (s & 1u); }
uint VoxX2Of(in const uint e) { return (e % 3u) == 0u ? e + 2u : e - 1u; }
uint VoxEntryOf(in const uvec2 h) { return VoxEntryOf(h.x); }
float VoxTagOf(in const uvec2 h) { return float((h.y >> 8u) + 1u); }      // (24 bits: exact in a float, never 0)
vec4 VoxFace(in const int L, in const ivec3 q, in const uint Face) { const uvec2 h = VoxHash(L, q, Face); const uint e = VoxEntryOf(h); return GIRefAccum[e + 1u].x == VoxTagOf(h) ? GIRefAccum[e] : vec4(0.0); }
// (A15, the lean moments: xlionrender_gi_gather_frag.glsl VoxLeanScale's copy - the hits' read too)
float VoxLean(in const uint e, in const vec4 v, in const vec3 n, in const int a)
{
    const int  u = (a + 1) % 3, w = (a + 2) % 3;
    const float El2 = unpackHalf2x16(floatBitsToUint(GIRefAccum[e + 1u].w)).y, Lm = dot(v.rgb, vec3(0.2126, 0.7152, 0.0722)) / v.w;
    const uvec2 dU = floatBitsToUint(GIRefAccum[e + 1u].yz);
    const vec2 EL = unpackHalf2x16(dU.x & ~1u), El = unpackHalf2x16(dU.y);
    if (Lm <= 1e-4 || v.w < 8.0) return 1.0;
    const float Var = max(0.5 * (El2 - dot(El, El)), 0.01);
    vec2 Dn = vec2(n[u], n[w]) - El; const float Lim = 2.0 * sqrt(Var); if (dot(Dn, Dn) > Lim * Lim) Dn *= Lim / length(Dn);      // (as VoxLeanScale: no extrapolation, trusted with samples, 2x at most)
    return mix(1.0, clamp(1.0 + dot((EL - Lm * El) / Var, Dn) / Lm, 0.5, 2.0), clamp((v.w - 8.0) / 248.0, 0.0, 1.0));
}
vec3 VoxRead(in const vec3 Pw, in const vec3 n, in const vec3 Anchor, in const float Cell0)
{
    // (C38, 2026-11-13: the first level with any face was taken whole - a face of 1 or 2 samples at a hit fed its colour back into the store: the
    // magenta and red splats while flying (XLION_VOX_ABLATE=3, no bounce read at the hits: none of 240 frames; with it 25-49). Now the first level
    // with vox_read_min_v samples; short of that, the levels' means by their samples)
    int L = -1; ivec3 Org; vec3 g;
    vec3 Pool = vec3(0.0); float PoolN = 0.0;
    while (VoxCell(Pw, Anchor, Cell0, L + 1, L, Org, g))          // (nothing here: the parents - the samples feed them too)
    {
        vec3 Sum = vec3(0.0); float SumW = 0.0, Samples = 0.0;
        for (int a = 0; a < 3; ++a)
        {
            const float wa = n[a] * n[a];
            if (wa < 0.02) continue;
            vec3 Num = vec3(0.0); float Den = 0.0, Raw = 0.0;
            vec3 gv = vec3(Org) + g; gv[a] *= vox_layers_v;
            const ivec3 i = ivec3(floor(gv)); const vec3 f = gv - vec3(i);
            for (int c = 0; c < 8; ++c)
            {
                const ivec3 o = ivec3(c & 1, (c >> 1) & 1, (c >> 2) & 1);
                const float t = (o.x != 0 ? f.x : 1.0 - f.x) * (o.y != 0 ? f.y : 1.0 - f.y) * (o.z != 0 ? f.z : 1.0 - f.z);
                const uvec2  h = VoxHash(L, i + o, uint(2 * a + (n[a] < 0.0 ? 1 : 0))); const uint e = VoxEntryOf(h);
                const vec4  v = GIRefAccum[e + 1u].x == VoxTagOf(h) ? GIRefAccum[e] : vec4(0.0);
                if (v.w > 0.0) { const float cw = min(v.w, 32.0) * t; Num += v.rgb / v.w * VoxLean(e, v, n, a) * cw; Den += cw; Raw += v.w * t; }     // (as the read: each corner its mean, trusted up to 32 samples)
            }
            if (Den > 1e-3) { Sum += Num / Den * wa; SumW += wa; Samples += Raw * wa; }
        }
        if (SumW > 0.5)
        {
            const float Ns = Samples / SumW;
            if (Ns >= vox_read_min_v) return Sum / SumW;
            Pool += Sum / SumW * Ns; PoolN += Ns;
        }
    }
    return PoolN > 0.0 ? Pool / PoolN : vec3(0.0);
}
