// The voxels of light (experiment X1c, xlionrender_gi_ref_comp.glsl keeps them; the gather has its own read): the cells, the sparse store's lookup and a read at a world point. The includer has the store as GIRefAccum
const int vox_dim_v = 64, vox_levels_v = 4;
const uint vox_slots_v = 1u << 21;        // (gpu_trace::vox_slots_v)
// A face's cell is finer along the face's own axis: surfaces that look the same way at different depths (a bench's front, a wall behind it) are told apart; a surface only fills the few layers it crosses
const float vox_layers_v = 8.0;
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
uint VoxHash(in const int L, in const ivec3 q, in const uint Face) { uint h = uint(q.x) * 73856093u ^ uint(q.y) * 19349663u ^ uint(q.z) * 83492791u ^ (uint(L) * 36u + Face) * 2654435761u; h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; return h; }
uint VoxEntryOf(in const uint h) { return (h % vox_slots_v) * 3u; }
float VoxTagOf(in const uint h) { return float(((h * 3266489917u) >> 8u) + 1u); }
vec4 VoxFace(in const int L, in const ivec3 q, in const uint Face) { const uint h = VoxHash(L, q, Face), e = VoxEntryOf(h); return GIRefAccum[e + 2u].x == VoxTagOf(h) ? GIRefAccum[e] : vec4(0.0); }
vec3 VoxRead(in const vec3 Pw, in const vec3 n, in const vec3 Anchor, in const float Cell0)
{
    int L = -1; ivec3 Org; vec3 g;
    while (VoxCell(Pw, Anchor, Cell0, L + 1, L, Org, g))          // (nothing here: the parents - the samples feed them too)
    {
        vec3 Sum = vec3(0.0); float SumW = 0.0;
        for (int a = 0; a < 3; ++a)
        {
            const float wa = n[a] * n[a];
            if (wa < 0.02) continue;
            vec3 Num = vec3(0.0); float Den = 0.0;
            vec3 gv = vec3(Org) + g; gv[a] *= vox_layers_v;
            const ivec3 i = ivec3(floor(gv)); const vec3 f = gv - vec3(i);
            for (int c = 0; c < 8; ++c)
            {
                const ivec3 o = ivec3(c & 1, (c >> 1) & 1, (c >> 2) & 1);
                const float t = (o.x != 0 ? f.x : 1.0 - f.x) * (o.y != 0 ? f.y : 1.0 - f.y) * (o.z != 0 ? f.z : 1.0 - f.z);
                const vec4  v = VoxFace(L, i + o, uint(2 * a + (n[a] < 0.0 ? 1 : 0)));
                if (v.w > 0.0) { const float cw = min(v.w, 32.0) * t; Num += v.rgb / v.w * cw; Den += cw; }     // (as the read: each corner its mean, trusted up to 32 samples)
            }
            if (Den > 1e-3) { Sum += Num / Den * wa; SumW += wa; }
        }
        if (SumW > 0.5) return Sum / SumW;
    }
    return vec3(0.0);
}
