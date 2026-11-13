// The voxel store's multi-level read (experiment X1c: clipmap levels of hashed faces, xlionrender_gi_ref.glsl writes them), shared by the gather
// (xlionrender_gi_gather_frag.glsl, VOX_READ_SCREEN: the screen-space reach test too) and the published light's resolve
// (xlionrender_gi_publish_comp.glsl - gi_published_store_plan_2026-11-08.md). The includer declares GIClip[] (uvec4), GITable[] and push.TableBase
#ifndef XLIONRENDER_VOX_LEVELS_GLSL
#define XLIONRENDER_VOX_LEVELS_GLSL
const int clip_dim_v = 64, clip_levels_v = 4;
float TF(in const uint i) { return uintBitsToFloat(GITable[push.TableBase + i]); }
const uint vox_slots_v = 1u << 21;        // (gpu_trace::vox_slots_v)
const uint vox_probe_base_v = 3u * vox_slots_v + 4u;        // (Milestone A: the movers' probes after the faces - gpu_trace::vox_probe_base_v)
const uint vox_movers_v = 64u;        // (gpu_trace::mover_max_v; after the probes, 64 x 12 x 8 rays' scratch entries)
const float vox_layers_v = 4.0;            // (xlionrender_gi_ref_comp.glsl)
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
#ifndef VOX_LEAN_V
const bool vox_lean_v = true;
#else
#define vox_lean_v VOX_LEAN_V      // (an includer's switch: the GI bricks' check, XLION_GI_BRICKS's flags)
#endif
// (A15, the lean moments the store keeps - xlionrender_gi_ref.glsl VoxStore: the light's slope across the normals that fed a face, by component, at this reader's normal; a ratio, clamped)
float VoxLeanScale(in const vec4 v, in const uint Wv, in const uvec4 dU, in const vec3 n, in const int a)
{
    if (!vox_lean_v) return 1.0;
    const int  u = (a + 1) % 3, w = (a + 2) % 3;
    const float El2 = unpackHalf2x16(Wv).y, Lm = dot(v.rgb, vec3(0.2126, 0.7152, 0.0722)) / v.w;
    const vec2 EL = unpackHalf2x16(dU.y & ~1u), El = unpackHalf2x16(dU.z);
    if (Lm <= 1e-4 || v.w < 8.0) return 1.0;
    const float Var = max(0.5 * (El2 - dot(El, El)), 0.01);      // (the leans' spread, one number for both: the spare words hold six)
    const vec2  S = (EL - Lm * El) / Var;
    // (2026-11-12, the flashing at thin edges and on young faces - 95 % of it was this lean: no extrapolation past the normals that fed the face (2 sigma of
    // their spread), trusted as its samples build (8 -> 256), at most 2x either way; was 0.25-4x from 8 samples on)
    vec2 Dn = vec2(n[u], n[w]) - El; const float Lim = 2.0 * sqrt(Var); if (dot(Dn, Dn) > Lim * Lim) Dn *= Lim / length(Dn);
    return mix(1.0, clamp(1.0 + dot(S, Dn) / Lm, 0.5, 2.0), clamp((v.w - 8.0) / 248.0, 0.0, 1.0));
}
bool  g_ClipFloat = false;     // (experiment X1c: the faces hold floats)
float g_ClipTrust = 0.0;       // (X1c: how much the last ClipLevel can be believed: its samples, how far from its edge)
uint g_VoxBucket = 0u;      // (XLION_VOX_AO_BUCKET: the records' kind a read takes, 1: a contact's; the faces' keys + 64)
#ifdef VOX_READ_SCREEN
// B1 (XLION_VOX_REACH=<margin m>, word 36; 0: off - b1_door_leak_design_2026-11-07.md): a corner the point cannot reach without crossing what the
// screen shows weighs almost nothing, the others take its share (Godot SDFGI's probe occlusion, at read time against the depth buffer). The
// segment from the point to the corner in the face's plane, lifted by the margin along the normal, at 3 points: a point the scene's depth
// shows something in front of (by more than the margin) is hidden. ponytail: screen space only - an occluder off screen does not count
mat4  g_ReachW2C = mat4(1.0);
float g_ReachMargin = 0.0;
float VoxReach(in const vec3 A, in const vec3 B)
{
    const vec3  Eye  = vec3(TF(25u), TF(26u), TF(27u));
    const ivec2 Size = textureSize(GPreDepth, 0);
    const mat4  C2W  = mat4(TF(0u), TF(1u), TF(2u), TF(3u), TF(4u), TF(5u), TF(6u), TF(7u), TF(8u), TF(9u), TF(10u), TF(11u), TF(12u), TF(13u), TF(14u), TF(15u));
    float Seen = 0.0;
    for (int k = 1; k <= 3; ++k)
    {
        const vec3 X = mix(A, B, float(k) / 3.0) - Eye;                         // (camera-centred)
        const vec4 C = g_ReachW2C * vec4(X, 1.0);
        if (C.w <= 0.0) { Seen += 1.0; continue; }
        const vec2 uv = C.xy / C.w * 0.5 + 0.5;
        if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) { Seen += 1.0; continue; }
        const ivec2 q = min(ivec2(uv * vec2(Size)), Size - 1);
        const vec4  Ws = C2W * vec4(uv * 2.0 - 1.0, texelFetch(GPreDepth, q, 0).r, 1.0);
        Seen += length(Ws.xyz / Ws.w) + g_ReachMargin >= length(X) ? 1.0 : 0.0;
    }
    return max(Seen / 3.0, 0.02);
}
#endif
vec4 ClipLevel(in const int L, in const vec3 Pw, in const vec3 n, in const vec3 EyeL, in const float Cell0)
{
    const float Cell = Cell0 * float(1 << L);
    const ivec3 Org  = ivec3(floor(EyeL / Cell)) - ivec3(clip_dim_v / 2);
    const vec3  g    = Pw / Cell - vec3(Org) - 0.5;
    if (any(lessThan(g, vec3(0.0))) || any(greaterThanEqual(g, vec3(clip_dim_v - 1)))) return vec4(0.0);
    const ivec3 i = ivec3(floor(g));
    const vec3  f = g - vec3(i);
    vec3  Sum = vec3(0.0);
    float SumW = 0.0, Samples = 0.0;
    for (int a = 0; a < 3; ++a)
    {
        const float wa = n[a] * n[a];
        if (wa < 0.02) continue;
        const uint Face = uint(2 * a + (n[a] < 0.0 ? 1 : 0)) + 64u * g_VoxBucket;
        vec3  Num = vec3(0.0);
        float Den = 0.0, Raw = 0.0;
        vec3 gv = vec3(Org) + g; gv[a] *= vox_layers_v;     // (X1c: finer along the face's own axis)
        const ivec3 iv = ivec3(floor(gv));
        const vec3  fc = g_ClipFloat ? gv - vec3(iv) : f;
        for (int c = 0; c < 8; ++c)
        {
            const ivec3 o = ivec3(c & 1, (c >> 1) & 1, (c >> 2) & 1), q = i + o;
            const float t = (o.x != 0 ? fc.x : 1.0 - fc.x) * (o.y != 0 ? fc.y : 1.0 - fc.y) * (o.z != 0 ? fc.z : 1.0 - fc.z);
            if (g_ClipFloat)      // (X1c: the sparse store: a face no surface fed, or lost to another, adds nothing)
            {
                const uvec2 h = VoxHash(L, iv + o, Face); const uint e = VoxEntryOf(h);
                // (each corner its mean, trusted up to 32 samples: by raw count a corner with thousands of samples drowned one with dozens, the light outdoors over the floor indoors)
                const uvec4 dU = GIClip[e + 1u];
                float Reach = 1.0;
#ifdef VOX_READ_SCREEN
                if (g_ReachMargin > 0.0 && L <= 1)      // (the fine levels: a coarse cell's corners are far - the leak is a fine level's)
                {
                    vec3 Cw = (vec3(iv + o) + 0.5) * Cell; Cw[a] = Pw[a];        // (the corner in the face's plane through the point)
                    Reach = VoxReach(Pw + n * g_ReachMargin, Cw + n * g_ReachMargin);
                }
#endif
                if (uintBitsToFloat(dU.x) == VoxTagOf(h)) { const vec4 v = uintBitsToFloat(GIClip[e]); if (v.w > 0.0) { const float cw = min(v.w, 32.0) * t * Reach; Num += v.rgb / v.w * VoxLeanScale(v, dU.w, dU, n, a) * cw; Den += cw; Raw += min(v.w, 256.0) * t; } }
                continue;
            }
            const uint  e = (uint(L) * uint(clip_dim_v * clip_dim_v * clip_dim_v) + uint(q.x) + uint(q.y) * uint(clip_dim_v) + uint(q.z) * uint(clip_dim_v * clip_dim_v)) * 6u + Face;
            const uvec4 v = GIClip[e];
            Num += vec3(v.xyz) * t; Den += float(v.w) * t;
        }
        if (Den > (g_ClipFloat ? 1e-3 : 1.0)) { Sum += Num / Den * wa; SumW += wa; Samples += Raw * wa; }
    }
    if (g_ClipFloat && SumW > 0.0)
    {
        const vec3 Inside = min(g, vec3(clip_dim_v - 1) - g);
        g_ClipTrust = clamp(Samples / SumW / (TF(35u) > 0.0 ? TF(35u) : 256.0), 0.0, 1.0) * clamp(min(Inside.x, min(Inside.y, Inside.z)) / 4.0, 0.0, 1.0);
    }
    return SumW > 0.0 ? vec4(Sum / SumW, SumW) : vec4(0.0);
}
// X1c's read: from the finest level, each as much as it can be believed, the rest from its parent: new cells and a level's edge fade in
vec4 VoxReadLevels(in const vec3 Pcam, in const vec3 n)
{
#ifdef VOX_READ_SCREEN
    g_ReachMargin = uintBitsToFloat(GITable[push.TableBase + 36u]);
    if (g_ReachMargin > 0.0) g_ReachW2C = inverse(mat4(TF(0u), TF(1u), TF(2u), TF(3u), TF(4u), TF(5u), TF(6u), TF(7u), TF(8u), TF(9u), TF(10u), TF(11u), TF(12u), TF(13u), TF(14u), TF(15u)));
#endif
    const vec3 EyeL = vec3(TF(22u), TF(23u), TF(24u)), Pw = Pcam + vec3(TF(25u), TF(26u), TF(27u));
    vec3 Sum = vec3(0.0), Last = vec3(0.0); float Left = 1.0; bool bAny = false;
    for (int L = 0; L < clip_levels_v && Left > 1e-3; ++L)
    {
        const vec4 R = ClipLevel(L, Pw, n, EyeL, TF(21u));
        if (R.w <= 0.0) continue;
        Sum += R.xyz * (Left * g_ClipTrust); Left *= 1.0 - g_ClipTrust;
        Last = R.xyz; bAny = true;
    }
    return bAny ? vec4(Sum + Last * Left, 1.0) : vec4(0.0);      // (what no level vouches for: the coarsest that has anything)
}
// ---- The published light (XLION_VOX_PUBLISHED, gi_published_store_plan_2026-11-08.md): today's whole multi-level read resolved once per
// corner of the finest level that holds a point (xlionrender_gi_publish_comp.glsl), the pixels reading those corners - one level, 8
// corners a face - instead of every level's. The includer declares Pub[] (uvec4) and PubU[] (uint, the same buffer). Layout, in uvec4:
// [0], [1] the request lists' counts by frame parity (x); [2] statistics (x pixels read published, y fell back to the multi-level read,
// z requests dropped: the list full); [3] spare; [4, 4 + 16384) a bit per slot, this frame's requests already listed; then a published
// entry per slot (x, y: RGB as halves, y's high half 0; z the face's tag; w the frame it was resolved, bit 31: it holds light); then the
// two request lists (q.xyz, level << 8 | face)
#ifdef VOX_PUBLISHED_BUFFER
const uint pub_head_v = 4u, pub_bits_v = 16384u, pub_cap_v = 1u << 19;
const uint pub_entry0_v = pub_head_v + pub_bits_v, pub_list0_v = pub_entry0_v + vox_slots_v;
// The finest level whose window holds the point (ClipLevel's own test), -1: none
int PubLevel(in const vec3 Pw, in const vec3 EyeL, in const float Cell0)
{
    for (int L = 0; L < clip_levels_v; ++L)
    {
        const float Cell = Cell0 * float(1 << L);
        const ivec3 Org  = ivec3(floor(EyeL / Cell)) - ivec3(clip_dim_v / 2);
        const vec3  g    = Pw / Cell - vec3(Org) - 0.5;
        if (!(any(lessThan(g, vec3(0.0))) || any(greaterThanEqual(g, vec3(clip_dim_v - 1))))) return L;
    }
    return -1;
}
// A corner's centre in the world (the face grid is vox_layers_v finer along the face's axis a)
vec3 PubCornerWorld(in const ivec3 q, in const int L, in const int a)
{
    vec3 gv = vec3(q); gv[a] /= vox_layers_v;
    return (gv + 0.5) * (TF(21u) * float(1 << L));
}
#endif
#endif
