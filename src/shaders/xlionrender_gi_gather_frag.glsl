#version 450
// The GI gather (gi_gbuffer_gather_plan.md, stage 2): the half-resolution GI read without geometry - a full-screen pass a
// GI buffer, at half size, on the pixels the classification gave it (xlionrender_gi_gather_vert.glsl: tested EQUAL). The surface from
// the G-buffer prepass (xlionrender_gi_gather_common.glsl): its position from the depth, into mesh space through its instance's
// transform; its normal; its GI reference from the table. The read is mb_standard_pbr.frag GIPixelGI's per-pixel one (the surfels, then
// the local mean; no vertex read: no vertices here), into the same buffer the scene pass upsamples, with the same temporal reuse
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_ballot : require
#include "xlionrender_surfel_gi.glsl"
#include "xlionrender_gi_gather_common.glsl"

layout(early_fragment_tests) in;
struct InstanceData { vec4 Rows[3]; vec4 Material[3]; vec4 PrevRows[3]; };     // (xlionrender_static_geom_body.glsl)
layout(std430, set = 1, binding = 1) readonly buffer InstanceBuffer { InstanceData Instances[]; } instances;
layout(std430, set = 1, binding = 11) buffer GIScreenBuffer { uvec4 GIScreen[]; };
layout(std430, set = 1, binding = 13) readonly buffer GIScreenPrevBuffer { uvec4 GIScreenPrev[]; };
layout(std430, set = 1, binding = 20) readonly buffer AOBuffer { vec2 AOData[]; };    // the ambient occlusion on this grid (xlionrender_ao_comp.glsl): AO, distance

// The AO at this pixel, its 3 x 3 neighbours weighed by their distance's match (the denoiser's spatial pass: edge-aware)
float GatherAO(in const ivec2 P, in const uint W, in const uint H, in const float Dist)
{
    float S = 0.0, Wt = 0.0;
    for (int y = -1; y <= 1; ++y) for (int x = -1; x <= 1; ++x)
    {
        const ivec2 q = clamp(P + ivec2(x, y), ivec2(0), ivec2(int(W) - 1, int(H) - 1));
        const vec2  A = AOData[uint(q.y) * W + uint(q.x)];
        if (A.y <= 0.0) continue;
        const float w = max(0.0, 1.0 - abs(A.y - Dist) / (0.02 * Dist + 0.01)) * ((x == 0 && y == 0) ? 2.0 : 1.0);
        S += A.x * w; Wt += w;
    }
    return Wt > 1e-4 ? S / Wt : 1.0;
}
// (XLION_SSGI, push.AO 2: the AO pass's screen GI at this pixel, the near field's irradiance - after the AO's 4 x 4 filter)
vec3 GatherSSGI(in const ivec2 P, in const uint W, in const uint H)
{
    const uint N = W * H, i = uint(P.y) * W + uint(P.x);
    return max(vec3(AOData[N + i], AOData[2u * N + i].x), vec3(0.0));
}

// Experiment X1 (XLION_GI_CLIP, xlionrender_gi_clip_comp.glsl): the light at a world point for a normal, from the clipmap: the finest level
// holding it, 8 cells x the 3 faces the normal looks toward (squared cosines), the weighted light over the weights - cells no surfel
// reaches count for nothing. A level's edge blends into the next
layout(std430, set = 1, binding = 21) readonly buffer GIClipBuffer { uvec4 GIClip[]; };
layout(std430, set = 1, binding = 22) buffer GIRefSampleBuffer { vec4 GIRefSample[]; };        // the ground truth (XLION_GI_REF, xlionrender_gi_ref_comp.glsl)
layout(std430, set = 1, binding = 23) readonly buffer GIRefAccumBuffer { vec4 GIRefAccum[]; };
const int clip_dim_v = 64, clip_levels_v = 4;
float TF(in const uint i) { return uintBitsToFloat(GITable[push.TableBase + i]); }
const uint vox_slots_v = 1u << 21;        // (gpu_trace::vox_slots_v)
const float vox_layers_v = 4.0;            // (xlionrender_gi_ref_comp.glsl)
// Where a face lives: a fixed pool, one probe; a slot knows whose it is (its tag). Another face in it reads as nothing (idTech 8's: a collision drops), so a miss is one load
uint VoxHash(in const int L, in const ivec3 q, in const uint Face) { uint h = uint(q.x) * 73856093u ^ uint(q.y) * 19349663u ^ uint(q.z) * 83492791u ^ (uint(L) * 4096u + Face) * 2654435761u; h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; return h; }
uint VoxEntryOf(in const uint h) { return (h % vox_slots_v) * 3u; }
float VoxTagOf(in const uint h) { return float(((h * 3266489917u) >> 8u) + 1u); }
const bool vox_lean_v = true;
// (A15, the lean moments the store keeps - xlionrender_gi_ref.glsl VoxStore: the light's slope across the normals that fed a face, by component, at this reader's normal; a ratio, clamped)
float VoxLeanScale(in const vec4 v, in const uint Wv, in const uvec4 dU, in const vec3 n, in const int a)
{
    if (!vox_lean_v) return 1.0;
    const int  u = (a + 1) % 3, w = (a + 2) % 3;
    const float El2 = unpackHalf2x16(Wv).y, Lm = dot(v.rgb, vec3(0.2126, 0.7152, 0.0722)) / v.w;
    const vec2 EL = unpackHalf2x16(dU.z & ~1u), El = unpackHalf2x16(dU.w);
    if (Lm <= 1e-4 || v.w < 8.0) return 1.0;
    const float Var = max(0.5 * (El2 - dot(El, El)), 0.01);      // (the leans' spread, one number for both: the spare words hold six)
    const vec2  S = (EL - Lm * El) / Var;
    return clamp(1.0 + dot(S, vec2(n[u], n[w]) - El) / Lm, 0.25, 4.0);
}
bool  g_ClipFloat = false;     // (experiment X1c: the faces hold floats)
float g_ClipTrust = 0.0;       // (X1c: how much the last ClipLevel can be believed: its samples, how far from its edge)
uint g_VoxBucket = 0u;      // (XLION_VOX_AO_BUCKET: the records' kind a read takes, 1: a contact's; the faces' keys + 64)
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
                const uint h = VoxHash(L, iv + o, Face), e = VoxEntryOf(h);
                // (each corner its mean, trusted up to 32 samples: by raw count a corner with thousands of samples drowned one with dozens, the light outdoors over the floor indoors)
                const uvec4 dU = GIClip[e + 2u];
                if (uintBitsToFloat(dU.x) == VoxTagOf(h)) { const vec4 v = uintBitsToFloat(GIClip[e]); if (v.w > 0.0) { const float cw = min(v.w, 32.0) * t; Num += v.rgb / v.w * VoxLeanScale(v, GIClip[e + 1u].w, dU, n, a) * cw; Den += cw; Raw += min(v.w, 256.0) * t; } }
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
// The light around the mirror direction, from the rays this place's own face sent that way (xlionrender_gi_ref_comp.glsl VoxStoreDir): the three axes R leans on, by their squares, as a face read.
// Finest level that knows, the rest from its parent; what none vouches for: the place's irradiance (Fallback; EnvLighting then gives about its older ratio), so a new cell fades in
vec4 VoxBin(in const uvec4 q) { return vec4(vec3(q.xyz) / 256.0, float(q.w) / 16.0); }     // (a direction bin's fixed point sum: xlionrender_gi_ref.glsl VoxStoreDir)
vec4 VoxReflect(in const vec3 Pcam, in const vec3 n, in const vec3 Fallback)
{
    const vec3 EyeL = vec3(TF(22u), TF(23u), TF(24u)), Pw = Pcam + vec3(TF(25u), TF(26u), TF(27u));
    const vec3 Rd = reflect(normalize(Pcam), n), An = abs(n);
    const int  a = An.x >= An.y && An.x >= An.z ? 0 : An.y >= An.z ? 1 : 2, u = (a + 1) % 3, v = (a + 2) % 3;
    const uint Base = 6u + uint(2 * a + (n[a] < 0.0 ? 1 : 0)) * 5u;
    const uint  Bk[3] = uint[3](4u, Rd[u] > 0.0 ? 0u : 1u, Rd[v] > 0.0 ? 2u : 3u);
    const float Bw[3] = float[3](Rd[a] * Rd[a], Rd[u] * Rd[u], Rd[v] * Rd[v]);
    const float Cell0 = TF(21u);
    // A bin is wide: the rays of a floor's bin toward the street go up to about 55 degrees, into the sky; a reflection near the surface stays low, among the walls
    const float SkyTake = GITable[push.TableBase + 31u] != 0u ? 1.0 : smoothstep(0.0, 0.5, abs(Rd[a]));     // (no sky left out unless XLION_VOX_NO_BIN_PDF: the bin's own mean is meant to be right near the surface)
    vec3 Sum = vec3(0.0); float Left = 1.0;
    for (int L = 0; L < clip_levels_v && Left > 1e-3; ++L)
    {
        const float Cell = Cell0 * float(1 << L);
        const ivec3 Org  = ivec3(floor(EyeL / Cell)) - ivec3(clip_dim_v / 2);
        const vec3  g    = Pw / Cell - vec3(Org) - 0.5;
        if (any(lessThan(g, vec3(0.0))) || any(greaterThanEqual(g, vec3(clip_dim_v - 1)))) continue;
        const vec3  gv = vec3(Org) + g;
        const ivec3 iv = ivec3(floor(gv));
        const vec3  f  = gv - vec3(iv);
        vec3 Lit = vec3(0.0); float WSum = 0.0, Trust = 0.0;
        for (int b = 0; b < 3; ++b)
        {
            if (Bw[b] < 0.02) continue;
            vec3 Num = vec3(0.0); float Den = 0.0;
            for (int c = 0; c < 4; ++c)        // (the face's plane: along its axis the corner its samples rounded to)
            {
                ivec3 o = ivec3(0); o[u] = c & 1; o[v] = c >> 1; o[a] = f[a] >= 0.5 ? 1 : 0;
                const float t = (o[u] != 0 ? f[u] : 1.0 - f[u]) * (o[v] != 0 ? f[v] : 1.0 - f[v]);
                const uint  h = VoxHash(L, iv + o, Base + Bk[b]), e = VoxEntryOf(h);
                if (uintBitsToFloat(GIClip[e + 2u].x) == VoxTagOf(h)) { const vec4 w = VoxBin(GIClip[e]); if (w.w > 0.0) { const vec4 m0 = VoxBin(GIClip[e + 1u]); const vec4 m = m0 * ((1.0 - SkyTake) * smoothstep(8.0, 48.0, w.w - m0.w));      /* (the sky left out only where enough rays hit something: a few hits' mean flips between tiles: black squares, Tomas) */ const float cw = min(w.w, 128.0) * t, nl = w.w - m.w; Num += (nl > 0.5 ? (w.rgb - m.rgb) / nl : w.rgb / w.w) * cw; Den += cw; } }
            }
            if (Den > 1e-3) { Lit += Num / Den * Bw[b]; WSum += Bw[b]; Trust += Den * Bw[b]; }
        }
        if (WSum < 0.5) continue;
        const float Tr = clamp(Trust / WSum / 64.0, 0.0, 1.0);
        Sum += Lit / WSum * (Left * Tr); Left *= 1.0 - Tr;
    }
    return vec4(Sum + Fallback * Left, 1.0);
}
// XLION_VOX_DIRS (word 32: bit 3 on; bits 0, 1 level; bit 2 k 16; bits 8 to 15 roughness x 255; word 33 the frame; xlionrender_gi_ref.glsl VoxDirsAdd): the read, samples of the lobe, a position in the surface's plane by chance (bilinear over time)
uint DirsHash(inout uint s) { s ^= s >> 16; s *= 0x7FEB352Du; s ^= s >> 15; s *= 0x846CA68Bu; s ^= s >> 16; return s; }
vec4 VoxDirsRead(in const vec3 Pcam, in const vec3 n, in const vec4 Fallback, in const ivec2 Px)
{
    const uint  Cfg = GITable[push.TableBase + 32u];
    const int   L = int(Cfg & 3u); const uint k = (Cfg & 4u) != 0u ? 16u : 8u;
    const float Rg = float((Cfg >> 8) & 255u) / 255.0, al = max(Rg * Rg, 0.002), Cell = TF(21u) * float(1 << L);
    const vec3  Pw = Pcam + vec3(TF(25u), TF(26u), TF(27u)), Ve0 = -normalize(Pcam), An = abs(n);
    const int   a = An.x >= An.y && An.x >= An.z ? 0 : An.y >= An.z ? 1 : 2, u = (a + 1) % 3, v = (a + 2) % 3;
    const float sg = n[a] < 0.0 ? -1.0 : 1.0; const uint Face = uint(2 * a + (n[a] < 0.0 ? 1 : 0));
    const vec3  T = normalize(cross(abs(n.x) < 0.9 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0), n)), B = cross(n, T), Ve = vec3(dot(Ve0, T), dot(Ve0, B), max(dot(Ve0, n), 1e-3));
    const vec3  Vh = normalize(vec3(al * Ve.x, al * Ve.y, Ve.z));
    const float Lq = Vh.x * Vh.x + Vh.y * Vh.y;
    const vec3  T1 = Lq > 1e-9 ? vec3(-Vh.y, Vh.x, 0.0) / sqrt(Lq) : vec3(1.0, 0.0, 0.0), T2 = cross(Vh, T1);
    uint Seed = uint(Px.x) * 73856093u ^ uint(Px.y) * 19349663u ^ GITable[push.TableBase + 33u] * 83492791u;
    vec3 Sum = vec3(0.0);
    const bool bFixed = (Cfg & 16u) != 0u;      // (the affordable read: 4 fixed directions of the lobe, each over the 4 records around, 16 fetches, no noise)
    const int nS = bFixed ? 16 : 8;
    const vec2 Fx = fract(Pw[u] / Cell - 0.5) * vec2(1.0, 0.0) + fract(Pw[v] / Cell - 0.5) * vec2(0.0, 1.0);
    float SumW = 0.0;
    for (int i = 0; i < nS; ++i)
    {
        const float r1 = bFixed ? (float(i >> 2 & 1) + 0.5) * 0.5 : float(DirsHash(Seed)) / 4294967296.0, r2 = bFixed ? (float(i >> 3 & 1) * 0.5 + float(i >> 2 & 1) * 0.25 + 0.125) : float(DirsHash(Seed)) / 4294967296.0;
        const float j1 = bFixed ? float(i & 1) : float(DirsHash(Seed)) / 4294967296.0, j2 = bFixed ? float(i >> 1 & 1) : float(DirsHash(Seed)) / 4294967296.0;
        const float Wt = bFixed ? ((i & 1) != 0 ? Fx.x : 1.0 - Fx.x) * ((i & 2) != 0 ? Fx.y : 1.0 - Fx.y) : 1.0;
        const float r = sqrt(r1), Phi = 6.2831853 * r2, t1 = r * cos(Phi), s = 0.5 * (1.0 + Vh.z), t2 = (1.0 - s) * sqrt(max(0.0, 1.0 - t1 * t1)) + s * r * sin(Phi);
        const vec3  Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * Vh, Ne = normalize(vec3(al * Nh.x, al * Nh.y, max(0.0, Nh.z)));
        vec3 Le = reflect(-Ve, Ne); if (Le.z <= 1e-4) Le.z = abs(Le.z) + 1e-4;
        const vec3  D = normalize(T * Le.x + B * Le.y + n * Le.z);
        vec3 q = Pw / Cell; if (bFixed) { q[u] = floor(q[u] - 0.5) + j1 + 0.5; q[v] = floor(q[v] - 0.5) + j2 + 0.5; } else { q[u] += j1 - 0.5; q[v] += j2 - 0.5; }
        const float sd = abs(D[u]) + abs(D[v]) + max(D[a] * sg, 0.0) + 1e-6, x = D[u] / sd, y = D[v] / sd;
        const uint  di = min(uint((x + y + 1.0) * 0.5 * float(k)), k - 1u), dj = min(uint((x - y + 1.0) * 0.5 * float(k)), k - 1u);
        const uint  h = VoxHash(L, ivec3(floor(q)), 1000u + Face * 256u + di * k + dj), e = VoxEntryOf(h);
        const bool  bOurs = uintBitsToFloat(GIClip[e + 2u].x) == VoxTagOf(h);
        const vec4  w = bOurs ? VoxBin(GIClip[e]) : vec4(0.0);
        const float Tr = clamp(w.w / 64.0, 0.0, 1.0);        // (under a few rays: what the bins say, explicitly)
        Sum += Wt * mix(Fallback.rgb, w.w > 0.0 ? w.rgb / w.w : vec3(0.0), Tr); SumW += Wt;
    }
    return vec4(Sum / max(SumW, 1e-4), 1.0);
}
vec4 ClipRead(in const vec3 Pcam, in const vec3 n)
{
    const vec3 EyeL = vec3(TF(22u), TF(23u), TF(24u)), Eye = vec3(TF(25u), TF(26u), TF(27u));
    const vec3 Pw   = Pcam + Eye;
    for (int L = 0; L < clip_levels_v; ++L)
    {
        const vec4 R = ClipLevel(L, Pw, n, EyeL, TF(21u));
        if (R.w > 0.5) return vec4(R.xyz, 1.0);
    }
    return vec4(0.0);
}

const uint screen_header_v  = 8u;
const uint screen_max_age_v = 3u;

vec3 ScreenNormal(in const uint w)
{
    const vec2 o = vec2(float((w >> 16) & 63u), float((w >> 22) & 63u)) / 63.0 * 2.0 - 1.0;
    vec3 n = vec3(o, 1.0 - abs(o.x) - abs(o.y));
    if (n.z < 0.0) n.xy = (1.0 - abs(n.yx)) * vec2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0);
    return normalize(n);
}

void main()
{
    const ivec2 P = ivec2(gl_FragCoord.xy);
    const gather_sample S = GatherSample(P);

    // Mesh space: world = A local + t + the instance buffer's origin (the rows of A | t)
    const InstanceData I  = instances.Instances[S.bHit ? S.Slot : 0u];
    const mat3  At = mat3(I.Rows[0].xyz, I.Rows[1].xyz, I.Rows[2].xyz);
    const mat3  Ai = inverse(transpose(At));
    const vec3  Off = vec3(uintBitsToFloat(GITable[push.TableBase + 16u]), uintBitsToFloat(GITable[push.TableBase + 17u]), uintBitsToFloat(GITable[push.TableBase + 18u]));
    const vec3  Pm = Ai * (S.P - Off - vec3(I.Rows[0].w, I.Rows[1].w, I.Rows[2].w));
    // The read's face normal from the derivatives of a position at the same texel of every block (the checkerboard's alternate texels
    // skewed them: a wrong face on curved surfaces, sunlit surfels behind read - Tomas's Sponza arch), before any lane leaves
    vec3 Pd = Pm;
    {
        const ivec2 Size = textureSize(GPreDepth, 0);
        const ivec2 Qd   = min(P * int(push.Div) + ivec2(int(push.Div) / 2), Size - 1);
        const float zd   = texelFetch(GPreDepth, Qd, 0).r;
        const mat4  C2W  = mat4(uintBitsToFloat(uvec4(GITable[push.TableBase + 0u], GITable[push.TableBase + 1u], GITable[push.TableBase + 2u], GITable[push.TableBase + 3u])),
                                uintBitsToFloat(uvec4(GITable[push.TableBase + 4u], GITable[push.TableBase + 5u], GITable[push.TableBase + 6u], GITable[push.TableBase + 7u])),
                                uintBitsToFloat(uvec4(GITable[push.TableBase + 8u], GITable[push.TableBase + 9u], GITable[push.TableBase + 10u], GITable[push.TableBase + 11u])),
                                uintBitsToFloat(uvec4(GITable[push.TableBase + 12u], GITable[push.TableBase + 13u], GITable[push.TableBase + 14u], GITable[push.TableBase + 15u])));
        const vec4  Wd   = C2W * vec4((vec2(Qd) + 0.5) / vec2(Size) * 2.0 - 1.0, zd, 1.0);
        const float dd   = texelFetch(GPreId, Qd, 0).z;
        const vec3  Pw   = dd > 0.0 ? normalize(Wd.xyz / Wd.w) * dd : Wd.xyz / Wd.w;
        if (zd < 1.0) Pd = Ai * (Pw - Off - vec3(I.Rows[0].w, I.Rows[1].w, I.Rows[2].w));
    }
    const vec3  GFaceD = cross(dFdx(Pd), dFdy(Pd));
    if (!S.bHit) return;

    const uvec4 Hd = GIScreen[0];
    const uint  W = Hd.x, H = Hd.y;
    const bool  bIn = uint(P.x) < W && uint(P.y) < H;
    const vec3  Nr = S.N;                                   // (the vertex normal: the read's)
    const vec3  N  = S.bFront ? Nr : -Nr;                   // (its visible side: the guide's, the reflection's)
    const float Dist = length(S.P);
    vec2 o = N.xy / (abs(N.x) + abs(N.y) + abs(N.z));
    if (N.z < 0.0) o = (1.0 - abs(o.yx)) * vec2(o.x >= 0.0 ? 1.0 : -1.0, o.y >= 0.0 ? 1.0 : -1.0);
    const uvec2 Oq = uvec2(clamp(o * 0.5 + 0.5, 0.0, 1.0) * 63.0 + 0.5);
    const uint  Guide = (Oq.x << 16) | (Oq.y << 22);

    // Kind: 0 fresh by the pattern, 1 reprojected, 2 rejected (fresh), 3 too old (fresh)
    uint  Kind = 0u;
    uvec4 Out  = uvec4(0u);
    bool  bKeep = false;
    const uint Rate = 1u << ((Hd.w >> 1) & 3u);
    const uint Tile = uint(((P.x >> 3) & 1) | (((P.y >> 3) & 1) << 1));
    if ((Hd.w & 1u) != 0u && (Tile % Rate) != (Hd.z % Rate))
    {
        Kind = 2u;
        const mat4  PrevW2C = mat4(uintBitsToFloat(GIScreen[1]), uintBitsToFloat(GIScreen[2]), uintBitsToFloat(GIScreen[3]), uintBitsToFloat(GIScreen[4]));
        const vec3  Prev    = S.P + uintBitsToFloat(GIScreen[5].xyz);
        const vec4  C       = PrevW2C * vec4(Prev, 1.0);
        if (C.w > 1e-6)
        {
            const ivec2 Q = ivec2(floor((C.xy / C.w * 0.5 + 0.5) * vec2(float(W), float(H))));
            if (Q.x >= 0 && Q.y >= 0 && uint(Q.x) < W && uint(Q.y) < H)
            {
                const uvec4 R   = GIScreenPrev[screen_header_v + uint(Q.y) * W + uint(Q.x)];
                const float Sd  = unpackHalf2x16(R.y).y, Pd = length(Prev);
                const uint  Age = (R.w >> 28) & 3u;
                if ((R.w & (1u << 30)) != 0u && abs(Sd - Pd) < 0.02 * Pd + 0.01 && dot(ScreenNormal(R.w), N) > 0.9)
                {
                    if (Age >= screen_max_age_v) Kind = 3u;
                    else
                    {
                        Kind = 1u;
                        Out  = uvec4(R.x, packHalf2x16(vec2(unpackHalf2x16(R.y).x, Dist)), R.z, (R.w & 0xC000FFFFu) | Guide | ((Age + 1u) << 28));
                    }
                }
            }
        }
    }
    if (Kind != 1u)
    {
        // mb_standard_pbr.frag GIPixelGI, the per-pixel read (inNormal (0, 0, 1): the surface's own normal). ponytail: a copy of its
        // policy until the material compiler's step (stage 2c) moves it into xlionrender_surfel_gi.glsl for both
        const uint Offset = S.Entry.x, Ref = S.Entry.y;
        const vec3 GN0    = normalize(At * Nr);
        vec3       GFace  = dot(GFaceD, GFaceD) > 1e-24 ? normalize(dot(GFaceD, GN0) < 0.0 ? -GFaceD : GFaceD) : GN0;
        if (dot(GFace, GN0) < 0.7) GFace = GN0;
        vec4 E = vec4(0.0);
        bool bWeak = true;            // (no surfel supports it: the local mean - flagged, the upsample prefers its supported neighbours)
        g_GIRefl = vec4(0.0);
        if (GITable[push.TableBase + 20u] == 2u)                   // the ground truth: this pixel's point for the tracer, its mean so far (no reuse, no reflection)
        {
            const uint Rw = GITable[push.TableBase + 28u], Ri = uint(P.y) * Rw + uint(P.x);
            const vec3 Pwr = S.P + vec3(TF(25u), TF(26u), TF(27u));
            GIRefSample[2u * Ri] = vec4(Pwr, 1.0);
            GIRefSample[2u * Ri + 1u] = vec4(N, 0.0);
            GIRefSample[2u * W * H] = vec4(TF(25u), TF(26u), TF(27u), TF(30u));      // (the eye, the roughness the reflections' ground truth grades: xlionrender_gi_ref.glsl)
            const vec4 Ac = GIRefAccum[2u * Ri], Pa = GIRefAccum[2u * Ri + 1u];
            if (Ac.w > 0.5 && Pa.w > 0.5 && dot(Pa.xyz - Pwr, Pa.xyz - Pwr) < 1e-4) { E = vec4(Ac.rgb / Ac.w, 1.0); bWeak = false; }
            else E = vec4(0.0, 0.0, 0.0, 1.0);
        }
        else if (GITable[push.TableBase + 20u] == 3u)                   // experiment X1c: voxels of light on the surfaces, this pixel a place rays start from
        {
            const uint Rw = GITable[push.TableBase + 28u], Ri = uint(P.y) * Rw + uint(P.x);
            const vec3 Eyw = vec3(TF(25u), TF(26u), TF(27u));
            const float Ah = push.AO != 0u ? GatherAO(P, W, H, Dist) : 1.0;
            GIRefSample[2u * Ri] = vec4(S.P + Eyw, 1.0);
            GIRefSample[2u * Ri + 1u] = vec4(N, Ah);      // (w: this reader's AO - XLION_VOX_AO_BUCKET sorts its rays' samples by it)
            GIRefSample[2u * W * H] = vec4(TF(22u), TF(23u), TF(24u), TF(21u));
            g_ClipFloat = true;
            // (XLION_VOX_AO_BUCKET=<AO>, P1 #1 prototype: readers in a contact - their AO under it - read the records only such readers' rays fed, not the open floor's beside them; none yet: the shared ones)
            const float BucketAO = uintBitsToFloat(GITable[push.TableBase + 34u]);
            E = VoxReadLevels(S.P, N);
            const float Wc = BucketAO > 0.0 ? smoothstep(BucketAO + 0.05, BucketAO - 0.1, Ah) : 0.0;      // (a soft hand-over by the AO: a hard one speckled - the AO a pixel at a time)
            if (Wc > 0.0) { g_VoxBucket = 1u; const float Tr = g_ClipTrust; const vec4 Ec = VoxReadLevels(S.P, N); g_VoxBucket = 0u; g_ClipTrust = Tr; if (Ec.w > 0.0) E.rgb = mix(E.rgb, Ec.rgb, Wc); }
            bWeak = E.w < 0.5;
            E.w = 1.0;
            if (GITable[push.TableBase + 29u] == 0u) { g_GIRefl = VoxReflect(S.P, N, E.rgb); if ((GITable[push.TableBase + 32u] & 8u) != 0u) g_GIRefl = VoxDirsRead(S.P, N, g_GIRefl, P); }
        }
        else if (GITable[push.TableBase + 20u] != 0u)                   // experiment X1: the clipmap, both reads
        {
            E = ClipRead(S.P, N);
            bWeak = E.w < 0.5;
            const vec4 Rc = ClipRead(S.P, reflect(normalize(S.P), N));
            g_GIRefl = Rc;
            if (E.w < 0.5) E = vec4(SurfelGILocalMean(Pm, GFace, Offset, Ref & 0xFFu), 1.0);
        }
        else if (Ref == gi_probes_v) E = SurfelGIProbes(S.P, N);        // (a skinned surface: the probe grid, GIMesh its buffer)
        else if ((Ref & gi_per_pixel_v) != 0u)
        {
            g_GIMoments      = (Ref & gi_no_moments_v) == 0u;
            g_GIPixelCap     = GIPixelCap(Ref);
            g_GINoCorners    = (Ref & 0x01000000u) != 0u;
            g_GIAblateLookup = (Ref >> 25u) & 3u;
            g_GIReflDir      = normalize(Ai * reflect(normalize(S.P), N));        // (the eye at the origin: V = -P)
            const GIQueryResult Q = SurfelGIQuery(Pm, GFace, GN0, Offset, Ref & 0xFFu);
            // (the full read's policy: a low confidence - the surfels near mostly out of sight - blends into a smooth read; there the vertex
            // read, here the local mean: unblended, a weak read's noise and leaks showed as speckled patches - Tomas's Sponza arch)
            bWeak = !Q.bSupported;
            if (Q.bSupported) E = vec4(mix(SurfelGILocalMean(Pm, GFace, Offset, Ref & 0xFFu), Q.Irradiance, smoothstep(0.0, 0.25, Q.Confidence)), 1.0);
            if (!Q.bSupported || Q.Confidence < 0.25) g_GIRefl = vec4(0.0);
        }
        if (E.w < 0.5 && (Ref & 0x8000u) == 0u) E = vec4(SurfelGILocalMean(Pm, GFace, Offset, Ref & 0xFFu), 1.0);
        if (push.AO != 0u) E.rgb *= GatherAO(P, W, H, Dist);
        if (push.AO == 2u && E.w > 0.5) E.rgb += GatherSSGI(P, W, H);      // (the store's light kept for the free sectors, the screen's for the occluded ones)
        if (any(isnan(E.rgb)) || any(isinf(E.rgb))) E = vec4(0.0);     // (never a non-finite entry: the reprojection would carry it on)      // (a fresh read only: a reprojected one carries last frame's)
        if (GITable[push.TableBase + 29u] != 0u) g_GIRefl = vec4(0.0);
        const bool bE = E.w > 0.5, bR = g_GIRefl.w > 0.5;
        bKeep = !bE && !bR;           // (no GI here: the AO's own entry stays)
        // (never a non-finite or negative reflection, as E above: reprojection carries it on, black and blocky on the moto, Tomas 2026-11-04)
        vec3 Rr = bR && !any(isnan(g_GIRefl.rgb)) && !any(isinf(g_GIRefl.rgb)) ? max(g_GIRefl.rgb, vec3(0.0)) : vec3(0.0);
        // (the read along R over time, every pixel every frame: no tiles, no squares. Its bins' means move a little with each ray: on a still moto, 7 in 8 of the flicker went with the read)
        if ((Hd.w & 1u) != 0u && bR)
        {
            const mat4 PrevW2C = mat4(uintBitsToFloat(GIScreen[1]), uintBitsToFloat(GIScreen[2]), uintBitsToFloat(GIScreen[3]), uintBitsToFloat(GIScreen[4]));
            const vec3 Prev    = S.P + uintBitsToFloat(GIScreen[5].xyz);
            const vec4 C       = PrevW2C * vec4(Prev, 1.0);
            if (C.w > 1e-6)
            {
                const ivec2 Q = ivec2(floor((C.xy / C.w * 0.5 + 0.5) * vec2(float(W), float(H))));
                if (Q.x >= 0 && Q.y >= 0 && uint(Q.x) < W && uint(Q.y) < H)
                {
                    const uvec4 Rp = GIScreenPrev[screen_header_v + uint(Q.y) * W + uint(Q.x)];
                    const float Sd = unpackHalf2x16(Rp.y).y, Ld = length(Prev);
                    if ((Rp.w & (1u << 31)) != 0u && abs(Sd - Ld) < 0.02 * Ld + 0.01 && dot(ScreenNormal(Rp.w), N) > 0.9)
                        Rr = mix(vec3(unpackHalf2x16(Rp.z), unpackHalf2x16(Rp.w & 0xFFFFu).x), Rr, 0.1);
                }
            }
        }
        Out = uvec4(packHalf2x16(E.rg), packHalf2x16(vec2(E.b, Dist)), packHalf2x16(Rr.rg),
                    (packHalf2x16(vec2(Rr.b, 0.0)) & 0xFFFFu) | Guide | (bE ? 1u << 30 : 0u) | (bR ? 1u << 31 : 0u) | (bE && bWeak ? 3u << 28 : 0u));     // (age 3, never a fresh one's: weak - a hole in the surfels' coverage, the Cornell cube's dashes)
    }
    if (bIn && !(bKeep && push.AO != 0u)) GIScreen[screen_header_v + uint(P.y) * W + uint(P.x)] = Out;

    const uvec4 Mine = bIn ? uvec4(Kind == 0u ? 1u : 0u, Kind == 1u ? 1u : 0u, Kind == 2u ? 1u : 0u, Kind == 3u ? 1u : 0u) : uvec4(0u);
    const uvec4 Sum  = subgroupAdd(Mine);
    if (subgroupElect())
    {
        if (Sum.x != 0u) atomicAdd(GIScreen[6].x, Sum.x);
        if (Sum.y != 0u) atomicAdd(GIScreen[6].y, Sum.y);
        if (Sum.z != 0u) atomicAdd(GIScreen[6].z, Sum.z);
        if (Sum.w != 0u) atomicAdd(GIScreen[6].w, Sum.w);
    }
}
