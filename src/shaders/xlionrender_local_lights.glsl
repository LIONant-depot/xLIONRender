// Point and spot lights (xLIONRender, HDR build only): the scene's local lights in UBOLight.Local[], each shaded with the same
// Cook-Torrance + EON model as the main light and shadowed from its tiles of the shadow atlas (features/shadows/xlionrender_shadow_local.h
// builds them with the same face basis and projection as LocalShadow below). Included by mb_standard_pbr*.frag after the BRDF functions.
// ponytail: a plain loop over at most 32 lights; a clustered (froxel) light list when scenes have many.

// One light's direct contribution for a unit radiance from direction L (towards the light)
vec3 PBRDirect(in const vec3 N, in const vec3 V, in const float NdotV, in const vec3 Albedo, in const vec3 F0, in const float PerceptualRoughness, in const float Metallic, in const vec3 L)
{
    const float NdotL = saturate(dot(N, L));
    if (NdotL < 1e-4) return vec3(0.0);
    const vec3  H     = normalize(V + L);
    const float alpha = PerceptualRoughness * PerceptualRoughness;
    const vec3  F     = FresnelSchlick(saturate(dot(L, H)), F0);
    const vec3  Spec  = DistributionGGX(saturate(dot(N, H)), alpha) * F * VisibilitySmithCorrelated(NdotV, NdotL, alpha);
    const vec3  kD    = (vec3(1.0) - F) * (1.0 - Metallic);
    const vec3  Diff  = kD * DiffuseEON(Albedo, PerceptualRoughness, NdotL, NdotV, dot(L, V) - NdotL * NdotV);
    return (Diff + Spec) * NdotL;
}

#include "xlionrender_octahedral.glsl"
#include "xlionrender_light_eval.glsl"

layout(std430, set = 1, binding = 3) readonly buffer LocalLightBuffer { LocalLightData LocalLightList[]; };
layout(std430, set = 1, binding = 4) readonly buffer LightCellBuffer  { uint LightCells[]; };

// The local lights' shadow pages (features/shadows/xlionrender_shadow_local_pages.h): a face a virtual map of kLocalLevel0Pages^2
// pages in kLocalLevels levels, its entries in the page table (LightCells at UBOLight.ShadowParams.y: 16 bits, slot + 1); the slots
// in the atlas's local part (x from ShadowParams.x, ShadowParams.z slots a row). Set by the material before LocalLights: the pixel's
// world footprint and the triangle's own normal (the derivatives, in uniform control flow)
const int kLocalLevels      = 5;
const int kLocalLevel0Pages = 16;
const int kLocalPageTexels  = 128;
const int kLocalFaceEntries = 341;
float g_LocalFoot  = 0.0;
int   g_LocalTaps  = 1;                 // 1 the filter (a 3 x 3 texel tent), 0 one tap (the resampling's estimate)
vec3  g_LocalFaceN = vec3(0.0, 1.0, 0.0);

// 1 lit .. 0 shadowed. Receiver: the surface minus the light's position. The face it is seen through (a point light's cube face, a
// spot's frustum), the level whose texel fits the pixel's footprint (the request pass picks it alike: xlionrender_local_request_comp.glsl),
// coarser while a tap's page is missing (the coarsest is pinned). Against acne the receiver moves along its triangle's normal by half
// a texel, with the pages' slope-scaled depth bias (measured on the visibility reference: 1.5 texels and a texel nearer missed contact
// shadows, 3.8% against 2.5% disagreement, no less acne). A 3 x 3 texel tent (4 x 4 taps, bilinear: the sun's filter), each tap through the table (a tap past the face's edge clamps to it)
float LocalShadow(in const LocalLightData Ld, in const vec3 Receiver, in const vec3 N)
{
    if (Ld.Shadow.z <= 0.0 || UBOLight.ShadowParams.x <= 0.0) return 1.0;

    const bool  bPoint  = Ld.DirCosOuter.w < -1.5;
    const float TanHalf = bPoint ? 1.0 : Ld.Shadow.w;
    const float Radius  = Ld.PosRadius.w;
    const float Near    = max(Radius * 0.01, 0.005);                           // shadows::LocalNear
    vec3 Axis;
    int  Face = 0;
    if (bPoint)
    {
        const vec3 A = abs(Receiver);
        if (A.x >= A.y && A.x >= A.z) { Face = Receiver.x >= 0.0 ? 0 : 1; Axis = vec3(Receiver.x >= 0.0 ? 1.0 : -1.0, 0.0, 0.0); }
        else if (A.y >= A.z)          { Face = Receiver.y >= 0.0 ? 2 : 3; Axis = vec3(0.0, Receiver.y >= 0.0 ? 1.0 : -1.0, 0.0); }
        else                          { Face = Receiver.z >= 0.0 ? 4 : 5; Axis = vec3(0.0, 0.0, Receiver.z >= 0.0 ? 1.0 : -1.0); }
    }
    else Axis = Ld.DirCosOuter.xyz;
    const vec3 Up = abs(Axis.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    const vec3 R  = normalize(cross(Up, Axis));
    const vec3 U  = cross(Axis, R);
    const float d0 = dot(Receiver, Axis);
    if (d0 <= Near) return 1.0;
    const float Texel0 = 2.0 * d0 * TanHalf / float(kLocalLevel0Pages * kLocalPageTexels);
    const int   Level  = clamp(int(floor(log2(max(g_LocalFoot, 1e-9) / Texel0) + UBOLight.ShadowParams.w)), 0, kLocalLevels - 1);
    const uint  FaceBase = uint(Ld.Shadow.x) + uint(Face * kLocalFaceEntries);
    const uint  TableBase = uint(UBOLight.ShadowParams.y);
    const int   Row = int(UBOLight.ShadowParams.z);
    const float A = Radius / (Radius - Near), B = -Radius * Near / (Radius - Near);     // shadows::FaceW2C's depth

    for (int l = Level; l < kLocalLevels; ++l)
    {
        const float TexelW = Texel0 * float(1 << l);
        const vec3  P  = Receiver + g_LocalFaceN * (0.5 * TexelW);
        const float d  = dot(P, Axis);
        if (d <= Near) return 1.0;
        const vec2  UV = vec2(dot(P, R), dot(P, U)) / (d * TanHalf);
        if (!bPoint && any(greaterThan(abs(UV), vec2(1.0)))) return 1.0;      // outside the spot's frustum
        const float Ref = A + B / d;
        const int   NT  = (kLocalLevel0Pages >> l) * kLocalPageTexels, W = kLocalLevel0Pages >> l;
        uint LB = 0u;
        for (int k = 0; k < l; ++k) LB += uint((kLocalLevel0Pages >> k) * (kLocalLevel0Pages >> k));
        // the 3 x 3 texel tent, bilinear (4 x 4 taps weighted by the point's place in its texel, as the sun's); one tap: the nearest texel
        const vec2  U  = (clamp(UV, vec2(-1.0), vec2(1.0)) * 0.5 + 0.5) * float(NT) - 0.5;
        const ivec2 T  = g_LocalTaps == 0 ? ivec2(floor(U + 0.5)) : ivec2(floor(U)) - 1;
        const vec2  f  = U - floor(U);
        const vec4  Wx = vec4(1.0 - f.x, 1.0, 1.0, f.x), Wy = vec4(1.0 - f.y, 1.0, 1.0, f.y);
        const int   nT = g_LocalTaps == 0 ? 1 : 4;
        float Lit = 0.0;
        bool  bAll = true;
        for (int y = 0; y < nT && bAll; ++y)
        for (int x = 0; x < nT; ++x)
        {
            const ivec2 t = clamp(T + ivec2(x, y), ivec2(0), ivec2(NT - 1));
            const ivec2 Pg = t / kLocalPageTexels;
            const uint  i  = FaceBase + LB + uint(Pg.y * W + Pg.x);
            const uint  E  = (LightCells[TableBase + (i >> 1)] >> ((i & 1u) * 16u)) & 0xFFFFu;
            if (E == 0u) { bAll = false; break; }
            const int   s  = int(E) - 1;
            const ivec2 At = ivec2(int(UBOLight.ShadowParams.x) + (s % Row) * kLocalPageTexels, (s / Row) * kLocalPageTexels) + (t - Pg * kLocalPageTexels);
            Lit += ((texelFetch(SamplerShadowMap, At, 0).r < Ref) ? 0.0 : 1.0) * (nT == 1 ? 1.0 : Wx[x] * Wy[y]);
        }
        if (bAll) return nT == 1 ? Lit : Lit / 9.0;
    }
    return 1.0;                                                                 // (no page yet: the light was never rendered)
}

// The local lights (storage buffers, so a scene is not limited to a uniform block): the lights, and the world-cell lists
// (features/lights/xlionrender_light_cells.h builds them; levels_v 5, cells_v 16, a packed uint header per cell, 2 uint entries)

const int kCellLevels = 5;
const int kCells      = 16;
const int kLevelCells = kCells * kCells * kCells;

// One light's contribution at the surface without its shadow (0 outside its radius)
vec3 LocalLightUnshadowed(in const int i, in const vec3 N, in const vec3 V, in const float NdotV, in const vec3 Albedo, in const vec3 F0, in const float PerceptualRoughness, in const float Metallic)
{
    const LocalLightData Ld = LocalLightList[int(UBOLight.LocalBase.x) + i];
    const vec3  ToLight = Ld.PosRadius.xyz - In.wSpacePosition.xyz;            // both camera-centered
    const float Dist    = length(ToLight);
    const float s       = Dist / Ld.PosRadius.w;
    if (s >= 1.0 || Dist < 1e-6) return vec3(0.0);
    float Att = LightDistanceFalloff(s, 1.0);
    const vec3 L = ToLight / Dist;
    if (Ld.DirCosOuter.w > -1.5) Att *= LightSpotCone(dot(-L, Ld.DirCosOuter.xyz), Ld.DirCosOuter.w, Ld.Color.w);
    if (Att < 1e-4) return vec3(0.0);
    return PBRDirect(N, V, NdotV, Albedo, F0, PerceptualRoughness, Metallic, L) * Ld.Color.rgb * Att;
}

// Its shadow at the surface (1 lit .. 0 shadowed)
float LocalLightShadow(in const int i, in const vec3 N)
{
    const LocalLightData Ld = LocalLightList[int(UBOLight.LocalBase.x) + i];
    return LocalShadow(Ld, In.wSpacePosition.xyz - Ld.PosRadius.xyz, N);
}

// One light's contribution at the surface (0 outside its radius)
vec3 LocalLight(in const int i, in const vec3 N, in const vec3 V, in const float NdotV, in const vec3 Albedo, in const vec3 F0, in const float PerceptualRoughness, in const float Metallic)
{
    const vec3 Direct = LocalLightUnshadowed(i, N, V, NdotV, Albedo, F0, PerceptualRoughness, Metallic);
    if (all(equal(Direct, vec3(0.0)))) return vec3(0.0);                       // (facing away: no shadow lookups)
    return Direct * LocalLightShadow(i, N);
}

// The cell holding a world position: the finest level whose window holds it (-1: none)
int LocalLightCell(in const vec3 World)
{
    for (int Level = 0; Level < kCellLevels; ++Level)
    {
        const float S = UBOLight.LocalBase.z * float(1 << Level);
        const ivec3 C = ivec3(floor(World / S));
        const ivec3 R = C - ivec3(UBOLight.CellOrigin[Level].xyz);
        if (any(lessThan(R, ivec3(0))) || any(greaterThanEqual(R, ivec3(kCells)))) continue;
        const ivec3 M = C & ivec3(kCells - 1);                                  // index mod 16 (two's complement)
        return Level * kLevelCells + (M.z * kCells + M.y) * kCells + M.x;
    }
    return -1;
}

// [0, 1) per pixel and per seed (Stream: independent numbers for the same pixel and seed)
float LocalRandom(in const uint Stream)
{
    uint h = uint(gl_FragCoord.x) * 1973u + uint(gl_FragCoord.y) * 9277u + uint(UBOLight.LocalCount.z) * 26699u + Stream * 0x9E3779B9u;
    h = (h ^ 61u) ^ (h >> 16); h *= 9u; h ^= h >> 4; h *= 0x27d4eb2du; h ^= h >> 15;
    return float(h >> 8) * (1.0 / 16777216.0);
}
float LocalRandom() { return LocalRandom(0u); }

// Modes: 0 every light; 1 the cell's list (the same result: the list holds every light that reaches the cell); 2 the cell's
// head exactly + tail lights J drawn with probability p(J) (LocalBase.w stratified samples), each weighted 1 / (p(J) samples). Outside the cells: every light.
vec3 LocalLights(in const vec3 N, in const vec3 V, in const float NdotV, in const vec3 Albedo, in const vec3 F0, in const float PerceptualRoughness, in const float Metallic)
{
    vec3 Sum = vec3(0.0);
    const int Mode = int(UBOLight.LocalCount.y);
    const int Cell = Mode == 0 ? -1 : LocalLightCell(In.wSpacePosition.xyz + UBOLight.CameraPos.xyz);
    if (Cell < 0)
    {
        const int Count = int(UBOLight.LocalCount.x);
        for (int i = 0; i < Count; ++i) Sum += LocalLight(i, N, V, NdotV, Albedo, F0, PerceptualRoughness, Metallic);
        return Sum;
    }

    const uint Base      = uint(UBOLight.LocalBase.y);
    const uint Header    = LightCells[Base + uint(Cell)];               // bits 0-19 first entry, 20-27 count, 28-31 head count
    const uint EntryBase = Base + uint(kCellLevels * kLevelCells);
    const uint First     = Header & 0xFFFFFu, Count = (Header >> 20) & 0xFFu, Head = Header >> 28;
    if (Count == 255u)                                                  // the cell's list did not fit: every light (exact)
    {
        const int nLights = int(UBOLight.LocalCount.x);
        for (int i = 0; i < nLights; ++i) Sum += LocalLight(i, N, V, NdotV, Albedo, F0, PerceptualRoughness, Metallic);
        return Sum;
    }
    const uint nExact    = Mode == 1 ? Count : Head;
    for (uint k = 0u; k < nExact; ++k) Sum += LocalLight(int(LightCells[EntryBase + 2u * (First + k)]), N, V, NdotV, Albedo, F0, PerceptualRoughness, Metallic);

    if (Mode == 2 && Count > Head && abs(UBOLight.LocalCount.w) > 1.5)
    {
        // Resampled (RIS, P3's first step): M stratified candidates from the cell's tail distribution q, each weighted by the light's
        // unshadowed contribution at this pixel p (its luminance) over q, one of them picked in proportion to its weight and shaded with its
        // shadow: f(Y) / p(Y) * sum(w) / M. Unbiased (each candidate is distributed by q; p > 0 wherever the light reaches, f <= the
        // unshadowed term): the cell's diffuse importance becomes the pixel's own - a glossy highlight's light is found. With the visibility
        // (LocalCount.w < 0, the default) a candidate's weight also holds its shadow, one tap, at least 0.1 (p stays > 0 wherever the filtered
        // shadow can be: unbiased; the picked light is shaded with its filtered shadow): a light behind a wall is rarely drawn (MegaLights
        // weighs its samples by their traced visibility alike). Lights behind a wall: per-frame error 0.41 -> 0.26, for 0.036 -> 0.048 ms
        const int   M  = int(abs(UBOLight.LocalCount.w));
        const bool  bVis = UBOLight.LocalCount.w < 0.0;
        const float u0 = LocalRandom(0u);
        float       u1 = LocalRandom(1u);
        int   k    = 0, Pick = -1;
        float Prev = 0.0, WSum = 0.0, PickP = 1.0;
        vec3  PickC = vec3(0.0);
        for (uint e = Head; e < Count && k < M; ++e)
        {
            const float Cdf = uintBitsToFloat(LightCells[EntryBase + 2u * (First + e) + 1u]);
            int nHits = 0;
            while (k < M && ((float(k) + u0) / float(M) < Cdf || e + 1u == Count)) { ++nHits; ++k; }
            if (nHits > 0)
            {
                const int   l = int(LightCells[EntryBase + 2u * (First + e)]);
                const vec3  C = LocalLightUnshadowed(l, N, V, NdotV, Albedo, F0, PerceptualRoughness, Metallic);
                float S = 1.0;
                if (bVis && any(greaterThan(C, vec3(0.0)))) { g_LocalTaps = 0; S = max(LocalLightShadow(l, N), 0.1); g_LocalTaps = 1; }
                const float P = dot(C, vec3(0.2126, 0.7152, 0.0722)) * S;
                const float W = float(nHits) * P / (Cdf - Prev);                  // (its nHits candidates at once)
                WSum += W;
                if (W > 0.0)
                {
                    const float a = W / WSum;                                       // (a streaming pick: u1 stays uniform)
                    if (u1 < a) { Pick = l; PickC = C; PickP = P; u1 = u1 / a; }
                    else        u1 = (u1 - a) / (1.0 - a);
                }
            }
            Prev = Cdf;
        }
        if (Pick >= 0) Sum += PickC * LocalLightShadow(Pick, N) * (WSum / (float(M) * PickP));
    }
    else if (Mode == 2 && Count > Head)
    {
        // Samples stratified tail samples (u_k = (k + u) / Samples): one pass, the cumulative probabilities only grow
        const int   Samples = max(int(UBOLight.LocalBase.w), 1);
        const float u0      = LocalRandom();
        int   k    = 0;
        float Prev = 0.0;
        for (uint e = Head; e < Count && k < Samples; ++e)
        {
            const float Cdf = uintBitsToFloat(LightCells[EntryBase + 2u * (First + e) + 1u]);
            int nHits = 0;
            while (k < Samples && ((float(k) + u0) / float(Samples) < Cdf || e + 1u == Count)) { ++nHits; ++k; }
            if (nHits > 0) Sum += LocalLight(int(LightCells[EntryBase + 2u * (First + e)]), N, V, NdotV, Albedo, F0, PerceptualRoughness, Metallic) * (float(nHits) / (float(Samples) * (Cdf - Prev)));
            Prev = Cdf;
        }
    }
    return Sum;
}
