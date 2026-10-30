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

// The same direct term demodulated (the screen tail: xlionrender_tail_screen_frag.glsl): the diffuse with albedo 1 and no Fresnel, the
// specular with F = 1 - the material multiplies its own terms back (LocalTailScreen: nothing is divided)
void PBRDirectDemod(in const vec3 N, in const vec3 V, in const float NdotV, in const float PerceptualRoughness, in const vec3 L, out vec3 Diff, out vec3 Spec)
{
    Diff = vec3(0.0); Spec = vec3(0.0);
    const float NdotL = saturate(dot(N, L));
    if (NdotL < 1e-4) return;
    const vec3  H     = normalize(V + L);
    const float alpha = PerceptualRoughness * PerceptualRoughness;
    Spec = vec3(DistributionGGX(saturate(dot(N, H)), alpha) * VisibilitySmithCorrelated(NdotV, NdotL, alpha) * NdotL);
    Diff = DiffuseEON(vec3(1.0), PerceptualRoughness, NdotL, NdotV, dot(L, V) - NdotL * NdotV) * NdotL;
}

#include "xlionrender_octahedral.glsl"
#include "xlionrender_light_eval.glsl"

layout(std430, set = 1, binding = 3) readonly buffer LocalLightBuffer { LocalLightData LocalLightList[]; };
layout(std430, set = 1, binding = 4) readonly buffer LightCellBuffer  { uint LightCells[]; };
#ifndef XLION_TAIL_SCREEN
// The denoised screen tail (instance: the tail pass and the denoiser, xlionrender_tail_denoise_comp.glsl): a record a half-size texel - x:
// the diffuse's rg, y: its b and the specular's r, z: the specular's gb (halves, demodulated), w: the surface's view depth (clip w; half,
// bits 0-15; 0: no surface), its octahedral normal (4 + 4 bits from 16), its frames - from record UBOLight.TailParams.w
layout(std430, set = 1, binding = 14) readonly buffer TailScreenBuffer { uvec4 TailScreen[]; };
#endif

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

// A shadowed light's face for a receiver (LocalShadow): its basis, projection and table entries
struct local_face
{
    vec3  Axis, R, U;
    float TanHalf, Near, A, B, Texel0;
    uint  FaceBase;
    bool  bPoint;
};
const int kLocalSoftLevels = 2;         // a light with a source radius: its blocker search and filter up to this many levels coarser

// A tap's depth through the table at level l (texel t of the level's NT^2): -1 when its page is not there
float LocalTexel(in const local_face F, in const int l, in const ivec2 t)
{
    const int   W  = kLocalLevel0Pages >> l;
    uint LB = 0u;
    for (int k = 0; k < l; ++k) LB += uint((kLocalLevel0Pages >> k) * (kLocalLevel0Pages >> k));
    const ivec2 Pg = t / kLocalPageTexels;
    const uint  i  = F.FaceBase + LB + uint(Pg.y * W + Pg.x);
    const uint  E  = (LightCells[uint(UBOLight.ShadowParams.y) + (i >> 1)] >> ((i & 1u) * 16u)) & 0xFFFFu;
    if (E == 0u) return -1.0;
    const int   s   = int(E) - 1, Row = int(UBOLight.ShadowParams.z);
    const ivec2 At  = ivec2(int(UBOLight.ShadowParams.x) + (s % Row) * kLocalPageTexels, (s / Row) * kLocalPageTexels) + (t - Pg * kLocalPageTexels);
    return texelFetch(SamplerShadowMap, At, 0).r;
}

// The receiver at level l: moved along its triangle's normal by half the level's texel; its face coordinates (UV, -1..1), depth along
// the axis and stored depth to compare with. False: nearer than the near plane or outside the spot's frustum (lit)
bool LocalAt(in const local_face F, in const vec3 Receiver, in const int l, out vec2 UV, out float d, out float Ref)
{
    const float TexelW = F.Texel0 * float(1 << l);
    const vec3  P  = Receiver + g_LocalFaceN * (0.5 * TexelW);
    d   = dot(P, F.Axis);
    UV  = vec2(0.0); Ref = 0.0;
    if (d <= F.Near) return false;
    UV  = vec2(dot(P, F.R), dot(P, F.U)) / (d * F.TanHalf);
    if (!F.bPoint && any(greaterThan(abs(UV), vec2(1.0)))) return false;      // outside the spot's frustum
    Ref = F.A + F.B / d;
    return true;
}

// Level l's filtered visibility: the 3 x 3 texel tent, bilinear (4 x 4 taps weighted by the point's place in its texel, as the sun's);
// one tap (g_LocalTaps 0): the nearest texel. -1: a tap's page is not there; -2: lit (LocalAt)
float LocalShadowAt(in const local_face F, in const vec3 Receiver, in const int l)
{
    vec2 UV; float d, Ref;
    if (!LocalAt(F, Receiver, l, UV, d, Ref)) return -2.0;
    const int   NT = (kLocalLevel0Pages >> l) * kLocalPageTexels;
    const vec2  U  = (clamp(UV, vec2(-1.0), vec2(1.0)) * 0.5 + 0.5) * float(NT) - 0.5;
    const ivec2 T  = g_LocalTaps == 0 ? ivec2(floor(U + 0.5)) : ivec2(floor(U)) - 1;
    const vec2  f  = U - floor(U);
    const vec4  Wx = vec4(1.0 - f.x, 1.0, 1.0, f.x), Wy = vec4(1.0 - f.y, 1.0, 1.0, f.y);
    const int   nT = g_LocalTaps == 0 ? 1 : 4;
    float Lit = 0.0;
    for (int y = 0; y < nT; ++y)
    for (int x = 0; x < nT; ++x)
    {
        const float z = LocalTexel(F, l, clamp(T + ivec2(x, y), ivec2(0), ivec2(NT - 1)));
        if (z < 0.0) return -1.0;
        Lit += (z < Ref ? 0.0 : 1.0) * (nT == 1 ? 1.0 : Wx[x] * Wy[y]);
    }
    return nT == 1 ? Lit : Lit / 9.0;
}

// Contact-hardening's blocker search at level l: the 4 x 4 texels around the point, the mean distance along the axis of those nearer the
// light than the receiver, and the receiver's. Returns that blocker distance; -1 no blocker, -2 a page is not there or the point is lit
float LocalBlocker(in const local_face F, in const vec3 Receiver, in const int l, out float dR)
{
    vec2 UV; float Ref;
    dR = 0.0;
    if (!LocalAt(F, Receiver, l, UV, dR, Ref)) return -2.0;
    const int   NT = (kLocalLevel0Pages >> l) * kLocalPageTexels;
    const ivec2 T  = ivec2(floor((clamp(UV, vec2(-1.0), vec2(1.0)) * 0.5 + 0.5) * float(NT) - 0.5)) - 1;
    float Sum = 0.0, n = 0.0;
    for (int y = 0; y < 4; ++y)
    for (int x = 0; x < 4; ++x)
    {
        const float z = LocalTexel(F, l, clamp(T + ivec2(x, y), ivec2(0), ivec2(NT - 1)));
        if (z < 0.0) return -2.0;
        if (z < Ref) { Sum += F.B / (z - F.A); n += 1.0; }                          // (the stored depth A + B / d, back to d)
    }
    return n > 0.0 ? Sum / n : -1.0;
}

// 1 lit .. 0 shadowed. Receiver: the surface minus the light's position. The face it is seen through (a point light's cube face, a
// spot's frustum), the level whose texel fits the pixel's footprint (the request pass picks it alike: xlionrender_local_request_comp.glsl),
// coarser while a tap's page is missing (the coarsest is pinned). Against acne the receiver moves along its triangle's normal by half
// a texel, with the pages' slope-scaled depth bias (measured on the visibility reference: 1.5 texels and a texel nearer missed contact
// shadows, 3.8% against 2.5% disagreement, no less acne). A 3 x 3 texel tent (4 x 4 taps, bilinear: the sun's filter), each tap through the table (a tap past the face's edge clamps to it).
// A light with a source radius (Ld.Shadow.y, world units; 0: hard) hardens at the contact as the sun does: a blocker search (the coarsest of
// kLocalSoftLevels levels up, then finer), the penumbra's half-width at the receiver - radius x (receiver - blocker) / blocker, similar
// triangles - and the filter at the level whose taps span it (two levels blended)
float LocalShadow(in const LocalLightData Ld, in const vec3 Receiver, in const vec3 N)
{
    if (Ld.Shadow.z <= 0.0 || UBOLight.ShadowParams.x <= 0.0) return 1.0;

    local_face F;
    F.bPoint  = Ld.DirCosOuter.w < -1.5;
    F.TanHalf = F.bPoint ? 1.0 : Ld.Shadow.w;
    const float Radius  = Ld.PosRadius.w;
    F.Near    = max(Radius * 0.01, 0.005);                                     // shadows::LocalNear
    int Face = 0;
    if (F.bPoint)
    {
        const vec3 A = abs(Receiver);
        if (A.x >= A.y && A.x >= A.z) { Face = Receiver.x >= 0.0 ? 0 : 1; F.Axis = vec3(Receiver.x >= 0.0 ? 1.0 : -1.0, 0.0, 0.0); }
        else if (A.y >= A.z)          { Face = Receiver.y >= 0.0 ? 2 : 3; F.Axis = vec3(0.0, Receiver.y >= 0.0 ? 1.0 : -1.0, 0.0); }
        else                          { Face = Receiver.z >= 0.0 ? 4 : 5; F.Axis = vec3(0.0, 0.0, Receiver.z >= 0.0 ? 1.0 : -1.0); }
    }
    else F.Axis = Ld.DirCosOuter.xyz;
    const vec3 Up = abs(F.Axis.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    F.R = normalize(cross(Up, F.Axis));
    F.U = cross(F.Axis, F.R);
    const float d0 = dot(Receiver, F.Axis);
    if (d0 <= F.Near) return 1.0;
    F.Texel0   = 2.0 * d0 * F.TanHalf / float(kLocalLevel0Pages * kLocalPageTexels);
    const int Level = clamp(int(floor(log2(max(g_LocalFoot, 1e-9) / F.Texel0) + UBOLight.ShadowParams.w)), 0, kLocalLevels - 1);
    F.FaceBase = uint(Ld.Shadow.x) + uint(Face * kLocalFaceEntries);
    F.A = Radius / (Radius - F.Near); F.B = -Radius * F.Near / (Radius - F.Near);     // shadows::FaceW2C's depth

    float Lf = float(Level);
    if (Ld.Shadow.y > 0.0 && g_LocalTaps != 0)
    {
        const int Top = min(Level + kLocalSoftLevels, kLocalLevels - 1);
        for (int ls = Top; ls >= Level; ls -= 2)                                   // (the coarsest, then finer: a thin caster lost in a coarse level)
        {
            float dR;
            const float dB = LocalBlocker(F, Receiver, ls, dR);
            if (dB < 0.0) continue;
            const float Penumbra = Ld.Shadow.y * max(dR - dB, 0.0) / max(dB, 1e-6);  // world units, half the width
            Lf = clamp(float(Level) + log2(max(Penumbra, 1e-9) / (1.5 * F.Texel0 * float(1 << Level))), float(Level), float(Top));
            break;
        }
    }
    const int   L0 = int(floor(Lf));
    const float t  = Lf - float(L0);
    for (int l = L0; l < kLocalLevels; ++l)
    {
        float S = LocalShadowAt(F, Receiver, l);
        if (S == -2.0) return 1.0;
        if (S < 0.0) continue;
        if (t > 0.0 && l == L0 && l + 1 < kLocalLevels)
        {
            const float S1 = LocalShadowAt(F, Receiver, l + 1);
            if (S1 >= 0.0) S = mix(S, S1, t);
        }
        return S;
    }
    return 1.0;                                                                 // (no page yet: the light was never rendered)
}

// The local lights (storage buffers, so a scene is not limited to a uniform block): the lights, and the world-cell lists
// (features/lights/xlionrender_light_cells.h builds them; levels_v 5, cells_v 16, a packed uint header per cell, 2 uint entries)

const int kCellLevels = 5;
const int kCells      = 16;
const int kLevelCells = kCells * kCells * kCells;

// An emitter light (DirCosOuter.w -3: the GI's emitting surfel, instance::refreshEmitterLights): a one-sided disc of area A (Shadow.w)
// facing DirCosOuter.xyz, its radiance the color - irradiance radiance x A cos / (d^2 + A), the GI's own form factor (its link weight),
// windowed to 0 at the light's reach
float LocalEmitterAtt(in const LocalLightData Ld, in const vec3 L, in const float Dist, in const float s)
{
    const float A = Ld.Shadow.w;
    return A * max(dot(-L, Ld.DirCosOuter.xyz), 0.0) / (Dist * Dist + A) * (1.0 - s * s * s * s);
}
// What the cut-off (1e-4 of the color) weighs: an emitter's color is a radiance (its attenuation is the small one), the others' an intensity
float LocalAttScale(in const LocalLightData Ld) { return Ld.DirCosOuter.w < -2.5 ? dot(Ld.Color.rgb, vec3(0.2126, 0.7152, 0.0722)) : 1.0; }

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
    else if (Ld.DirCosOuter.w < -2.5) Att = LocalEmitterAtt(Ld, L, Dist, s);
    if (Att * LocalAttScale(Ld) < 1e-4) return vec3(0.0);
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

// The tail entry a sample u falls in: the first from Lo with a cumulative probability above u (the last at worst) - a binary search,
// O(log n) reads (the cell's whole list walked before: the tail cost as much as CELLS' loop)
float LocalTailCdf(in const uint EntryBase, in const uint First, in const uint e) { return uintBitsToFloat(LightCells[EntryBase + 2u * (First + e) + 1u]); }
uint LocalTailFind(in const uint EntryBase, in const uint First, in const uint Lo, in const uint Count, in const float u)
{
    uint a = Lo, b = Count - 1u;
    while (a < b)
    {
        const uint m = (a + b) >> 1;
        if (u < LocalTailCdf(EntryBase, First, m)) b = m; else a = m + 1u;
    }
    return a;
}

// One light's demodulated contribution without its shadow (PBRDirectDemod x its color and falloff; false: none)
bool LocalLightDemod(in const int i, in const vec3 N, in const vec3 V, in const float NdotV, in const float PerceptualRoughness, out vec3 Diff, out vec3 Spec)
{
    Diff = vec3(0.0); Spec = vec3(0.0);
    const LocalLightData Ld = LocalLightList[int(UBOLight.LocalBase.x) + i];
    const vec3  ToLight = Ld.PosRadius.xyz - In.wSpacePosition.xyz;
    const float Dist    = length(ToLight);
    const float s       = Dist / Ld.PosRadius.w;
    if (s >= 1.0 || Dist < 1e-6) return false;
    float Att = LightDistanceFalloff(s, 1.0);
    const vec3 L = ToLight / Dist;
    if (Ld.DirCosOuter.w > -1.5) Att *= LightSpotCone(dot(-L, Ld.DirCosOuter.xyz), Ld.DirCosOuter.w, Ld.Color.w);
    else if (Ld.DirCosOuter.w < -2.5) Att = LocalEmitterAtt(Ld, L, Dist, s);
    if (Att * LocalAttScale(Ld) < 1e-4) return false;
    PBRDirectDemod(N, V, NdotV, PerceptualRoughness, L, Diff, Spec);
    Diff *= Ld.Color.rgb * Att; Spec *= Ld.Color.rgb * Att;
    return any(greaterThan(Diff + Spec, vec3(0.0)));
}

#ifdef XLION_TAIL_SCREEN
// The tail pass's screen trace (xlionrender_tail_screen_frag.glsl): 1 the way to the light is clear in the half-size depth .. 0 blocked
float TailContactShadow(in const vec3 P, in const vec3 ToLight);
// P4: 1 the way to the light is clear through the GI's BVH (the opaque rigid GI meshes, off screen too) .. 0 blocked
float TailBvhShadow(in const vec3 P, in const vec3 ToLight);
#endif

// The screen tail's sample (xlionrender_tail_screen_frag.glsl): the cell's tail resampled as LocalLights' mode 2 does (M candidates, their
// weights with their one-tap visibility), the target the demodulated term's luminance; the picked light's demodulated diffuse and specular
// with its filtered shadow, weighted f / p. False: no tail at the point (no cell, or its list fits in the head)
bool LocalTailDemod(in const vec3 N, in const vec3 V, in const float NdotV, in const float PerceptualRoughness, out vec3 Diff, out vec3 Spec)
{
    Diff = vec3(0.0); Spec = vec3(0.0);
    if (UBOLight.LocalCount.x < 0.5) return false;
    const int Cell = LocalLightCell(In.wSpacePosition.xyz + UBOLight.CameraPos.xyz);
    if (Cell < 0) return false;
    const uint Base      = uint(UBOLight.LocalBase.y);
    const uint Header    = LightCells[Base + 2u * uint(Cell) + 1u];
    const uint EntryBase = Base + uint(2 * kCellLevels * kLevelCells);
    const uint First     = LightCells[Base + 2u * uint(Cell)], Count = Header & 0xFFFFFFu, Head = Header >> 24;
    if (Count <= Head) return false;
    const int   M  = max(int(abs(UBOLight.LocalCount.w)), 1);
    const bool  bVis = UBOLight.LocalCount.w < 0.0;
    const float u0 = LocalRandom(0u);
    float       u1 = LocalRandom(1u);
    int   k = 0, Pick = -1;
    float WSum = 0.0, PickP = 1.0;
    vec3  PickD = vec3(0.0), PickS = vec3(0.0);
    uint  e = Head;
    while (k < M)
    {
        e = LocalTailFind(EntryBase, First, e, Count, (float(k) + u0) / float(M));
        const float Cdf  = LocalTailCdf(EntryBase, First, e);
        const float Prev = e > Head ? LocalTailCdf(EntryBase, First, e - 1u) : 0.0;
        int nHits = 1;
        ++k;
        while (k < M && ((float(k) + u0) / float(M) < Cdf || e + 1u == Count)) { ++nHits; ++k; }
        const int l = int(LightCells[EntryBase + 2u * (First + e)]);
        vec3 D, Sp;
        if (LocalLightDemod(l, N, V, NdotV, PerceptualRoughness, D, Sp))
        {
            float Sh = 1.0;
            if (bVis) { g_LocalTaps = 0; Sh = max(LocalLightShadow(l, N), 0.1); g_LocalTaps = 1; }
            const float P = dot(D + Sp, vec3(0.2126, 0.7152, 0.0722)) * Sh;
            const float W = float(nHits) * P / (Cdf - Prev);
            WSum += W;
            if (W > 0.0)
            {
                const float a = W / WSum;
                if (u1 < a) { Pick = l; PickD = D; PickS = Sp; PickP = P; u1 = u1 / a; }
                else        u1 = (u1 - a) / (1.0 - a);
            }
        }
        ++e;
    }
    if (Pick < 0) return true;                                                  // (a tail, all of it dark here: 0)
    float Wt = LocalLightShadow(Pick, N) * (WSum / (float(M) * PickP));
#ifdef XLION_TAIL_SCREEN
    // a light without shadow pages: contact-shadowed by the screen trace (the picked light only - unbiased: its contribution x its visibility)
    const LocalLightData Lp = LocalLightList[int(UBOLight.LocalBase.x) + Pick];
    // and through the GI's BVH (P4: off screen, behind the first surface - what the screen does not see); what the BVH does not hold (skinned,
    // masked, meshes without GI) is the screen trace's alone
    if (Lp.Shadow.z <= 0.0 && UBOLight.TailTrace.x > 0.0) Wt *= TailContactShadow(In.wSpacePosition.xyz, Lp.PosRadius.xyz - In.wSpacePosition.xyz);
    if (Lp.Shadow.z <= 0.0 && UBOLight.TailBvh.x > 0.0 && Wt > 0.0) Wt *= TailBvhShadow(In.wSpacePosition.xyz, Lp.PosRadius.xyz - In.wSpacePosition.xyz);
#endif
    Diff = PickD * Wt; Spec = PickS * Wt;
    return true;
}

#ifndef XLION_TAIL_SCREEN
// The material's tail from the denoised screen tail: the 4 half-size texels around the pixel, bilinear, each weighed by how close its
// distance to the eye is to this pixel's and how much its normal agrees (SurfelGIScreen's weights), times the material's own terms -
// the diffuse x albedo x (1 - F0) x (1 - metallic), the specular x the Fresnel the lobe averages (F0 A + B over A + B: Karis's analytic
// environment BRDF); none weighed: the most alike of the 4 (its view depth within 25%, its normal on the same side: a silhouette, a
// thin object - the GI's screen read does the same). 1: T is the tail; 2: hidden - the texel nearest the pixel holds a surface nearer than
// this fragment (overdraw where no depth prepass ran: this fragment is drawn over); 0: none of the 4 is this surface (a silhouette, a blended surface in front, a material without a
// G-buffer variant): the caller shades its tail exactly
int LocalTailScreen(in const vec3 N, in const float NdotV, in const vec3 Albedo, in const vec3 F0, in const float PerceptualRoughness, in const float Metallic, out vec3 T)
{
    T = vec3(0.0);
    const int  W = int(UBOLight.TailParams.y), H = int(UBOLight.TailParams.z);
    const uint R0 = uint(UBOLight.TailParams.w);
    if (W <= 0 || R0 + uint(W * H) > uint(TailScreen.length())) return 0;         // (a pass without the screen tail: its dummy bound)
    const vec2 hp = gl_FragCoord.xy * 0.5 - 0.5;
    const vec2 i0 = floor(hp), f = hp - i0;
    const float Dist = 1.0 / gl_FragCoord.w;                                    // (the view depth: clip w, as the records')
    vec3  Sd = vec3(0.0), Ss = vec3(0.0);
    float Sw = 0.0;
    float NearestW = -1.0, NearestD = 0.0;                                      // (the texel nearest the pixel: what is drawn there)
    float Best = 0.0;                                                           // (none alike within the weights: the most alike, if close enough)
    vec3  BestD = vec3(0.0), BestS = vec3(0.0);
    for (int k = 0; k < 4; ++k)
    {
        const ivec2 P = clamp(ivec2(i0) + ivec2(k & 1, k >> 1), ivec2(0), ivec2(W - 1, H - 1));
        const uvec4 Q = TailScreen[R0 + uint(P.y * W + P.x)];
        const float d = unpackHalf2x16(Q.w & 0xFFFFu).x;
        if (d <= 0.0) continue;
        const vec3  n  = OctDecode(vec2(float((Q.w >> 16) & 15u), float((Q.w >> 20) & 15u)) / 15.0 * 2.0 - 1.0);
        const float wd = max(0.0, 1.0 - abs(d - Dist) / max(0.05 * Dist, 1e-3)), wn = pow(max(dot(n, N), 0.0), 8.0);
        const float wb = ((k & 1) != 0 ? f.x : 1.0 - f.x) * ((k >> 1) != 0 ? f.y : 1.0 - f.y);
        if (wb > NearestW) { NearestW = wb; NearestD = d; }
        const float w  = wb * wd * wd * wn;
        const float a  = max(0.0, 1.0 - abs(d - Dist) / (0.25 * Dist)) * max(dot(n, N), 0.0);
        if (a > Best) { Best = a; BestD = vec3(unpackHalf2x16(Q.x), unpackHalf2x16(Q.y).x); BestS = vec3(unpackHalf2x16(Q.y).y, unpackHalf2x16(Q.z)); }
        const vec3 qd = vec3(unpackHalf2x16(Q.x), unpackHalf2x16(Q.y).x), qs = vec3(unpackHalf2x16(Q.y).y, unpackHalf2x16(Q.z));
        if (any(isnan(qd)) || any(isinf(qd)) || any(isnan(qs)) || any(isinf(qs))) continue;      // (not finite: none)
        Sd += w * qd;
        Ss += w * qs;
        Sw += w;
    }
    if (Sw < 1e-4)
    {
        if (Best < 0.25) return NearestW >= 0.0 && NearestD < Dist * 0.95 ? 2 : 0;     // (nothing like it: hidden where its own texel holds a nearer surface, else exact)
        Sd = BestD; Ss = BestS; Sw = 1.0;                                       // (a silhouette, a thin object: its most alike texel)
    }
    const vec4  c0 = vec4(-1.0, -0.0275, -0.572, 0.022), c1 = vec4(1.0, 0.0425, 1.04, -0.04);      // (Karis, "Physically based shading on mobile")
    const vec4  r  = PerceptualRoughness * c0 + c1;
    const float a004 = min(r.x * r.x, exp2(-9.28 * NdotV)) * r.x + r.y;
    const vec2  AB = vec2(-1.04, 1.04) * a004 + r.zw;
    const vec3  SpecR = (F0 * AB.x + AB.y) / max(AB.x + AB.y, 1e-4);
    const vec3  DiffR = Albedo * (vec3(1.0) - F0) * (1.0 - Metallic);
    T = Sd / Sw * DiffR + Ss / Sw * SpecR;
    return 1;
}
#endif

// Modes: 0 every light; 1 the cell's list (the same result: the list holds every light that reaches the cell); 2 the cell's
// head exactly + tail lights J drawn with probability p(J) (LocalBase.w stratified samples), each weighted 1 / (p(J) samples); 3 the head
// exactly + the denoised screen tail (LocalTailScreen; where it does not apply, mode 2's tail). Outside the cells: every light.
vec3 LocalLights(in const vec3 N, in const vec3 V, in const float NdotV, in const vec3 Albedo, in const vec3 F0, in const float PerceptualRoughness, in const float Metallic)
{
    vec3 Sum = vec3(0.0);
    if (UBOLight.LocalCount.x < 0.5) return Sum;                       // (no lights: no cells either)
    const int Mode = int(UBOLight.LocalCount.y);
    const int Cell = Mode == 0 ? -1 : LocalLightCell(In.wSpacePosition.xyz + UBOLight.CameraPos.xyz);
    if (Cell < 0)
    {
        const int Count = int(UBOLight.LocalCount.x);
        for (int i = 0; i < Count; ++i) Sum += LocalLight(i, N, V, NdotV, Albedo, F0, PerceptualRoughness, Metallic);
        return Sum;
    }

    const uint Base      = uint(UBOLight.LocalBase.y);
    const uint Header    = LightCells[Base + 2u * uint(Cell) + 1u];      // the header: the first entry; the count (bits 0-23), the head count (24-31)
    const uint EntryBase = Base + uint(2 * kCellLevels * kLevelCells);
    const uint First     = LightCells[Base + 2u * uint(Cell)], Count = Header & 0xFFFFFFu, Head = Header >> 24;
    const uint nExact = Mode == 1 ? Count : Head;
    bool bInlineTail = Mode == 2;
#ifndef XLION_TAIL_SCREEN
    if (Mode == 3 && Count > Head)
    {
        // the screen tail; where it does not apply - a silhouette it has no texel for, a blended surface, a material without a G-buffer
        // variant - the inline resampled tail (unbiased, noisy); a hidden fragment (drawn over where no depth prepass ran) or a helper
        // invocation: none. (The whole list instead cost 3.4 ms against 0.25 at the rig's 4096 lights: the overdraw's hidden fragments)
        vec3 T;
        const int Screen = LocalTailScreen(N, NdotV, Albedo, F0, PerceptualRoughness, Metallic, T);
        if (Screen == 1) Sum = T;
        bInlineTail = Screen == 0 && !gl_HelperInvocation;
    }
#endif
    for (uint k = 0u; k < nExact; ++k) Sum += LocalLight(int(LightCells[EntryBase + 2u * (First + k)]), N, V, NdotV, Albedo, F0, PerceptualRoughness, Metallic);

    if (bInlineTail && Count > Head && abs(UBOLight.LocalCount.w) > 1.5)
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
        float WSum = 0.0, PickP = 1.0;
        vec3  PickC = vec3(0.0);
        uint  e    = Head;
        while (k < M)
        {
            // the next candidate's entry (the samples grow: from the last one on), and the samples after it that fall in it too
            e = LocalTailFind(EntryBase, First, e, Count, (float(k) + u0) / float(M));
            const float Cdf  = LocalTailCdf(EntryBase, First, e);
            const float Prev = e > Head ? LocalTailCdf(EntryBase, First, e - 1u) : 0.0;
            int nHits = 1;
            ++k;
            while (k < M && ((float(k) + u0) / float(M) < Cdf || e + 1u == Count)) { ++nHits; ++k; }
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
            ++e;
        }
        if (Pick >= 0) Sum += PickC * LocalLightShadow(Pick, N) * (WSum / (float(M) * PickP));
    }
    else if (bInlineTail && Count > Head)
    {
        // Samples stratified tail samples (u_k = (k + u) / Samples), each found by a binary search from the last one on
        const int   Samples = max(int(UBOLight.LocalBase.w), 1);
        const float u0      = LocalRandom();
        int   k    = 0;
        uint  e    = Head;
        while (k < Samples)
        {
            e = LocalTailFind(EntryBase, First, e, Count, (float(k) + u0) / float(Samples));
            const float Cdf  = LocalTailCdf(EntryBase, First, e);
            const float Prev = e > Head ? LocalTailCdf(EntryBase, First, e - 1u) : 0.0;
            int nHits = 1;
            ++k;
            while (k < Samples && ((float(k) + u0) / float(Samples) < Cdf || e + 1u == Count)) { ++nHits; ++k; }
            Sum += LocalLight(int(LightCells[EntryBase + 2u * (First + e)]), N, V, NdotV, Albedo, F0, PerceptualRoughness, Metallic) * (float(nHits) / (float(Samples) * (Cdf - Prev)));
            ++e;
        }
    }
    return Sum;
}
