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

// 1 lit .. 0 shadowed. Receiver: the surface minus the light's position, N its normal. The light's map (Shadow: xy origin in the
// atlas, z size) holds the linear distance to the nearest caster / radius: octahedral for a point light, the frustum for a spot.
// Against acne the receiver moves along its normal (normal offset: the error grows as the light grazes the
// surface, a constant bias does not follow it) by two texels, plus a one texel bias.
float LocalShadow(in const LocalLightData Ld, in const vec3 Receiver, in const vec3 N)
{
    if (Ld.Shadow.z <= 0.0) return 1.0;

    const float kBorder = 2.0;                                                  // shadows::local_border_v
    const float Size    = Ld.Shadow.z;
    const float Inner   = Size - 2.0 * kBorder;
    const bool  bPoint  = Ld.DirCosOuter.w < -1.5;
    const float TexelWorld = length(Receiver) * (bPoint ? 4.0 : 2.0 * Ld.Shadow.w) / Inner;    // a texel's size at the receiver (the
                                                                                               // octahedral square spans ~4 radians)
    const vec3  FromLight = Receiver + N * (2.0 * TexelWorld);
    const float Dist      = length(FromLight);
    vec2  UV;
    if (bPoint)
    {
        UV = OctEncode(FromLight / Dist);
    }
    else
    {
        const vec3  F       = Ld.DirCosOuter.xyz;
        const vec3  Up      = abs(F.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
        const vec3  R       = normalize(cross(Up, F));
        const vec3  U       = cross(F, R);
        const float TanHalf = Ld.Shadow.w;
        const float d       = dot(FromLight, F);
        if (d <= 0.0) return 1.0;
        UV = vec2(dot(FromLight, R), dot(FromLight, U)) / (d * TanHalf);
        if (any(greaterThan(abs(UV), vec2(1.0)))) return 1.0;                  // outside the spot's frustum
    }

    const float Ref   = (Dist - TexelWorld) / Ld.PosRadius.w;
    const ivec2 Base  = ivec2(Ld.Shadow.xy);
    const ivec2 Texel = ivec2(kBorder + floor((UV * 0.5 + 0.5) * Inner));
    float Lit = 0.0;
    for (int y = -1; y <= 1; ++y)
    for (int x = -1; x <= 1; ++x)
    {
        const ivec2 P = clamp(Texel + ivec2(x, y), ivec2(0), ivec2(int(Size) - 1));     // the border holds the folded neighbors
        Lit += (texelFetch(SamplerShadowMap, Base + P, 0).r < Ref) ? 0.0 : 1.0;
    }
    return Lit / 9.0;
}

// The local lights (storage buffers, so a scene is not limited to a uniform block): the lights, and the world-cell lists
// (features/lights/xlionrender_light_cells.h builds them; levels_v 5, cells_v 16, a packed uint header per cell, 2 uint entries)
layout(std430, set = 1, binding = 3) readonly buffer LocalLightBuffer { LocalLightData LocalLightList[]; };
layout(std430, set = 1, binding = 4) readonly buffer LightCellBuffer  { uint LightCells[]; };

const int kCellLevels = 5;
const int kCells      = 16;
const int kLevelCells = kCells * kCells * kCells;

// One light's contribution at the surface (0 outside its radius)
vec3 LocalLight(in const int i, in const vec3 N, in const vec3 V, in const float NdotV, in const vec3 Albedo, in const vec3 F0, in const float PerceptualRoughness, in const float Metallic)
{
    const LocalLightData Ld = LocalLightList[int(UBOLight.LocalBase.x) + i];
    const vec3  ToLight = Ld.PosRadius.xyz - In.wSpacePosition.xyz;            // both camera-centered
    const float Dist    = length(ToLight);
    const float s       = Dist / Ld.PosRadius.w;
    if (s >= 1.0 || Dist < 1e-6) return vec3(0.0);
    float Att = (1.0 - s * s * s * s) / (1.0 + 2.0 * s * s);                    // the main light's falloff (Attenuate) without its area term
    const vec3 L = ToLight / Dist;
    if (Ld.DirCosOuter.w > -1.5) Att *= smoothstep(Ld.DirCosOuter.w, Ld.Color.w, dot(-L, Ld.DirCosOuter.xyz));     // spot cone: outer -> inner
    if (Att < 1e-4) return vec3(0.0);
    return PBRDirect(N, V, NdotV, Albedo, F0, PerceptualRoughness, Metallic, L) * Ld.Color.rgb * Att * LocalShadow(Ld, -ToLight, N);
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

// [0, 1) per pixel and per seed
float LocalRandom()
{
    uint h = uint(gl_FragCoord.x) * 1973u + uint(gl_FragCoord.y) * 9277u + uint(UBOLight.LocalCount.z) * 26699u;
    h = (h ^ 61u) ^ (h >> 16); h *= 9u; h ^= h >> 4; h *= 0x27d4eb2du; h ^= h >> 15;
    return float(h >> 8) * (1.0 / 16777216.0);
}

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

    if (Mode == 2 && Count > Head)
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
