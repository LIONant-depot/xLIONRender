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

// 1 lit .. 0 shadowed. FromLight: the receiver minus the light's position.
float LocalShadow(in const LocalLightData Ld, in const vec3 FromLight)
{
    if (Ld.Shadow.x < 0.0) return 1.0;

    int  Tile = int(Ld.Shadow.x);
    vec3 F;
    if (Ld.DirCosOuter.w < -1.5)
    {
        // Point light: the cube face of the major axis (faces +X -X +Y -Y +Z -Z, consecutive tiles)
        const vec3 A = abs(FromLight);
        if      (A.x >= A.y && A.x >= A.z) { F = vec3(sign(FromLight.x), 0.0, 0.0); Tile += FromLight.x > 0.0 ? 0 : 1; }
        else if (A.y >= A.z)               { F = vec3(0.0, sign(FromLight.y), 0.0); Tile += FromLight.y > 0.0 ? 2 : 3; }
        else                               { F = vec3(0.0, 0.0, sign(FromLight.z)); Tile += FromLight.z > 0.0 ? 4 : 5; }
    }
    else F = Ld.DirCosOuter.xyz;

    const vec3  Up      = abs(F.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    const vec3  R       = normalize(cross(Up, F));
    const vec3  U       = cross(F, R);
    const float Near    = Ld.Shadow.y;
    const float Far     = Ld.PosRadius.w;
    const float TanHalf = Ld.Shadow.z;

    const float d = dot(FromLight, F);
    if (d <= Near) return 1.0;
    const vec2 XY = vec2(dot(FromLight, R), dot(FromLight, U)) / (d * TanHalf);
    if (any(greaterThan(abs(XY), vec2(1.0)))) return 1.0;                      // outside a spot light's frustum

    const float Bias  = 1.5 * 2.0 * d * TanHalf / kTileTexels;                   // a texel and a half at that distance
    const float dd    = max(d - Bias, Near * 1.0001);
    const float Depth = Far / (Far - Near) - (Far * Near) / ((Far - Near) * dd);

    const ivec2 Base  = ivec2((int(kAtlasTiles) + Tile % int(kAtlasTiles)) * int(kTileTexels), (Tile / int(kAtlasTiles)) * int(kTileTexels));
    const ivec2 Texel = ivec2(floor((XY * 0.5 + 0.5) * kTileTexels));
    float Lit = 0.0;
    for (int y = -1; y <= 1; ++y)
    for (int x = -1; x <= 1; ++x)
    {
        const ivec2 P = clamp(Texel + ivec2(x, y), ivec2(0), ivec2(int(kTileTexels) - 1));
        Lit += (texelFetch(SamplerShadowMap, Base + P, 0).r < Depth) ? 0.0 : 1.0;
    }
    return Lit / 9.0;
}

vec3 LocalLights(in const vec3 N, in const vec3 V, in const float NdotV, in const vec3 Albedo, in const vec3 F0, in const float PerceptualRoughness, in const float Metallic)
{
    vec3 Sum = vec3(0.0);
    const int Count = int(UBOLight.LocalCount.x);
    for (int i = 0; i < Count; ++i)
    {
        const LocalLightData Ld = UBOLight.Local[i];
        const vec3  ToLight = Ld.PosRadius.xyz - In.wSpacePosition.xyz;        // both camera-centered
        const float Dist    = length(ToLight);
        const float s       = Dist / Ld.PosRadius.w;
        if (s >= 1.0 || Dist < 1e-6) continue;
        float Att = (1.0 - s * s * s * s) / (1.0 + 2.0 * s * s);                // the main light's falloff (Attenuate) without its area term
        const vec3 L = ToLight / Dist;
        if (Ld.DirCosOuter.w > -1.5) Att *= smoothstep(Ld.DirCosOuter.w, Ld.Color.w, dot(-L, Ld.DirCosOuter.xyz));     // spot cone: outer -> inner
        if (Att < 1e-4) continue;
        Sum += PBRDirect(N, V, NdotV, Albedo, F0, PerceptualRoughness, Metallic, L) * Ld.Color.rgb * Att * LocalShadow(Ld, -ToLight);
    }
    return Sum;
}
