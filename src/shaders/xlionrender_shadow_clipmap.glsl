// The directional light's shadow: a lookup in the clipmap of cached tiles of the shadow atlas (features/shadows/xlionrender_shadow_clipmap.h
// builds them; the constants below must match it). Included by mb_standard_pbr.frag after lighting_uniforms (UBOLight). The atlas is the
// sampler at binding 0 (SamplerShadowMap, mb_standard_shadow.frag).

const int   kClipmapLevels  = 4;
const int   kClipmapTiles   = 4;        // tiles per level side

// World position -> 1 lit, 0 in shadow (3x3 PCF inside the tile). The finest level that holds the point decides; outside them all: lit.
float ShadowClipmap(in const vec3 World)
{
    if (UBOLight.ShadowParams.x == 0.0) return 1.0;

    const vec3 L = (UBOLight.W2Light * vec4(World, 1.0)).xyz;
    for (int Level = 0; Level < kClipmapLevels; ++Level)
    {
        const float kTileTexels = UBOLight.ShadowParams.z;    // texels per tile side (shadows::setup)
        const vec4  Lv   = UBOLight.ShadowLevels[Level];       // xy: first tile of the level's window, z: tile size, w: depth center
        const vec2  T    = L.xy / Lv.z;
        const vec2  Tile = floor(T);
        const vec2  Rel  = Tile - Lv.xy;
        if (any(lessThan(Rel, vec2(0.0))) || any(greaterThanEqual(Rel, vec2(float(kClipmapTiles))))) continue;

        const float Half  = UBOLight.ShadowParams.y * float(1 << Level);                 // the level's depth half range
        const float Depth = clamp((L.z - (Lv.w - Half)) / (2.0 * Half), 0.0, 1.0);
        const float Bias  = 1.5 * (Lv.z / kTileTexels) / (2.0 * Half);                  // a texel and a half, in the level's depth units
        const vec2  Slot  = mod(Tile, float(kClipmapTiles));                             // world-anchored: the slot is the index mod the window
        const vec2  Block = vec2(float(Level % 2), float(Level / 2)) * float(kClipmapTiles);
        // Exact texels (texelFetch, no filtering), clamped inside the tile: the neighbor slot of the atlas holds an unrelated part of the
        // world, and a filtered fetch at a tile's edge blended it in (shadow dots along the tile boundaries)
        const ivec2 Base  = ivec2(Block + Slot) * int(kTileTexels);
        const ivec2 Texel = ivec2(floor((T - Tile) * kTileTexels));

        float Lit = 0.0;
        for (int y = -1; y <= 1; ++y)
        for (int x = -1; x <= 1; ++x)
        {
            const ivec2 P = clamp(Texel + ivec2(x, y), ivec2(0), ivec2(int(kTileTexels) - 1));
            Lit += (texelFetch(SamplerShadowMap, Base + P, 0).r < Depth - Bias) ? 0.0 : 1.0;
        }
        return Lit / 9.0;
    }
    return 1.0;
}
