// The sun's shadow from its virtual pages (features/shadows/xlionrender_shadow_vsm.h; documentation/lighting_strategy.md sec. 7).
// Included by mb_standard_pbr.frag after xlionrender_local_lights.glsl (the page table is in LightCells: one buffer, no new binding).
//  - Level: the one whose texel fits the pixel's light-space footprint (max of the derivatives), as the request pass picks it.
//  - Filter: a 3 x 3 PCF with bilinear weights over the 4 x 4 texels around the point, compares by hand (texelFetch: a physical page's
//    neighbour in the atlas is an unrelated page). Inside one page: one table read. Across a page border: each tap's page read from
//    the table. A tap whose page is not VALID this frame fails the level: the next coarser one is tried, down to the pinned net.
//  - Nothing valid on any level (the net not ready): lit; the request pass counts it.
// SunShadow() picks this or the clipmap (ShadowClipmap) by UBOLight.VSMParams.x.

const int kVSMPageTexels = 128;

// The 16-bit entry of a level's page (0: none), from this frame's table
uint VSMEntry(in const int Level, in const ivec2 Page)
{
    const int   W = int(UBOLight.VSMParams2.x);
    const ivec2 T = ((Page % W) + W) % W;
    const uint  i = uint((Level * W + T.y) * W + T.x);
    return (LightCells[uint(UBOLight.VSMParams.w) + (i >> 1)] >> ((i & 1u) * 16u)) & 0xFFFFu;
}

bool VSMInWindow(in const int Level, in const ivec2 Page)
{
    const ivec2 Rel = Page - ivec2(UBOLight.VSMLevels[Level].xy);
    return all(greaterThanEqual(Rel, ivec2(0))) && all(lessThan(Rel, ivec2(int(UBOLight.VSMParams2.x))));
}

// One level: 1 lit .. 0 shadowed, or -1 when a tap's page is not there
float ShadowVSMLevel(in const int Level, in const vec3 L)
{
    const float Texel = UBOLight.VSMParams.y * float(1 << Level);
    const vec4  Lv    = UBOLight.VSMLevels[Level];
    const float Half  = Lv.z * UBOLight.VSMParams2.x;                                  // the level's depth half range (vsm::DepthHalfRange)
    const float Depth = clamp((L.z - (Lv.w - Half)) / (2.0 * Half), 0.0, 1.0);
    const float Bias  = 1.5 * Texel / (2.0 * Half);                                    // a texel and a half, as the clipmap's
    const vec2  U     = L.xy / Texel - 0.5;
    const ivec2 Lo    = ivec2(floor(U)) - 1;
    const vec2  f     = U - floor(U);
    const ivec2 PLo   = ivec2(floor(vec2(Lo) / float(kVSMPageTexels))), PHi = ivec2(floor(vec2(Lo + 3) / float(kVSMPageTexels)));
    if (!VSMInWindow(Level, PLo) || !VSMInWindow(Level, PHi)) return -1.0;

    const float Row = UBOLight.VSMParams2.z;                                           // slots per row of the pool
    const vec4  Wx  = vec4(1.0 - f.x, 1.0, 1.0, f.x), Wy = vec4(1.0 - f.y, 1.0, 1.0, f.y);
    float Lit = 0.0;
    if (PLo == PHi && UBOLight.VSMParams2.w == 0.0)                                     // (w: every tap through the table - the tests' check)
    {
        const uint E = VSMEntry(Level, PLo);
        if (E == 0u) return -1.0;
        const ivec2 Base = ivec2(int(E - 1u) % int(Row), int(E - 1u) / int(Row)) * kVSMPageTexels - PLo * kVSMPageTexels;
        for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 4; ++x)
            Lit += Wx[x] * Wy[y] * ((texelFetch(SamplerShadowMap, Base + Lo + ivec2(x, y), 0).r < Depth - Bias) ? 0.0 : 1.0);
        return Lit / 9.0;
    }
    for (int y = 0; y < 4; ++y)                                                        // across a page border: each tap through the table
    for (int x = 0; x < 4; ++x)
    {
        const ivec2 T = Lo + ivec2(x, y);
        const ivec2 P = ivec2(floor(vec2(T) / float(kVSMPageTexels)));
        const uint  E = VSMEntry(Level, P);
        if (E == 0u) return -1.0;
        const ivec2 Base = ivec2(int(E - 1u) % int(Row), int(E - 1u) / int(Row)) * kVSMPageTexels;
        Lit += Wx[x] * Wy[y] * ((texelFetch(SamplerShadowMap, Base + (T - P * kVSMPageTexels), 0).r < Depth - Bias) ? 0.0 : 1.0);
    }
    return Lit / 9.0;
}

float ShadowVSM(in const vec3 World)
{
    const vec3  L = (UBOLight.W2Light * vec4(World, 1.0)).xyz;
    const float Foot = max(max(length(dFdx(L.xy)), length(dFdy(L.xy))), 1e-9);
    const int   nLevels = int(UBOLight.VSMParams.z);
    for (int Level = clamp(int(floor(log2(Foot / UBOLight.VSMParams.y))), 0, nLevels - 1); Level < nLevels; ++Level)
    {
        const float S = ShadowVSMLevel(Level, L);
        if (S >= 0.0) return S;
    }
    return 1.0;
}

// The sun's shadow: the virtual pages, or the clipmap (the rollback)
float SunShadow(in const vec3 World)
{
    return UBOLight.VSMParams.x != 0.0 ? ShadowVSM(World) : ShadowClipmap(World);
}
