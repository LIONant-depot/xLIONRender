// The sun's shadow from its virtual pages (features/shadows/xlionrender_shadow_vsm.h; documentation/lighting_strategy.md sec. 7).
// Included by mb_standard_pbr.frag after xlionrender_local_lights.glsl (the page table is in LightCells: one buffer, no new binding).
//  - Level: the one whose texel fits the pixel's light-space footprint (max of the derivatives), as the request pass picks it.
//  - Against acne: the receiver moves along its geometric normal by kVSMNormalOffset texels of the level it reads (finer texels:
//    less bias, so a constant depth bias alone shadowed curved surfaces), then a kVSMDepthBias texel depth bias.
//  - Filter: a 3 x 3 PCF with bilinear weights over the 4 x 4 texels around the point, compares by hand (texelFetch: a physical page's
//    neighbour in the atlas is an unrelated page). Inside one page: one table read. Across a page border: each tap's page read from
//    the table. A tap whose page is not VALID this frame fails the level: the next coarser one is tried, down to the pinned net.
//  - Nothing valid on any level (the net not ready): lit; the request pass counts it.
// SunShadow() picks this or the clipmap (ShadowClipmap) by UBOLight.VSMParams.x.

#define XLIONRENDER_SHADOW_VSM_GLSL
const int   kVSMPageTexels   = 128;
const float kVSMNormalOffset = 1.5;
const float kVSMDepthBias    = 1.0;

// The pages' debug views (VSMParams2.w): 2 the level per pixel (a colour a level, page borders darker), 3 the footprint against the
// texel (green: within 2x; red: the texel larger than the pixel; blue: smaller), 4 what shaded it (green its own level, yellow a coarser
// one, red none). VSMDebugOr replaces the lit colour, as GIDebugOr does
vec3 g_VSMDebug = vec3(-1.0);
vec3 VSMDebugOr(in const vec3 Lit) { return g_VSMDebug.x >= 0.0 ? g_VSMDebug : Lit; }
vec3 VSMLevelColor(in const int Level)
{
    const vec3 C[8] = vec3[8](vec3(1, 0.2, 0.2), vec3(1, 0.6, 0.1), vec3(0.9, 0.9, 0.1), vec3(0.3, 0.9, 0.2), vec3(0.1, 0.8, 0.8), vec3(0.2, 0.4, 1), vec3(0.6, 0.3, 1), vec3(0.9, 0.3, 0.8));
    return C[Level % 8] * (Level >= 8 ? 0.5 : 1.0);
}

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
    const float Bias  = kVSMDepthBias * Texel / (2.0 * Half);
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

float ShadowVSM(in const vec3 World, in const vec3 Ng)
{
    const vec3  L = (UBOLight.W2Light * vec4(World, 1.0)).xyz;
    const float Foot = max(max(length(dFdx(L.xy)), length(dFdy(L.xy))), 1e-9);
    const int   nLevels = int(UBOLight.VSMParams.z);
    const int   Mode    = int(UBOLight.VSMParams2.w);
    const int   First   = clamp(int(floor(log2(Foot / UBOLight.VSMParams.y) + UBOLight.VSMParams3.x)), 0, nLevels - 1);
    for (int Level = First; Level < nLevels; ++Level)
    {
        const float S = ShadowVSMLevel(Level, (UBOLight.W2Light * vec4(World + Ng * (kVSMNormalOffset * UBOLight.VSMParams.y * float(1 << Level)), 1.0)).xyz);
        if (S < 0.0) continue;
        if (Mode >= 2)
        {
            const float Texel = UBOLight.VSMParams.y * float(1 << Level);
            const vec2  InPage = fract(L.xy / (Texel * float(kVSMPageTexels)));
            const float Border = any(lessThan(min(InPage, 1.0 - InPage), vec2(1.5 / float(kVSMPageTexels)))) ? 0.4 : 1.0;
            const float Ratio  = log2(Foot / Texel);                 // 0..1: the texel within the pixel; below: texels larger
            if (Mode == 2) g_VSMDebug = VSMLevelColor(Level) * Border * (0.5 + 0.5 * S);
            if (Mode == 3) g_VSMDebug = Ratio < 0.0 ? vec3(1, 0.2, 0.2) : Ratio < 1.0 ? vec3(0.2, 1, 0.2) : vec3(0.2, 0.3, 1);
            if (Mode == 4) g_VSMDebug = (Level == First ? vec3(0.2, 1, 0.2) : vec3(1, 0.9, 0.1)) * (0.5 + 0.5 * S);
        }
        return S;
    }
    if (Mode >= 2) g_VSMDebug = vec3(1, 0, 0);
    return 1.0;
}

// The sun's shadow: the virtual pages, or the clipmap (the rollback)
float SunShadow(in const vec3 World, in const vec3 Ng)
{
    return UBOLight.VSMParams.x != 0.0 ? ShadowVSM(World, Ng) : ShadowClipmap(World);
}
