// The sun's shadow from its virtual pages (features/shadows/xlionrender_shadow_vsm.h; documentation/lighting_strategy.md sec. 7).
// Included by mb_standard_pbr.frag after xlionrender_local_lights.glsl (the page table is in LightCells: one buffer, no new binding).
//  - Level: the one whose texel fits the pixel's light-space footprint (max of the derivatives), as the request pass picks it.
//  - Against acne: the receiver moves along its geometric normal by kVSMNormalOffset texels of the level it reads (finer texels:
//    less bias, so a constant depth bias alone shadowed curved surfaces), then a kVSMDepthBias texel depth bias, and per filter tap the
//    receiver plane's rise toward that tap's texel (Unreal's ComputeVirtualShadowMapOptimalSlopeBias: twice it, never negative, clamped
//    to kVSMMaxSlopeBias texels) - a grazing receiver (a low sun) compared with its own depth at each texel, not its centre's.
//  - Filter: a 3 x 3 PCF with bilinear weights over the 4 x 4 texels around the point, compares by hand (texelFetch: a physical page's
//    neighbour in the atlas is an unrelated page). Inside one page: one table read. Across a page border: each tap's page read from
//    the table. A tap whose page is not VALID this frame fails the level: the next coarser one is tried, down to the pinned net.
//  - Nothing valid on any level (the net not ready): lit; the request pass counts it.
// SunShadow(): the sun's shadow (the clipmap it replaced went 2026-10-27; its UBO fields stay reserved for the compiled materials).

#define XLIONRENDER_SHADOW_VSM_GLSL
const int   kVSMPageTexels   = 128;
const float kVSMNormalOffset = 1.5;
const float kVSMDepthBias    = 1.0;
const float kVSMDepthRange   = 4.0;
const float kVSMMaxSlopeBias = 8.0;     // texels of depth: the receiver plane's bias at most (Unreal clamps it too: flickering lit pixels)     // vsm::depth_range_v: a level's depth half range, in its window's extents

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

// The 16-bit entry of a level's page (0: none), from a table (this frame's epoch's, or the previous one's: TableBase)
uint VSMEntry(in const int Level, in const ivec2 Page, in const uint TableBase)
{
    const int   W = int(UBOLight.VSMParams2.x);
    const ivec2 T = ((Page % W) + W) % W;
    const uint  i = uint((Level * W + T.y) * W + T.x);
    return (LightCells[TableBase + (i >> 1)] >> ((i & 1u) * 16u)) & 0xFFFFu;
}

bool VSMInWindow(in const vec4 Lv, in const ivec2 Page)
{
    const ivec2 Rel = Page - ivec2(Lv.xy);
    return all(greaterThanEqual(Rel, ivec2(0))) && all(lessThan(Rel, ivec2(int(UBOLight.VSMParams2.x))));
}

// One level of an epoch (Lv: its level's window and depth centre, Texel0: its level 0 texel, TableBase: its table): 1 lit .. 0 shadowed,
// or -1 when a tap's page is not there
float ShadowVSMLevel(in const int Level, in const vec3 L, in const vec2 Slope, in const vec4 Lv, in const float Texel0, in const uint TableBase)
{
    const float Texel = Texel0 * float(1 << Level);
    const float Half  = Lv.z * UBOLight.VSMParams2.x * kVSMDepthRange;                 // the level's depth half range (vsm::DepthHalfRange)
    const float Depth = clamp((L.z - (Lv.w - Half)) / (2.0 * Half), 0.0, 1.0);
    const float Bias  = kVSMDepthBias * Texel / (2.0 * Half);
    const vec2  U     = L.xy / Texel - 0.5;
    const ivec2 Lo    = ivec2(floor(U)) - 1;
    const vec2  f     = U - floor(U);
    const vec2  SlopeT = Slope * Texel / (2.0 * Half);                                // the receiver plane's depth a texel along x, y
    const float MaxB   = kVSMMaxSlopeBias * Texel / (2.0 * Half);
    const ivec2 PLo   = ivec2(floor(vec2(Lo) / float(kVSMPageTexels))), PHi = ivec2(floor(vec2(Lo + 3) / float(kVSMPageTexels)));
    if (!VSMInWindow(Lv, PLo) || !VSMInWindow(Lv, PHi)) return -1.0;

    const float Row = UBOLight.VSMParams2.z;                                           // slots per row of the pool
    const vec4  Wx  = vec4(1.0 - f.x, 1.0, 1.0, f.x), Wy = vec4(1.0 - f.y, 1.0, 1.0, f.y);
    float Lit = 0.0;
    if (PLo == PHi && UBOLight.VSMParams2.w == 0.0)                                     // (w: every tap through the table - the tests' check)
    {
        const uint E = VSMEntry(Level, PLo, TableBase);
        if (E == 0u) return -1.0;
        const ivec2 Base = ivec2(int(E - 1u) % int(Row), int(E - 1u) / int(Row)) * kVSMPageTexels - PLo * kVSMPageTexels;
        for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 4; ++x)
            Lit += Wx[x] * Wy[y] * ((texelFetch(SamplerShadowMap, Base + Lo + ivec2(x, y), 0).r < Depth - Bias - min(2.0 * max(0.0, dot(SlopeT, vec2(Lo + ivec2(x, y)) - U)), MaxB)) ? 0.0 : 1.0);
        return Lit / 9.0;
    }
    // Across a page border: the footprint (4 texels) touches at most 2 x 2 pages - their entries once, each tap picks its page by which side
    // of the border it is on (was a table read and a division a tap)
    ivec2 Base[4];
    for (int j = 0; j < 4; ++j)
    {
        const ivec2 P = ivec2((j & 1) != 0 ? PHi.x : PLo.x, (j & 2) != 0 ? PHi.y : PLo.y);
        const uint  E = VSMEntry(Level, P, TableBase);
        if (E == 0u) return -1.0;
        Base[j] = ivec2(int(E - 1u) % int(Row), int(E - 1u) / int(Row)) * kVSMPageTexels - P * kVSMPageTexels;
    }
    const ivec2 Border = PHi * kVSMPageTexels;                                          // (PLo == PHi on an axis: no tap reaches it)
    for (int y = 0; y < 4; ++y)
    for (int x = 0; x < 4; ++x)
    {
        const ivec2 T = Lo + ivec2(x, y);
        const int   j = (T.x >= Border.x && PHi.x != PLo.x ? 1 : 0) + (T.y >= Border.y && PHi.y != PLo.y ? 2 : 0);
        Lit += Wx[x] * Wy[y] * ((texelFetch(SamplerShadowMap, Base[j] + T, 0).r < Depth - Bias - min(2.0 * max(0.0, dot(SlopeT, vec2(T) - U)), MaxB)) ? 0.0 : 1.0);
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
    const vec3  NL      = mat3(UBOLight.W2Light) * Ng;                                  // the receiver plane in light space: z per x, y
    const vec2  Slope   = -NL.xy / max(abs(NL.z), 1e-3) * sign(NL.z + 1e-9);
    const float Grazing = 1.0 + 2.0 * (1.0 - abs(NL.z));                              // the normal offset grows as the surface turns from the light
    // The previous epoch (the moving sun, P1e): where this one has no page yet its own, and while the cross-fade lasts the two blended
    const bool  bPrev   = UBOLight.VSMPrevParams.x != 0.0;
    const vec3  NLp     = mat3(UBOLight.VSMPrevW2Light) * Ng;
    const vec2  SlopeP  = -NLp.xy / max(abs(NLp.z), 1e-3) * sign(NLp.z + 1e-9);
    for (int Level = First; Level < nLevels; ++Level)
    {
        const vec3  Off = World + Ng * (kVSMNormalOffset * Grazing * UBOLight.VSMParams.y * float(1 << Level));
        float S = ShadowVSMLevel(Level, (UBOLight.W2Light * vec4(Off, 1.0)).xyz, Slope, UBOLight.VSMLevels[Level], UBOLight.VSMParams.y, uint(UBOLight.VSMParams.w));
        if (bPrev && (S < 0.0 || UBOLight.VSMPrevParams.y < 1.0))
        {
            const vec3  OffP = World + Ng * (kVSMNormalOffset * Grazing * UBOLight.VSMPrevParams.z * float(1 << Level));
            const float SP   = ShadowVSMLevel(Level, (UBOLight.VSMPrevW2Light * vec4(OffP, 1.0)).xyz, SlopeP, UBOLight.VSMPrevLevels[Level], UBOLight.VSMPrevParams.z, uint(UBOLight.VSMPrevParams.w));
            S = S < 0.0 ? SP : SP < 0.0 ? S : mix(SP, S, UBOLight.VSMPrevParams.y);
        }
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

// The sun's shadow (VSMParams.x 0: the sun casts none)
float SunShadow(in const vec3 World, in const vec3 Ng)
{
    return UBOLight.VSMParams.x != 0.0 ? ShadowVSM(World, Ng) : 1.0;
}
