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
const float kVSMLevelBand    = 0.1;     // the share of a level (in its footprint's log2) that cross-fades into the next
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

// The 2 x 2 texels from At in one gather (the values texelFetch reads: the atlas is sampled unfiltered): x (0, 1), y (1, 1), z (1, 0), w (0, 0)
vec4 VSMGather(in const ivec2 At)
{
    return textureGather(SamplerShadowMap, (vec2(At) + 1.0) / vec2(textureSize(SamplerShadowMap, 0)));
}

// A page's 4 x 4 texels from Lo (and the movers' page's, the nearer: bMover) in 4 gathers; VSMTap4x4 picks tap (x, y) - the sums keep their
// row order: bit for bit the 16 fetches' result
void VSMGather4x4(in const ivec2 Lo, in const ivec2 Base, in const ivec2 MBase, in const bool bMover, out vec4 G[4])
{
    for (int b = 0; b < 4; ++b)
    {
        const ivec2 At = Lo + ivec2((b & 1) * 2, (b >> 1) * 2);
        G[b] = VSMGather(Base + At);
        if (bMover) G[b] = min(G[b], VSMGather(MBase + At));
    }
}
float VSMTap4x4(in const vec4 G[4], in const int x, in const int y)
{
    const vec4 B = G[(y >> 1) * 2 + (x >> 1)];
    return (y & 1) != 0 ? ((x & 1) != 0 ? B.y : B.x) : ((x & 1) != 0 ? B.z : B.w);
}

// The mover page's origin for a page (the atlas texel of its page-space 0, as Base), or false when the movers have none there
bool VSMMoverBase(in const int Level, in const ivec2 Page, in const uint MoverBase, out ivec2 Base)
{
    Base = ivec2(0);
    if (MoverBase == 0u) return false;
    const uint E = VSMEntry(Level, Page, MoverBase - 1u);
    if (E == 0u) return false;
    const float Row = UBOLight.VSMParams2.z;
    Base = ivec2(int(E - 1u) % int(Row), int(E - 1u) / int(Row)) * kVSMPageTexels - Page * kVSMPageTexels;
    return true;
}

// One level of an epoch (Lv: its level's window and depth centre, Texel0: its level 0 texel, TableBase: its table, MoverBase: the
// movers' table + 1, 0: none): 1 lit .. 0 shadowed, or -1 when a tap's page is not there
float ShadowVSMLevel(in const int Level, in const vec3 L, in const vec2 Slope, in const vec4 Lv, in const float Texel0, in const uint TableBase, in const uint MoverBase)
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
        ivec2 MBase;
        const bool bM = VSMMoverBase(Level, PLo, MoverBase, MBase);
        vec4 G[4];                                                                     // (the 16 taps in 4 gathers - were 16 fetches)
        VSMGather4x4(Lo, Base, MBase, bM, G);
        for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 4; ++x)
            Lit += Wx[x] * Wy[y] * ((VSMTap4x4(G, x, y) < Depth - Bias - min(2.0 * max(0.0, dot(SlopeT, vec2(Lo + ivec2(x, y)) - U)), MaxB)) ? 0.0 : 1.0);
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
        float d = texelFetch(SamplerShadowMap, Base[j] + T, 0).r;
        if (MoverBase != 0u)                                                           // (movers: the tap's mover page looked up here - few registers kept)
        {
            ivec2 MB;
            if (VSMMoverBase(Level, ivec2((j & 1) != 0 ? PHi.x : PLo.x, (j & 2) != 0 ? PHi.y : PLo.y), MoverBase, MB)) d = min(d, texelFetch(SamplerShadowMap, MB + T, 0).r);
        }
        Lit += Wx[x] * Wy[y] * ((d < Depth - Bias - min(2.0 * max(0.0, dot(SlopeT, vec2(T) - U)), MaxB)) ? 0.0 : 1.0);
    }
    return Lit / 9.0;
}

// One level, the current epoch's, or the previous one's where the current has no page yet (the moving sun, P1e) and while the cross-fade
// lasts the two blended; -1 when neither has it
float ShadowVSMAt(in const int Level, in const vec3 World, in const vec3 Ng, in const vec2 Slope, in const vec2 SlopeP, in const float Grazing)
{
    const vec3 Off = World + Ng * (kVSMNormalOffset * Grazing * UBOLight.VSMParams.y * float(1 << Level));
    const vec3 Lo = (UBOLight.W2Light * vec4(Off, 1.0)).xyz;
    float S = UBOLight.VSMParams3.w == 0.0                                            // (no mover this frame: a copy without the movers' code)
            ? ShadowVSMLevel(Level, Lo, Slope, UBOLight.VSMLevels[Level], UBOLight.VSMParams.y, uint(UBOLight.VSMParams.w), 0u)
            : ShadowVSMLevel(Level, Lo, Slope, UBOLight.VSMLevels[Level], UBOLight.VSMParams.y, uint(UBOLight.VSMParams.w), uint(UBOLight.VSMParams3.w));
    if (UBOLight.VSMPrevParams.x != 0.0 && (S < 0.0 || UBOLight.VSMPrevParams.y < 1.0))
    {
        const vec3  OffP = World + Ng * (kVSMNormalOffset * Grazing * UBOLight.VSMPrevParams.z * float(1 << Level));
        const float SP   = ShadowVSMLevel(Level, (UBOLight.VSMPrevW2Light * vec4(OffP, 1.0)).xyz, SlopeP, UBOLight.VSMPrevLevels[Level], UBOLight.VSMPrevParams.z, uint(UBOLight.VSMPrevParams.w), 0u);
        S = S < 0.0 ? SP : SP < 0.0 ? S : mix(SP, S, UBOLight.VSMPrevParams.y);
    }
    return S;
}

// Contact-hardening's blocker search (the current epoch): the 4 x 4 texels of a level around the point, the light-space distance from the
// receiver to the mean of those nearer the light than it. -2: the level's pages are not there; -1: no blocker
float VSMBlockerDistance(in const int Level, in const vec3 L, in const uint MoverBase)
{
    const vec4  Lv    = UBOLight.VSMLevels[Level];
    const float Texel = UBOLight.VSMParams.y * float(1 << Level);
    const float Half  = Lv.z * UBOLight.VSMParams2.x * kVSMDepthRange;
    const float Depth = clamp((L.z - (Lv.w - Half)) / (2.0 * Half), 0.0, 1.0);
    const float Bias  = 2.0 * kVSMDepthBias * Texel / (2.0 * Half);
    const ivec2 Lo    = ivec2(floor(L.xy / Texel - 0.5)) - 1;
    const ivec2 PLo   = ivec2(floor(vec2(Lo) / float(kVSMPageTexels))), PHi = ivec2(floor(vec2(Lo + 3) / float(kVSMPageTexels)));
    if (!VSMInWindow(Lv, PLo) || !VSMInWindow(Lv, PHi)) return -2.0;
    const float Row = UBOLight.VSMParams2.z;
    float Sum = 0.0, n = 0.0;
    if (PLo == PHi)                                                                    // (inside one page: the 16 taps in 4 gathers)
    {
        const uint E = VSMEntry(Level, PLo, uint(UBOLight.VSMParams.w));
        if (E == 0u) return -2.0;
        const ivec2 Base = ivec2(int(E - 1u) % int(Row), int(E - 1u) / int(Row)) * kVSMPageTexels - PLo * kVSMPageTexels;
        ivec2 MBase;
        const bool bM = VSMMoverBase(Level, PLo, MoverBase, MBase);
        vec4 G[4];
        VSMGather4x4(Lo, Base, MBase, bM, G);
        for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 4; ++x)
        {
            const float d = VSMTap4x4(G, x, y);
            if (d < Depth - Bias) { Sum += d; n += 1.0; }
        }
        return n > 0.0 ? (Depth - Sum / n) * 2.0 * Half : -1.0;
    }
    // the 2 x 2 pages the footprint can touch: their entries once (one when it is inside a page), each tap picks its side of the border
    ivec2 Base[4];
    for (int j = 0; j < 4; ++j)
    {
        const ivec2 P = ivec2((j & 1) != 0 ? PHi.x : PLo.x, (j & 2) != 0 ? PHi.y : PLo.y);
        if (j > 0 && P == PLo) { Base[j] = Base[0]; continue; }
        const uint  E = VSMEntry(Level, P, uint(UBOLight.VSMParams.w));
        if (E == 0u) return -2.0;
        Base[j] = ivec2(int(E - 1u) % int(Row), int(E - 1u) / int(Row)) * kVSMPageTexels - P * kVSMPageTexels;
    }
    const ivec2 Border = PHi * kVSMPageTexels;
    for (int y = 0; y < 4; ++y)
    for (int x = 0; x < 4; ++x)
    {
        const ivec2 T = Lo + ivec2(x, y);
        const int   j = (T.x >= Border.x && PHi.x != PLo.x ? 1 : 0) + (T.y >= Border.y && PHi.y != PLo.y ? 2 : 0);
        float d = texelFetch(SamplerShadowMap, Base[j] + T, 0).r;
        if (MoverBase != 0u)
        {
            ivec2 MB;
            if (VSMMoverBase(Level, ivec2((j & 1) != 0 ? PHi.x : PLo.x, (j & 2) != 0 ? PHi.y : PLo.y), MoverBase, MB)) d = min(d, texelFetch(SamplerShadowMap, MB + T, 0).r);
        }
        if (d < Depth - Bias) { Sum += d; n += 1.0; }
    }
    return n > 0.0 ? (Depth - Sum / n) * 2.0 * Half : -1.0;
}

// The pixel's light-space footprint (the level's choice), from the screen derivatives: set by the material in uniform control flow,
// before a branch that skips the shadow (a surface facing away from the sun)
float g_VSMFoot = 1e-9;
void VSMFootprint(in const vec3 World)
{
    const vec3 L = (UBOLight.W2Light * vec4(World, 1.0)).xyz;
    g_VSMFoot = max(max(length(dFdx(L.xy)), length(dFdy(L.xy))), 1e-9);
}

float ShadowVSM(in const vec3 World, in const vec3 Ng)
{
    const vec3  L = (UBOLight.W2Light * vec4(World, 1.0)).xyz;
    const float Foot = g_VSMFoot;
    const int   nLevels = int(UBOLight.VSMParams.z);
    const int   Mode    = int(UBOLight.VSMParams2.w);
    const float Lx      = log2(Foot / UBOLight.VSMParams.y) + UBOLight.VSMParams3.x;
    const int   First   = clamp(int(floor(Lx)), 0, nLevels - 1);
    const vec3  NL      = mat3(UBOLight.W2Light) * Ng;                                  // the receiver plane in light space: z per x, y
    const vec2  Slope   = -NL.xy / max(abs(NL.z), 1e-3) * sign(NL.z + 1e-9);
    const float Grazing = 1.0 + 1.0 * (1.0 - abs(NL.z));                              // the normal offset grows as the surface turns from the light
    const vec3  NLp     = mat3(UBOLight.VSMPrevW2Light) * Ng;
    const vec2  SlopeP  = -NLp.xy / max(abs(NLp.z), 1e-3) * sign(NLp.z + 1e-9);

    // Contact-hardening (a sun with a disc, VSMParams3.y: tan of half its angle): the blocker search a few levels coarser (VSMParams3.z of
    // them, their pages asked for by the request pass), the penumbra's half-width - the distance to the blockers x tan - and the 4 x 4
    // filter at the level whose taps span it (two levels blended): sharp at the contact, wider and smooth with the distance, no noise
    // The last kVSMLevelBand of a level cross-fades into the next (CSM's cascade blend): no line where the level steps - it popped as the
    // camera moved (Tomas). The contact-hardening's level below replaces it where it is wider
    float Lf = float(First) + (int(floor(Lx)) == First && First + 1 < nLevels ? clamp((fract(Lx) - (1.0 - kVSMLevelBand)) / kVSMLevelBand, 0.0, 1.0) : 0.0);
    if (UBOLight.VSMParams3.y > 0.0)
    {
        const int Top = min(First + int(UBOLight.VSMParams3.z), nLevels - 1);
        for (int ls = Top; ls >= First; ls -= 2)                                       // (the coarsest, then two finer: a thin caster lost in a coarse level)
        {
            const float Tl = UBOLight.VSMParams.y * float(1 << ls);
            const vec3  Ls = (UBOLight.W2Light * vec4(World + Ng * (kVSMNormalOffset * Grazing * Tl), 1.0)).xyz;
            const float D  = UBOLight.VSMParams3.w == 0.0 ? VSMBlockerDistance(ls, Ls, 0u) : VSMBlockerDistance(ls, Ls, uint(UBOLight.VSMParams3.w));
            if (D < 0.0) continue;                                                     // (no page there, or no blocker: a finer level)
            {
                const float Penumbra = D * UBOLight.VSMParams3.y;                       // world units, half the width
                const float Lb = Lf;            // (no finer than the pixel's level, its band's cross-fade kept: First alone stepped at a boundary)
                Lf = clamp(float(First) + log2(max(Penumbra, 1e-9) / (1.5 * UBOLight.VSMParams.y * float(1 << First))), Lb, max(Lb, float(min(First + int(UBOLight.VSMParams3.z), nLevels - 1))));
            }
            break;
        }
    }
    const int   L0 = int(floor(Lf));
    const float t  = Lf - float(L0);
    for (int Level = L0; Level < nLevels; ++Level)
    {
        float S = ShadowVSMAt(Level, World, Ng, Slope, SlopeP, Grazing);
        if (S < 0.0) continue;
        if (t > 0.0 && Level == L0 && Level + 1 < nLevels)
        {
            const float S1 = ShadowVSMAt(Level + 1, World, Ng, Slope, SlopeP, Grazing);
            if (S1 >= 0.0) S = mix(S, S1, t);
        }
        if (Mode >= 2)
        {
            const float Texel = UBOLight.VSMParams.y * float(1 << Level);
            const vec2  InPage = fract(L.xy / (Texel * float(kVSMPageTexels)));
            const float Border = any(lessThan(min(InPage, 1.0 - InPage), vec2(1.5 / float(kVSMPageTexels)))) ? 0.4 : 1.0;
            const float Ratio  = log2(Foot / Texel);                 // 0..1: the texel within the pixel; below: texels larger
            if (Mode == 2) g_VSMDebug = VSMLevelColor(Level) * Border * (0.5 + 0.5 * S);
            if (Mode == 3) g_VSMDebug = Ratio < 0.0 ? vec3(1, 0.2, 0.2) : Ratio < 1.0 ? vec3(0.2, 1, 0.2) : vec3(0.2, 0.3, 1);
            if (Mode == 4) g_VSMDebug = (Level == First ? vec3(0.2, 1, 0.2) : vec3(1, 0.9, 0.1)) * (0.5 + 0.5 * S);
            if (Mode == 5) g_VSMDebug = VSMLevelColor(int(round(Lf)) - First + 2) * (0.5 + 0.5 * S);          // the contact-hardening's level over the pixel's
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
