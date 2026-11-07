#version 450
// The sun's shadow mask (xLION_Docs shadows_gpu_plan.md S1a): a full-screen pass after the G-buffer prepass, the sun's visibility per
// pixel by the materials' own code (xlionrender_shadow_vsm.glsl ShadowVSM - not a copy), from the prepass: the camera-centred position
// from its depth, the triangle's plane and the footprint from the neighbours (per screen axis the nearer of the two, as the request
// pass measures: a silhouette does not mix two surfaces - the materials' screen derivatives stay inside the triangle), the plane turned
// toward the vertex normal (the prepass's, turned for a back face) as the materials turn theirs. The materials' layout (set 0 the shadow
// atlas and the prepass's targets in slots 1-3, set 1 the lights and their cells, set 2 the lighting block: UBOLight.TailC2W this view's
// clip -> camera-centred world). Sky: 1. Packed with the surface's distance: a material reads it only where it is that surface
#define XMAT_VARYING_IN
#define XLION_NO_GI_PIXEL
#define XLION_TAIL_SCREEN
#include "mb_varying_definition_full.glsl"
VaryingFull In;
#include "mb_standard_pbr.frag"

layout(binding = 1) uniform sampler2D GPreDepth;
layout(binding = 2) uniform sampler2D GPreNormal;      // the vertex normal, octahedral (not turned to the viewer)
layout(binding = 3) uniform sampler2D GPreId;          // y < 0: a back face

layout(std430, set = 1, binding = 20) readonly buffer vsm_bounds_b { uint VSMBounds[]; };     // per slot: least depth, greatest; then per slot the frame it rendered (xlionrender_vsm_bounds_comp.glsl)
// The reuse (XLION_SUN_MASK 6): this view's camera-centred position -> last frame's clip, this eye - last frame's, x what may be kept (0
// nothing; 1 the pages checked; 2 a page not in the table was not last frame either; 3 no page changed: no check), y this frame's number,
// z the pool's slots, w the test's view (kept pixels half)
layout(push_constant) uniform mask_push_t { mat4 Reproject; vec4 Delta; uvec4 Params; } MaskPush;

layout(location = 0) out uint OutMask;          // the visibility in 16 bits, the distance to the eye as a half (xlionrender_shadow_vsm.glsl SunShadowMasked)

bool MaskPos(in const ivec2 p, in const ivec2 Size, out vec3 P)
{
    P = vec3(0.0);
    if (any(lessThan(p, ivec2(0))) || any(greaterThanEqual(p, Size))) return false;
    const float z = texelFetch(GPreDepth, p, 0).r;
    if (z >= 1.0) return false;
    const vec4 W = UBOLight.TailC2W * vec4((vec2(p) + 0.5) / vec2(Size) * 2.0 - 1.0, z, 1.0);
    P = W.xyz / W.w;
    return true;
}

// The difference to the nearer neighbour along an axis (0: neither is a surface)
vec3 MaskDelta(in const ivec2 p, in const ivec2 Step, in const ivec2 Size, in const vec3 P)
{
    vec3 A, B;
    const bool a = MaskPos(p + Step, Size, A), b = MaskPos(p - Step, Size, B);
    if (a && (!b || length(A - P) <= length(P - B))) return A - P;
    return b ? P - B : vec3(0.0);
}

// The early-out (S1b; UBOLight.ShadowLevels[0].y 1): every tap ShadowVSM could take is lit - on every level its search, filter, disc and
// level blend can use (First .. Last), the receiver's greatest comparison depth there (the normal offset's point, raised by the receiver
// plane across the filters' widest reach W, the depth bias taken off) is nearer than the least depth of every page that reach touches.
// Then ShadowVSM returns exactly 1 (all its weights lit). Anything not known - a page not in the table (it would fall to a coarser
// level), out of the window, the previous epoch (the moving sun), movers, a debug view, more than 3 x 3 pages a level - is unresolved
// The pages ShadowVSM could read for a point (every level its search, filter, disc and blend can use, each the widest reach): bLit - the
// early-out's test (each page farther than the receiver), else the reuse's (none rendered this frame, no mover page on it). False when
// a page is not in the table, out of the window, or a level would need more than 3 x 3
bool VSMReachCheck(in const vec3 World, in const vec3 Ng, in const bool bLit)
{
    if (UBOLight.VSMPrevParams.x != 0.0 || UBOLight.VSMParams2.w != 0.0 || (bLit && UBOLight.VSMParams3.w != 0.0)) return false;
    const int   nLevels = int(UBOLight.VSMParams.z);
    const float Lx      = log2(g_VSMFoot / UBOLight.VSMParams.y) + UBOLight.VSMParams3.x;
    const int   First   = clamp(int(floor(Lx)), 0, nLevels - 1);
    const vec3  NL      = mat3(UBOLight.W2Light) * Ng;
    const vec2  Slope   = -NL.xy / max(abs(NL.z), 1e-3) * sign(NL.z + 1e-9);
    const float Grazing = 1.0 + 1.0 * (1.0 - abs(NL.z));
    const int   Top     = UBOLight.VSMParams3.y > 0.0 ? min(First + int(UBOLight.VSMParams3.z), nLevels - 1) : First;
    const int   Last    = min(max(Top, First + 1) + 1, nLevels - 1);
    // (the disc reaches 1.5 texels of Top, its taps' filters and the search's 2 of their levels, the blend half a texel of Last: under 4 of Last)
    const float W       = 4.0 * UBOLight.VSMParams.y * float(1 << Last);
    const float Rise    = (abs(Slope.x) + abs(Slope.y)) * W;
    const uint  Table   = uint(UBOLight.VSMParams.w);
    for (int Lv = First; Lv <= Last; ++Lv)
    {
        const float Texel = UBOLight.VSMParams.y * float(1 << Lv);
        const vec3  Lo    = (UBOLight.W2Light * vec4(World + Ng * (kVSMNormalOffset * Grazing * Texel), 1.0)).xyz;
        const vec4  Lev   = UBOLight.VSMLevels[Lv];
        const float Half  = Lev.z * UBOLight.VSMParams2.x * kVSMDepthRange;
        const float Need  = clamp((Lo.z + Rise - (Lev.w - Half)) / (2.0 * Half), 0.0, 1.0) - kVSMDepthBias * Texel / (2.0 * Half);
        const vec2  Ext   = vec2(W + 3.0 * Texel);
        const ivec2 PLo   = ivec2(floor((Lo.xy - Ext) / (Texel * float(kVSMPageTexels)))), PHi = ivec2(floor((Lo.xy + Ext) / (Texel * float(kVSMPageTexels))));
        if (any(greaterThan(PHi - PLo, ivec2(2)))) return false;
        for (int py = PLo.y; py <= PHi.y; ++py)
        for (int px = PLo.x; px <= PHi.x; ++px)
        {
            const ivec2 P = ivec2(px, py);
            const bool  bMissingOk = !bLit && MaskPush.Params.x >= 2u;          // (the reuse, the eye still and no page gone: as last frame)
            if (!VSMInWindow(Lev, P)) { if (bMissingOk) continue; return false; }
            const uint E = VSMEntry(Lv, P, Table);
            if (E == 0u) { if (bMissingOk) continue; return false; }
            if (bLit) { if (float(VSMBounds[E - 1u] & 0xFFFFu) / 65535.0 < Need + 1e-5) return false; }
            else
            {
                if (VSMBounds[MaskPush.Params.z + E - 1u] == MaskPush.Params.y) return false;      // (rendered this frame: changed)
                ivec2 MB;
                if (UBOLight.VSMParams3.w < 0.0 && VSMMoverBase(min(Lv + 1, int(UBOLight.VSMParams.z) - 1), ivec2(floor(vec2(P) * 0.5)), uint(-UBOLight.VSMParams3.w), MB)) return false;     // (coarse movers: the page above)
                if (UBOLight.VSMParams3.w > 0.0 && VSMMoverBase(Lv, P, uint(UBOLight.VSMParams3.w), MB)) return false;           // (a mover's page: drawn each frame)
            }
        }
    }
    return true;
}

bool VSMProvenLit(in const vec3 World, in const vec3 Ng) { return VSMReachCheck(World, Ng, true); }

// Last frame's value where it still holds: the same surface there (reprojected; the stored distance to last frame's eye), the distance
// to the eye within 2% (the level the footprint picks), every page it read unchanged (VSMReachCheck). Else ShadowVSM again
bool SunMaskReuse(in const vec3 P, in const vec3 Ng, in const ivec2 Size, out float S)
{
    S = 1.0;
    if (MaskPush.Params.x == 0u) return false;
    if (VSMNearMoverMap(P + UBOLight.CameraPos.xyz, 0.25)) return false;      // (a skinned mover's map moves every frame: no reuse in its rect, a quarter metre around for last frame's)
    const vec4 C = MaskPush.Reproject * vec4(P, 1.0);
    if (C.w <= 1e-6) return false;
    const vec2 N = C.xy / C.w;
    if (any(greaterThan(abs(N), vec2(1.0)))) return false;
    const ivec2 q     = clamp(ivec2(floor((N * 0.5 + 0.5) * vec2(Size))), ivec2(0), Size - 1);
    const uint  m     = texelFetch(SamplerSunMask, q, 0).r;
    if ((m >> 16u) != 0u && (m >> 16u) != 0xFFFFu) return false;      // (a shadow edge - partly lit - evaluated again: the jitter moves q to a neighbour each frame, its value copied pixel to pixel flickered there, Tomas 2026-11-07)
    const float dPrev = length(P + MaskPush.Delta.xyz), dNow = length(P);
    if (abs(unpackHalf2x16(m & 0xFFFFu).x - dPrev) > kSunMaskTolerance * dPrev || abs(dPrev - dNow) > 0.02 * dNow) return false;
    if (MaskPush.Params.x < 3u && !VSMReachCheck(P + UBOLight.CameraPos.xyz, Ng, false)) return false;
    S = float(m >> 16u) / 65535.0;
    return true;
}

void main()
{
    const ivec2 p    = ivec2(gl_FragCoord.xy);
    const ivec2 Size = textureSize(GPreDepth, 0);
    vec3 P;
    if (!MaskPos(p, Size, P)) { OutMask = 0xFFFF0000u; return; }
    const vec3 Dx = MaskDelta(p, ivec2(1, 0), Size, P), Dy = MaskDelta(p, ivec2(0, 1), Size, P);

    // the materials' FaceN (mb_standard_pbr.frag): the plane, turned toward the vertex normal (turned for a back face)
    const vec3 Vn   = OctDecode(texelFetch(GPreNormal, p, 0).xy);
    const vec3 VtxN = Vn * (texelFetch(GPreId, p, 0).y < 0.0 ? -1.0 : 1.0);
    vec3 FaceN = cross(Dx, Dy);
    FaceN = dot(FaceN, FaceN) > 1e-20 ? normalize(FaceN) * sign(dot(FaceN, VtxN) + 1e-9) : VtxN;

    // VSMFootprint's: the light-space derivatives, at most twice the distance's
    const mat3  L3 = mat3(UBOLight.W2Light);
    const vec3  D  = normalize(P);
    const float Fl = max(length((L3 * Dx).xy), length((L3 * Dy).xy));
    const float Fd = 2.0 * length(P) * max(length(normalize(P + Dx) - D), length(normalize(P + Dy) - D));
    g_VSMFoot = max(min(Fl, Fd), 1e-9);

    // (y 2: the test's view - a pixel proven lit written half lit, to count them and check each against the full evaluation)
    float S;
    if (SunMaskReuse(P, FaceN, Size, S)) { if (MaskPush.Params.w != 0u) S *= 0.5; }      // (w: the test's view of the reused pixels - half)
    else S = UBOLight.ShadowLevels[0].y >= 1.0 && VSMProvenLit(P + UBOLight.CameraPos.xyz, FaceN) ? (UBOLight.ShadowLevels[0].y == 2.0 ? 0.5 : 1.0) : ShadowVSM(P + UBOLight.CameraPos.xyz, FaceN);
    OutMask = (uint(clamp(S, 0.0, 1.0) * 65535.0 + 0.5) << 16) | (packHalf2x16(vec2(length(P), 0.0)) & 0xFFFFu);
}
