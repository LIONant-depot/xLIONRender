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

    const float S = ShadowVSM(P + UBOLight.CameraPos.xyz, FaceN);
    OutMask = (uint(clamp(S, 0.0, 1.0) * 65535.0 + 0.5) << 16) | (packHalf2x16(vec2(length(P), 0.0)) & 0xFFFFu);
}
