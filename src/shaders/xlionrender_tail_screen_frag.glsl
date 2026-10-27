#version 450
// The screen tail's pass (documentation/lighting_p3_design.md "The decision" 2): a full-screen pass at half size over the thin G-buffer
// (the materials' G-buffer variants: the octahedral normal and roughness, and its depth). Per texel its camera-centred position from the
// depth, the cell's tail resampled (xlionrender_local_lights.glsl LocalTailDemod: the head left out, the candidates' weights with their
// one-tap visibility) and shaded demodulated - the diffuse with albedo 1, the specular with F = 1 - into two targets the denoiser reads.
// The materials' pipeline layout (set 0 the shadow atlas + the G-buffer in the material slots, set 1 the lights and their cells, set 2 the
// lighting block): the lights' code is the materials', not a copy. No surface: zeros (the denoiser keeps none of it)
#define XMAT_VARYING_IN
#define XLION_NO_GI_PIXEL
#define XLION_TAIL_SCREEN
#include "mb_varying_definition_full.glsl"
VaryingFull In;                                     // (wSpacePosition: the texel's camera-centred position, set below)
#include "mb_standard_pbr.frag"

layout(binding = 1) uniform sampler2D GBufferColor;     // xy: the octahedral normal (x 0.5 + 0.5), z: the roughness (the materials' G-buffer variants)
layout(binding = 2) uniform sampler2D GBufferDepth;

// P4: the GI's BVH (gi::gpu_trace::getBvh; shaders/xlionrender_gi_bvh.glsl's layouts), absolute world space; the GI dummy where it has none
struct GINode     { vec4 m_B[6]; ivec4 m_Child; };
struct GITri4     { vec4 m_V0[3]; vec4 m_E1[3]; vec4 m_E2[3]; ivec4 m_Tri; };
struct GIInstance { vec4 m_W2L[3]; ivec4 m_Info; };
layout(std430, set = 1, binding = 15) readonly buffer nodes_b     { GINode     GIBvhNodes[]; };
layout(std430, set = 1, binding = 16) readonly buffer tri4_b      { GITri4     GIBvhTri4[]; };
layout(std430, set = 1, binding = 17) readonly buffer instances_b { GIInstance GIBvhInstances[]; };
layout(std430, set = 1, binding = 18) readonly buffer tlas_b      { int        GIBvhTlasLeaves[]; };
#include "xlionrender_gi_bvh.glsl"

layout(location = 0) out vec4 OutDiffuse;
layout(location = 1) out vec4 OutSpecular;

// The screen trace (P3's "tail contact-shadowed"): from the point toward the light, at most TailTrace.x world units (or the light, if
// nearer), TailTrace.y steps jittered per texel and frame, each projected into this half-size depth: blocked where the depth there is in
// front of the ray by more than a small bias and less than the thickness (TailTrace.z of the view depth: a thin object does not shadow
// what is far behind it). What the screen does not see (off screen, behind the first surface) is not traced: lit
float TailContactShadow(in const vec3 P, in const vec3 ToLight)
{
    const float Len  = length(ToLight);
    const float Dist = min(UBOLight.TailTrace.x, Len * 0.95);
    if (Dist <= 0.0) return 1.0;
    const vec3  Dir   = ToLight / Len;
    const int   Steps = max(int(UBOLight.TailTrace.y), 1);
    const vec2  Size  = vec2(UBOLight.TailParams.y, UBOLight.TailParams.z);
    const float j     = LocalRandom(7u);
    const vec3  P0    = P + g_LocalFaceN * (0.02 + 0.01 * length(P));          // (off the surface: a small bias)
    for (int s = 0; s < Steps; ++s)
    {
        const float t  = Dist * (float(s) + j) / float(Steps);
        const vec4  C  = UBOLight.TailW2C * vec4(P0 + Dir * t, 1.0);
        if (C.w <= 1e-4) return 1.0;
        const vec3  Ndc = C.xyz / C.w;
        if (any(greaterThan(abs(Ndc.xy), vec2(1.0)))) return 1.0;               // (off screen: not traced)
        const ivec2 q  = ivec2((Ndc.xy * 0.5 + 0.5) * Size);
        const float zq = texelFetch(GBufferDepth, q, 0).r;
        if (zq >= 1.0) continue;
        const vec4  Wq = UBOLight.TailC2W * vec4((vec2(q) + 0.5) / Size * 2.0 - 1.0, zq, 1.0);
        const float dRay = C.w, dScene = 1.0 / Wq.w;                            // (view depths: clip w)
        if (dRay - dScene > 0.002 * dRay && dRay - dScene < UBOLight.TailTrace.z * dRay) return 0.0;
    }
    return 1.0;
}

// The shadow ray through the GI's BVH: any hit before the light (stopped short of it: the fixture around a light is not its blocker), see-
// through triangles blocking as gi::world::Occluded does
float TailBvhShadow(in const vec3 P, in const vec3 ToLight)
{
    const float Len = length(ToLight);
    const float TMax = Len * 0.95 - 0.02;
    if (TMax <= 0.0) return 1.0;
    g_GIBvhTlasRoot = floatBitsToInt(UBOLight.TailBvh.y); g_GIBvhPrimOffset = floatBitsToInt(UBOLight.TailBvh.z);
    g_GIBvhAnyHit = true; g_GIBvhSeeThrough = false;
    const vec3 O = P + UBOLight.CameraPos.xyz + g_LocalFaceN * (0.02 + 0.01 * length(P));      // (off the surface: as the screen trace)
    return GICast(O, ToLight / Len, TMax).m_Inst >= 0 ? 0.0 : 1.0;
}

void main()
{
    const ivec2 p = ivec2(gl_FragCoord.xy);
    const vec2  Size = vec2(UBOLight.TailParams.y, UBOLight.TailParams.z);
    const float z = texelFetch(GBufferDepth, p, 0).r;
    const vec2  Ndc = (vec2(p) + 0.5) / Size * 2.0 - 1.0;
    const vec4  W = UBOLight.TailC2W * vec4(Ndc, z, 1.0);
    const vec3  P = W.xyz / W.w;
    In.wSpacePosition = vec4(P, 1.0);

    // the triangle's own plane and the footprint (the local shadows' normal offset and level): the derivatives, in uniform control
    // flow; the footprint of the full-size pixel (half this texel's), whose level the request pass asked pages for
    const vec4 G  = texelFetch(GBufferColor, p, 0);
    const vec3 N  = OctDecode(G.xy * 2.0 - 1.0);
    vec3 FaceN = cross(dFdx(P), dFdy(P));
    FaceN = dot(FaceN, FaceN) > 1e-20 ? normalize(FaceN) * sign(dot(FaceN, N) + 1e-9) : N;
    if (dot(FaceN, N) < 0.5) FaceN = N;                                         // (a texel quad across an edge: its derivatives mix two surfaces)
    g_LocalFaceN = FaceN;
    g_LocalFoot  = 0.5 * max(length(dFdx(P)), length(dFdy(P)));

    OutDiffuse = vec4(0.0); OutSpecular = vec4(0.0);
    if (z >= 1.0) return;                                                       // (no surface)
    const vec3  V     = normalize(-P);
    const float NdotV = saturate(dot(N, V));
    vec3 D, S;
    LocalTailDemod(N, V, NdotV, max(G.z, MIN_ROUGHNESS), D, S);
    OutDiffuse  = vec4(D, 1.0);
    OutSpecular = vec4(S, 1.0);
}
