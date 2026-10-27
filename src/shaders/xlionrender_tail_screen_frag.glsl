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

layout(location = 0) out vec4 OutDiffuse;
layout(location = 1) out vec4 OutSpecular;

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
