#version 450
// The screen tail's denoiser (documentation/lighting_p3_design.md "The decision" 3; MegaLights' shape, our code): at half size, over the
// tail pass's demodulated diffuse and specular (xlionrender_tail_screen_frag.glsl) and the thin G-buffer. Two buffers of a record a
// texel (uvec4: x the diffuse's rg, y its b and the specular's r, z the specular's gb - halves -, w the surface's distance to the eye (half;
// 0: no surface), its octahedral normal (4 + 4 bits), the diffuse's and the specular's frames (4 + 4 bits)):
//   pass 0, temporal: Result (last frame's) -> Temporal. Last frame's records reprojected by the camera (the texel's world position into
//            last frame's clip space), its 2 x 2 texels kept where they were this surface (distance to last frame's eye within 3%, normals
//            within ~35 degrees), clamped to this frame's 3 x 3 neighbourhood (mean +- 1.25 standard deviations), accumulated up to 12
//            frames (the specular fewer on glossy surfaces: its lobe moves with the view).
//   pass 1, spatial: Temporal -> Result. The texel and 4 neighbours a few texels away (wider where the history is short - a disocclusion),
//            weighed by distance, normal and, for the specular, roughness. Result is what the materials read (binding 14:
//            xlionrender_local_lights.glsl LocalTailScreen) and next frame's history (the filtered result fed back, as SVGF's first level)
layout(local_size_x = 8, local_size_y = 8) in;

#include "xlionrender_octahedral.glsl"

struct frame_t
{
    mat4 C2W;               // this frame's clip -> camera-centred world
    mat4 Reproject;         // this frame's clip -> last frame's clip (the eye's move between)
    vec4 Move;              // xyz: this frame's eye - last frame's (a point's distance to last frame's eye: length(P + Move))
    vec4 Params;            // x, y: size; z: 1 there is a last frame; w: the frame's number (the spatial taps' rotation)
};
layout(std430, set = 0, binding = 0) readonly buffer frame_b   { frame_t F[]; };
layout(std430, set = 0, binding = 1) buffer          result_b  { uvec4 Result[]; };
layout(std430, set = 0, binding = 2) buffer          temporal_b{ uvec4 Temporal[]; };
layout(set = 1, binding = 0) uniform sampler2D TailDiffuse;
layout(set = 1, binding = 1) uniform sampler2D TailSpecular;
layout(set = 1, binding = 2) uniform sampler2D GColor;          // xy: the octahedral normal (x 0.5 + 0.5), z: roughness
layout(set = 1, binding = 3) uniform sampler2D GDepth;
layout(push_constant) uniform push_t { uint m_Pass, m_Slot, m_Pad0, m_Pad1, m_Pad2, m_Pad3, m_Pad4, m_Pad5; } Push;

const float kMaxFrames = 12.0;

ivec2 Size() { return ivec2(F[Push.m_Slot].Params.xy); }
vec3  WorldAt(ivec2 p, float z) { const vec2 Ndc = (vec2(p) + 0.5) / vec2(Size()) * 2.0 - 1.0; const vec4 W = F[Push.m_Slot].C2W * vec4(Ndc, z, 1.0); return W.xyz / W.w; }
vec3  Nrm(vec4 G) { return OctDecode(G.xy * 2.0 - 1.0); }

// The record's last word: the distance, the normal at 4 + 4 bits, the frames
uint  Meta(float Dist, vec3 N, float nD, float nS)
{
    const uvec2 o = uvec2(clamp(OctEncode(N) * 0.5 + 0.5, 0.0, 1.0) * 15.0 + 0.5);
    return (packHalf2x16(vec2(Dist, 0.0)) & 0xFFFFu) | (o.x << 16) | (o.y << 20) | (uint(nD + 0.5) << 24) | (uint(nS + 0.5) << 28);
}
float MetaDist(uint w)   { return unpackHalf2x16(w & 0xFFFFu).x; }
vec3  MetaNormal(uint w) { return OctDecode(vec2(float((w >> 16) & 15u), float((w >> 20) & 15u)) / 15.0 * 2.0 - 1.0); }
vec3  RecD(uvec4 R) { return vec3(unpackHalf2x16(R.x), unpackHalf2x16(R.y).x); }
vec3  RecS(uvec4 R) { return vec3(unpackHalf2x16(R.y).y, unpackHalf2x16(R.z)); }
uvec4 Rec(vec3 D, vec3 S, uint w) { return uvec4(packHalf2x16(D.rg), packHalf2x16(vec2(D.b, S.r)), packHalf2x16(S.gb), w); }

void TemporalPass(ivec2 p)
{
    const uint  i = uint(p.y * Size().x + p.x);
    const float z = texelFetch(GDepth, p, 0).r;
    if (z >= 1.0) { Temporal[i] = uvec4(0u); return; }
    const vec4  G    = texelFetch(GColor, p, 0);
    const vec3  N    = Nrm(G);
    const vec3  P    = WorldAt(p, z);
    const float Dist = length(P);
    const vec3  D = texelFetch(TailDiffuse, p, 0).rgb, S = texelFetch(TailSpecular, p, 0).rgb;

    // this frame's neighbourhood (the same surface's texels): mean and deviation, the history's clamp
    vec3 m1D = vec3(0.0), m2D = vec3(0.0), m1S = vec3(0.0), m2S = vec3(0.0);
    float k = 0.0;
    for (int y = -1; y <= 1; ++y)
    for (int x = -1; x <= 1; ++x)
    {
        const ivec2 q = clamp(p + ivec2(x, y), ivec2(0), Size() - 1);
        const float zq = texelFetch(GDepth, q, 0).r;
        if (zq >= 1.0 || abs(length(WorldAt(q, zq)) - Dist) > 0.05 * Dist) continue;
        const vec3 d = texelFetch(TailDiffuse, q, 0).rgb, s = texelFetch(TailSpecular, q, 0).rgb;
        m1D += d; m2D += d * d; m1S += s; m2S += s * s; k += 1.0;
    }
    m1D /= k; m1S /= k;
    const vec3 sdD = sqrt(max(m2D / k - m1D * m1D, 0.0)), sdS = sqrt(max(m2S / k - m1S * m1S, 0.0));

    // last frame's: this texel's world position into its clip space, the 2 x 2 texels around, bilinear, where they were this surface
    vec3 HD = vec3(0.0), HS = vec3(0.0);
    float nD = 0.0, nS = 0.0;
    if (F[Push.m_Slot].Params.z > 0.5)
    {
        const vec2  Ndc = (vec2(p) + 0.5) / vec2(Size()) * 2.0 - 1.0;
        const vec4  C   = F[Push.m_Slot].Reproject * vec4(Ndc, z, 1.0);
        const vec2  hp  = (C.xy / C.w * 0.5 + 0.5) * vec2(Size()) - 0.5;
        const vec2  i0  = floor(hp), f = hp - i0;
        const float dP  = length(P + F[Push.m_Slot].Move.xyz);                   // (this point's distance to last frame's eye)
        float Wsum = 0.0, Nd = 0.0, Ns = 0.0;
        for (int t = 0; t < 4; ++t)
        {
            const ivec2 q = ivec2(i0) + ivec2(t & 1, t >> 1);
            if (any(lessThan(q, ivec2(0))) || any(greaterThanEqual(q, Size()))) continue;
            const uvec4 R  = Result[uint(q.y * Size().x + q.x)];
            const float dq = MetaDist(R.w);
            if (dq <= 0.0 || abs(dq - dP) > 0.03 * dP + 0.02) continue;
            if (dot(MetaNormal(R.w), N) < 0.8) continue;
            const float w = ((t & 1) != 0 ? f.x : 1.0 - f.x) * ((t >> 1) != 0 ? f.y : 1.0 - f.y);
            HD += w * RecD(R); HS += w * RecS(R);
            Nd += w * float((R.w >> 24) & 15u); Ns += w * float(R.w >> 28);
            Wsum += w;
        }
        if (Wsum > 0.25) { HD /= Wsum; HS /= Wsum; nD = Nd / Wsum; nS = Ns / Wsum; }
    }
    HD = clamp(HD, m1D - 1.25 * sdD, m1D + 1.25 * sdD);
    HS = clamp(HS, m1S - 1.25 * sdS, m1S + 1.25 * sdS);
    const float MaxS = mix(4.0, kMaxFrames, clamp(G.z * 2.0, 0.0, 1.0));      // (glossy: a shorter specular history)
    nD = min(floor(nD + 0.5) + 1.0, kMaxFrames); nS = min(floor(nS + 0.5) + 1.0, MaxS);
    Temporal[i] = Rec(mix(HD, D, 1.0 / nD), mix(HS, S, 1.0 / nS), Meta(Dist, N, nD, nS));
}

void SpatialPass(ivec2 p)
{
    const uint  i = uint(p.y * Size().x + p.x);
    const uvec4 A = Temporal[i];
    const float Dist = MetaDist(A.w);
    if (Dist <= 0.0) { Result[i] = uvec4(0u); return; }
    const vec4  G = texelFetch(GColor, p, 0);
    const vec3  N = Nrm(G);
    const float n = float((A.w >> 24) & 15u);
    vec3  SD = RecD(A), SS = RecS(A);
    float WD = 1.0, WS = 1.0;
    // 4 taps on a rotated square, its radius 8 texels with one frame of history down to 1 with the full history
    const float Rad = mix(8.0, 1.0, clamp((n - 1.0) / (kMaxFrames - 1.0), 0.0, 1.0));
    const float a = F[Push.m_Slot].Params.w * 2.39996;
    const vec2  e0 = vec2(cos(a), sin(a)), e1 = vec2(-e0.y, e0.x);
    for (int t = 0; t < 4; ++t)
    {
        const vec2  o = ((t & 1) != 0 ? e0 : -e0) + ((t & 2) != 0 ? e1 : -e1);
        const ivec2 q = clamp(p + ivec2(round(o * Rad * 0.7071)), ivec2(0), Size() - 1);
        const uvec4 B = Temporal[uint(q.y * Size().x + q.x)];
        const float dq = MetaDist(B.w);
        if (dq <= 0.0) continue;
        const vec4  Gq = texelFetch(GColor, q, 0);
        const float wz = max(0.0, 1.0 - abs(dq - Dist) / (0.03 * Dist + 0.01));
        const float wn = pow(max(dot(Nrm(Gq), N), 0.0), 16.0);
        const float wr = max(0.0, 1.0 - abs(Gq.z - G.z) * 4.0);                 // (the specular: alike roughness only)
        const float w  = wz * wn;
        SD += w * RecD(B); WD += w;
        SS += w * wr * RecS(B); WS += w * wr;
    }
    Result[i] = Rec(SD / WD, SS / WS, A.w);
}

void main()
{
    const ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(p, Size()))) return;
    if (Push.m_Pass == 0u) TemporalPass(p); else SpatialPass(p);
}
