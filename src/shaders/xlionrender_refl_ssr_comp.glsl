#version 450
// Reflections, the screen trace (SSR; documentation/reflections_strategy.md): what is on screen - a character, the spheres, anything the
// probes' boxes flatten onto their walls - reflected where it really is. One ray a texel of the thin G-buffer (half size), the mirror
// direction or a GGX visible normal (Heitz 2018) on rough receivers, marched in screen space against that G-buffer's depth (the ray's
// NDC depth is affine along its projected line: exact at every step) up to m_Steps steps, then a binary refinement; a hit within the
// thickness (a surface, not a gap behind one) that faces the ray takes last frame's lit image (render_view::m_HDR, reprojected by the
// camera's move: the scene pass has not drawn this frame yet) - Unreal's SSR / Lumen's screen traces read the previous frame the same way.
// Its confidence fades at the screen's borders and toward the ray's length; a miss is coverage 0 (class 4: the reconstruction averages
// it with the hits around it into a fraction - the probes show by the rest). Writes the
// trace's records (xlionrender_refl_trace_comp.glsl's format, class 4) into the records' second region, where the BVH trace did not write
// this frame (m_bKeepTraced); the reconstruction (xlionrender_refl_denoise_comp.glsl) and the materials (ReflScreen) read them as the trace's
layout(local_size_x = 8, local_size_y = 8) in;

#include "xlionrender_octahedral.glsl"

struct frame_t { mat4 C2W; mat4 ToPrev; vec4 Params; };       // (the reconstruction's frame: clip -> camera-centred world, that -> last frame's clip; size, a last frame)
layout(std430, set = 0, binding = 0) readonly buffer frame_b   { frame_t F[]; };
layout(std430, set = 0, binding = 1)          buffer records_b { uvec4   Records[]; };      // the tail's W x H, then the reflections' W x H
layout(set = 1, binding = 0) uniform sampler2D GColor;     // xy: the octahedral normal (x 0.5 + 0.5), z: roughness
layout(set = 1, binding = 1) uniform sampler2D GDepth;
layout(set = 1, binding = 2) uniform sampler2D SceneHDR;   // last frame's lit image, full size

// m_MaxRough: the receivers up to which (1: all); m_MaxDist: the ray's length (world units); m_Thickness: a hit's depth behind the surface
// it passes, world units (plus 2% of its distance)
layout(push_constant) uniform push_t { uint m_Slot, m_bKeepTraced, m_Frame, m_Steps; float m_MaxDist, m_Thickness, m_MaxRough, m_Pad; } Push;

const uint refl_screen_v = 4u;      // the hit class (bits 24-27): HIT_SCREEN

uint Hash(uint x) { x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x; }

// (xlionrender_refl_trace_comp.glsl's)
vec3 SampleVNDF(vec3 Ve, float a, vec2 u)
{
    const vec3  Vh = normalize(vec3(a * Ve.xy, Ve.z));
    const float L2 = dot(Vh.xy, Vh.xy);
    const vec3  T1 = L2 > 0.0 ? vec3(-Vh.y, Vh.x, 0.0) * inversesqrt(L2) : vec3(1.0, 0.0, 0.0);
    const vec3  T2 = cross(Vh, T1);
    const float r = sqrt(u.x), phi = 6.28318530718 * u.y;
    const float t1 = r * cos(phi), s = 0.5 * (1.0 + Vh.z);
    const float t2 = (1.0 - s) * sqrt(max(0.0, 1.0 - t1 * t1)) + s * r * sin(phi);
    const vec3  Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * Vh;
    return normalize(vec3(a * Nh.xy, max(1e-6, Nh.z)));
}
float Lambda(float c, float a) { c = max(c, 1e-6); return 0.5 * (sqrt(1.0 + a * a * (1.0 - c * c) / (c * c)) - 1.0); }

vec3 Unproject(vec2 Ndc, float z) { const vec4 P = F[Push.m_Slot].C2W * vec4(Ndc, z, 1.0); return P.xyz / P.w; }

void main()
{
    const ivec2 Size = ivec2(F[Push.m_Slot].Params.xy);
    const ivec2 p    = ivec2(gl_GlobalInvocationID.xy);
    if (p.x >= Size.x || p.y >= Size.y) return;
    const uint i = uint(Size.x * Size.y) + uint(p.y * Size.x + p.x);
    if (Push.m_bKeepTraced != 0u && ((Records[i].w >> 24) & 15u) != 0u) return;       // (the BVH trace's record of this frame stays)
    const float z = texelFetch(GDepth, p, 0).r;
    uvec4 Rec = uvec4(0u);
    if (z < 1.0)
    {
        const vec2  Ndc = (vec2(p) + 0.5) / vec2(Size) * 2.0 - 1.0;
        const vec4  Pc  = F[Push.m_Slot].C2W * vec4(Ndc, z, 1.0);
        const vec3  P   = Pc.xyz / Pc.w;                                    // camera-centred
        const vec4  G   = texelFetch(GColor, p, 0);
        const vec3  N   = OctDecode(G.xy * 2.0 - 1.0);
        const float Rough = G.z;
        const vec2  OctN = OctEncode(N) * 0.5 + 0.5;
        Rec.w = packHalf2x16(vec2(Pc.w > 0.0 ? 1.0 / Pc.w : 0.0, 0.0)) | (uint(OctN.x * 15.0 + 0.5) << 16) | (uint(OctN.y * 15.0 + 0.5) << 20);
        const vec3  V  = normalize(-P);
        const vec3  T  = normalize(abs(N.z) < 0.999 ? cross(vec3(0.0, 0.0, 1.0), N) : cross(vec3(1.0, 0.0, 0.0), N));
        const vec3  B  = cross(N, T);
        const vec3  Ve = vec3(dot(V, T), dot(V, B), max(dot(V, N), 1e-4));
        // (a mirror's ray the same every frame: its start's jitter changing made its hits at silhouettes toggle - flicker; a rough one's
        // new every frame, for the reconstruction to average)
        const uint  h  = Hash(uint(p.x) * 73856093u ^ uint(p.y) * 19349663u ^ (G.z < 0.05 ? 0u : Push.m_Frame * 83492791u));
        const float a  = max(Rough, 0.045) * max(Rough, 0.045);
        const vec3  Hm = Rough < 0.05 ? vec3(0.0, 0.0, 1.0) : SampleVNDF(Ve, a, vec2(float(h & 0xFFFFu), float(h >> 16)) / 65536.0);
        const vec3  Lt = reflect(-Ve, Hm);
        if (Rough <= Push.m_MaxRough) Rec.w |= refl_screen_v << 24;      // (traced: a miss is coverage 0)
        if (Rough <= Push.m_MaxRough && Lt.z > 0.0)
        {
            const vec3  L = normalize(Lt.x * T + Lt.y * B + Lt.z * N);
            const float w = (1.0 + Lambda(Ve.z, a)) / (1.0 + Lambda(Ve.z, a) + Lambda(Lt.z, a));     // G2 / G1
            const vec3  O = P + N * (0.002 * length(P) + 0.001);
            const mat4  W2C = inverse(F[Push.m_Slot].C2W);
            vec4 h0 = W2C * vec4(O, 1.0), h1 = W2C * vec4(O + L * Push.m_MaxDist, 1.0);
            const float WNear = 0.05 * h0.w;                                   // (a ray toward the camera: cut before the eye)
            if (h1.w < WNear) h1 = mix(h0, h1, (h0.w - WNear) / (h0.w - h1.w));
            const vec3 s0 = h0.xyz / h0.w;
            vec3 s1 = h1.xyz / h1.w;
            // the projected line cut to the screen (and to the depth range)
            float tEnd = 1.0;
            const vec3 d = s1 - s0;
            for (int k = 0; k < 2; ++k) if (abs(d[k]) > 1e-6) tEnd = min(tEnd, ((d[k] > 0.0 ? 1.0 : -1.0) - s0[k]) / d[k]);
            if (abs(d.z) > 1e-9) tEnd = min(tEnd, ((d.z > 0.0 ? 1.0 : 0.0) - s0.z) / d.z);
            s1 = s0 + d * max(tEnd, 0.0);
            const vec2  Pix = abs(s1.xy - s0.xy) * 0.5 * vec2(Size);
            const int   n   = int(clamp(max(Pix.x, Pix.y), 1.0, float(Push.m_Steps)));
            const float j   = float(Hash(h) & 0xFFFFu) / 65536.0;
            float tIn = 0.0, tOut = -1.0;
            for (int k = 1; k <= n; ++k)
            {
                const float t = (float(k) - 0.5 + 0.5 * j) / float(n);
                const vec3  s = mix(s0, s1, t);
                const ivec2 q = clamp(ivec2((s.xy * 0.5 + 0.5) * vec2(Size)), ivec2(0), Size - 1);
                if (s.z > texelFetch(GDepth, q, 0).r) { tOut = t; break; }    // behind the depth there
                tIn = t;
            }
            if (tOut > 0.0)
            {
                for (int k = 0; k < 5; ++k)          // the crossing, refined
                {
                    const float t = 0.5 * (tIn + tOut);
                    const vec3  s = mix(s0, s1, t);
                    const ivec2 q = clamp(ivec2((s.xy * 0.5 + 0.5) * vec2(Size)), ivec2(0), Size - 1);
                    if (s.z > texelFetch(GDepth, q, 0).r) tOut = t; else tIn = t;
                }
                const vec3  s  = mix(s0, s1, tOut);
                const ivec2 q  = clamp(ivec2((s.xy * 0.5 + 0.5) * vec2(Size)), ivec2(0), Size - 1);
                const float zs = texelFetch(GDepth, q, 0).r;
                const vec3  Ph = Unproject(s.xy, zs), Pr = Unproject(s.xy, s.z);
                const vec3  Nh = OctDecode(texelFetch(GColor, q, 0).xy * 2.0 - 1.0);
                const float Dist = length(Ph - O);
                // a surface (the ray within the thickness behind it), not the sky, facing the ray. A ray toward the camera meets the back
                // of what it passes (a sphere nearer the eye: never on screen): its front's light stands for it, over more depth
                // (blended over the ray's angle to the view, not switched: on a sphere the switch's line - the ray across the view -
                // is a great circle, a straight line through its image: a wedge with hard edges in a mirror, Tomas)
                const float Toward = smoothstep(-0.3, 0.3, -dot(L, P) / max(length(P), 1e-6));
                const float Thick  = mix(Push.m_Thickness + 0.02 * length(Ph), 4.0 * Push.m_Thickness + 0.1 * length(Ph), Toward);
                if (zs < 1.0 && length(Pr) - length(Ph) <= Thick && dot(Nh, L) < 0.2 + 1.0 * Toward)
                {
                    const vec4 Pp  = F[Push.m_Slot].ToPrev * vec4(Ph, 1.0);
                    const vec2 uv  = s.xy * 0.5 + 0.5;
                    const vec2 uvp = F[Push.m_Slot].Params.z > 0.5 && Pp.w > 0.0 ? Pp.xy / Pp.w * 0.5 + 0.5 : uv;
                    const vec2 e   = min(min(uv, 1.0 - uv), min(uvp, 1.0 - uvp));
                    const float Conf = smoothstep(0.0, 0.2, min(e.x, e.y)) * (1.0 - smoothstep(0.7, 1.0, Dist / Push.m_MaxDist));
                    vec3 C = textureLod(SceneHDR, clamp(uvp, vec2(0.0), vec2(1.0)), 0.0).rgb;
                    C = any(isnan(C)) || any(isinf(C)) ? vec3(0.0) : min(C, vec3(64.0));     // (fireflies: the denoiser clamps the rest)
                    if (Conf > 0.0)
                    {
                        Rec.x = packHalf2x16(C.rg);
                        Rec.y = packHalf2x16(vec2(C.b, Conf));
                        Rec.z = packHalf2x16(vec2(Dist, w));
                    }
                }
            }
        }
    }
    Records[i] = Rec;
}
