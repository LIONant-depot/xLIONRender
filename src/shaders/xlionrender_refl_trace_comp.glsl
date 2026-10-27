#version 450
// Reflections (documentation/reflections_strategy.md sec. 2-6, milestone B): one reflection ray per texel of the thin G-buffer (half
// size: its mapped normal, roughness, depth), a GGX visible normal (Heitz 2018), the nearest hit in the GI's BVH (opaque rigid geometry)
// and the radiance toward the receiver: D(y) = L_e + albedo / pi (E_direct + E_indirect) fresh at the hit with the GI's own model
// (xlionrender_gi_pass_common.glsl: Direct - the sun and the lights, a BVH shadow ray each -, Indirect - the surfels' read -, SurfaceAt -
// the capture's albedo and emission -, Unsupported - its fallback); a miss: the sky (the environment atlas at the receiver's roughness).
// S(y, -w), the hit's own specular, is not evaluated yet (T0 without its probe term, labelled: metals at a hit are their diffuse capture).
// Writes U = the lobe-mean radiance (one sample: L itself; w = G2/G1 kept for the reconstruction) into the records' second region
// (instance::reflection records: uvec4 a half-size texel - x: U.rg, y: U.b and conf, z: the hit distance and w, w: the view depth (half,
// bits 0-15), the octahedral normal (4 + 4 bits), the hit class (bits 24-27)). Push.m_Mode 1..3: the benchmark only (mirror, roughness
// 0.5, the materials'; nothing written). Its counters per ring slot: rays, hits, the BVH's visits
layout(local_size_x = 8, local_size_y = 8) in;

#include "xlionrender_gi_pass_common.glsl"           // (bindings 0-4: the BVH and the GI's data)
#include "xlionrender_octahedral.glsl"
#include "xlionrender_environment_atlas.glsl"

struct frame_t { mat4 C2W; mat4 Reproject; vec4 Params; };          // (xlionrender_tail_denoise_comp.glsl's: C2W to the camera-centred world)
layout(std430, set = 0, binding = 5) readonly buffer frame_b   { frame_t F[]; };
layout(std430, set = 0, binding = 6)          buffer stats_b   { uint    Stats[]; };      // 8 per ring slot: rays, hits, visits, the planar texels, their roughness x 100
layout(std430, set = 0, binding = 7)          buffer records_b { uvec4   Records[]; };    // the view's records: the tail's W x H, then these
layout(std430, set = 0, binding = 8) readonly buffer mips_b    { uvec2   PlanarMips[]; }; // the planar view's pyramid (xlionrender_refl_planar_mips_comp.glsl)
struct planar_t { mat4 C2W; vec4 Info; uvec4 Level[8]; };                                  // the planar view's clip -> its eye's centred world; x: the half view's pixels a unit at distance 1, y: the pyramid's levels; per level: its base, w, h
layout(std430, set = 0, binding = 9) readonly buffer planar_b  { planar_t PL[]; };        // (a ring slot each)
layout(set = 1, binding = 0) uniform sampler2D GColor;     // xy: the octahedral normal (x 0.5 + 0.5), z: roughness
layout(set = 1, binding = 1) uniform sampler2D GDepth;
layout(set = 1, binding = 2) uniform sampler2D EnvAtlas;   // the sky's prefiltered atlas (xlionrender_environment.glsl EnvSample)
layout(set = 1, binding = 3) uniform sampler2D Planar;     // the planar view (instance::renderPlanar), this size, x mirrored; alpha: drawn (m_bPlanar)
layout(set = 1, binding = 4) uniform sampler2D PlanarDepth;

layout(push_constant) uniform push_t { uint m_Mode, m_Slot; int m_TlasRoot, m_PrimOffset; vec3 m_Eye; uint m_Frame; float m_EnvIntensity, m_MaxRough, m_bPlanar, m_Pad1; vec4 m_Plane; } Push;

const uint refl_hit_v = 1u, refl_miss_v = 2u, refl_planar_v = 3u, refl_none_v = 0u;    // the hit class (bits 24-27): HIT_WORLD_SUPPORTED, MISS_WORLD, HIT_PLANAR_SUPPORTED, nothing traced

shared uint s_Rays, s_Hits, s_Visits, s_Planar, s_PlanarRough;

// Heitz 2018: a visible normal of an isotropic GGX (alpha a) for the view Ve (local frame, z the normal)
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

// The planar view's pyramid at uv (0..1, the planar view's), level L (0: the view itself), bilinear: rgb, the coverage
vec4 PlanarLevel(int L, vec2 uv)
{
    if (L == 0) return textureLod(Planar, uv, 0.0);
    const uvec4 Lv = PL[Push.m_Slot].Level[L];
    const vec2  q  = uv * vec2(Lv.yz) - 0.5;
    const ivec2 i0 = ivec2(floor(q));
    const vec2  f  = q - vec2(i0);
    vec4 S = vec4(0.0);
    for (int k = 0; k < 4; ++k)
    {
        const ivec2 t = clamp(i0 + ivec2(k & 1, k >> 1), ivec2(0), ivec2(Lv.yz) - 1);
        const uvec2 m = PlanarMips[Lv.x + uint(t.y) * Lv.y + uint(t.x)];
        const vec4  c = vec4(unpackHalf2x16(m.x), unpackHalf2x16(m.y));
        S += vec4(c.rgb * c.a, c.a) * (((k & 1) != 0 ? f.x : 1.0 - f.x) * ((k >> 1) != 0 ? f.y : 1.0 - f.y));
    }
    return S.a > 1e-4 ? vec4(S.rgb / S.a, S.a) : vec4(0.0);
}
vec4 PlanarAt(vec2 uv, float Lod)
{
    const float Lc = clamp(Lod, 0.0, PL[Push.m_Slot].Info.y);
    const int   L0 = int(floor(Lc));
    const vec4  A  = PlanarLevel(L0, uv);
    if (Lc - float(L0) < 1e-3 || float(L0) >= PL[Push.m_Slot].Info.y) return A;
    const vec4  B  = PlanarLevel(L0 + 1, uv);
    return mix(A, B, Lc - float(L0));
}

void main()
{
    if (gl_LocalInvocationIndex == 0u) { s_Rays = 0u; s_Hits = 0u; s_Visits = 0u; s_Planar = 0u; s_PlanarRough = 0u; }
    barrier();
    const ivec2 Size = ivec2(F[Push.m_Slot].Params.xy);
    const ivec2 p    = ivec2(gl_GlobalInvocationID.xy);
    const bool  bIn  = p.x < Size.x && p.y < Size.y;
    const uint  i    = uint(Size.x * Size.y) + uint(p.y * Size.x + p.x);
    const float z    = bIn ? texelFetch(GDepth, p, 0).r : 1.0;
    uvec4 Rec = uvec4(0u);
    if (z < 1.0)
    {
        g_GIBvhTlasRoot   = Push.m_TlasRoot;
        g_GIBvhPrimOffset = Push.m_PrimOffset;
        const vec2  Ndc = (vec2(p) + 0.5) / vec2(Size) * 2.0 - 1.0;
        const vec4  Pc  = F[Push.m_Slot].C2W * vec4(Ndc, z, 1.0);
        const vec3  P   = Pc.xyz / Pc.w;                                    // camera-centred
        const vec4  G   = texelFetch(GColor, p, 0);
        const vec3  N   = OctDecode(G.xy * 2.0 - 1.0);
        const float Rough = Push.m_Mode == 1u ? 0.0 : Push.m_Mode == 2u ? 0.5 : G.z;
        const vec2  OctN = OctEncode(N) * 0.5 + 0.5;
        Rec.w = packHalf2x16(vec2(Pc.w > 0.0 ? 1.0 / Pc.w : 0.0, 0.0)) | (uint(OctN.x * 15.0 + 0.5) << 16) | (uint(OctN.y * 15.0 + 0.5) << 20);    // (the view depth: clip w, as the tail's)
        const float PlaneDist = dot(Push.m_Plane.xyz, P + Push.m_Eye) + Push.m_Plane.w;
        if (Push.m_Mode == 0u && Push.m_bPlanar > 0.5 && Rough < Push.m_MaxRough && abs(PlaneDist) < 0.01 + 0.002 * length(P) && dot(N, Push.m_Plane.xyz) > 0.95)
        {
            // a planar receiver: the planar view at the mirrored texel (a point of the plane lands there). Rough: its pyramid at the level
            // of the lobe's footprint - the reflected distance t (the planar depth) times alpha, in the view's pixels at the receiver's
            // distance (a labelled approximation of the rough planar class: an ideal image filtered, it cannot show what the mirror misses).
            // Nothing drawn there: the probes (the coverage is the confidence)
            const ivec2 m    = ivec2(Size.x - 1 - p.x, p.y);
            const vec2  uv   = (vec2(m) + 0.5) / vec2(Size);
            const float zp   = texelFetch(PlanarDepth, m, 0).r;
            float Lod = 0.0, t = 0.0;
            if (zp < 1.0)
            {
                const vec4 Q = PL[Push.m_Slot].C2W * vec4(uv * 2.0 - 1.0, zp, 1.0);
                t = max(length(Q.xyz / Q.w) - length(P), 0.0);
                const float a  = max(Rough, 0.045) * max(Rough, 0.045);
                Lod = log2(max(2.0 * t * a / max(length(P), 1e-3) * PL[Push.m_Slot].Info.x, 1.0));
            }
            const vec4 C = PlanarAt(uv, Lod);
            Rec.x = packHalf2x16(C.rg);
            Rec.y = packHalf2x16(vec2(C.b, clamp(C.a, 0.0, 1.0)));
            Rec.z = packHalf2x16(vec2(t, 1.0));
            Rec.w |= refl_planar_v << 24;
            atomicAdd(s_Planar, 1u);
            atomicAdd(s_PlanarRough, uint(Rough * 100.0 + 0.5));
        }
        else if (Push.m_Mode != 0u || Rough < Push.m_MaxRough)
        {
            const vec3  V   = normalize(-P);
            const vec3  T   = normalize(abs(N.z) < 0.999 ? cross(vec3(0.0, 0.0, 1.0), N) : cross(vec3(1.0, 0.0, 0.0), N));
            const vec3  B   = cross(N, T);
            const vec3  Ve  = vec3(dot(V, T), dot(V, B), max(dot(V, N), 1e-4));
            const uint  h   = Hash32(uint(p.x) * 73856093u ^ uint(p.y) * 19349663u ^ Push.m_Frame * 83492791u);
            const vec2  u   = vec2(float(h & 0xFFFFu), float(h >> 16)) / 65536.0;
            const float a   = max(Rough, 0.045) * max(Rough, 0.045);
            const vec3  H   = Push.m_Mode == 1u ? vec3(0.0, 0.0, 1.0) : SampleVNDF(Ve, a, u);
            const vec3  Lt  = reflect(-Ve, H);
            if (Lt.z > 0.0)
            {
                const vec3  L = normalize(Lt.x * T + Lt.y * B + Lt.z * N);
                const float w = (1.0 + Lambda(Ve.z, a)) / (1.0 + Lambda(Ve.z, a) + Lambda(Lt.z, a));     // G2 / G1
                const vec3  O = P + Push.m_Eye + N * (0.002 * length(P) + 0.001);
                g_GIBvhSeeThrough = true;                   // (the GI's rays: see-through triangles let them pass)
                const gi_hit Hit = GICast(O, L, 1e30);
                atomicAdd(s_Rays, 1u);
                atomicAdd(s_Visits, uint(g_GIBvhVisits));
                if (Push.m_Mode == 0u)
                {
                    vec3 Lr; uint Class; float Dist = 0.0;
                    if (Hit.m_Inst >= 0)
                    {
                        atomicAdd(s_Hits, 1u);
                        const uint  R = InstRec(Hit.m_Inst), M = MeshOf(R);
                        const uvec3 Tr = MTri(M, Hit.m_Tri);
                        const float b0 = 1.0 - Hit.m_U - Hit.m_V;
                        const vec3  Py = ToWorld(R, MPos(M, Tr.x) * b0 + MPos(M, Tr.y) * Hit.m_U + MPos(M, Tr.z) * Hit.m_V);
                        vec3 Ny = NormalToWorld(R, MNrm(M, Tr.x) * b0 + MNrm(M, Tr.y) * Hit.m_U + MNrm(M, Tr.z) * Hit.m_V);
                        if (dot(Ny, L) > 0.0) Ny = -Ny;
                        vec3 Albedo, Emission;
                        SurfaceAt(Hit.m_Inst, Hit.m_Tri, Hit.m_U, Hit.m_V, Albedo, Emission);
                        vec3 Ei;
                        if (!Indirect(Hit.m_Inst, Hit.m_Tri, Hit.m_U, Hit.m_V, Ei)) Ei = Unsupported(Hit.m_Inst, Hit.m_Tri, Hit.m_U, Hit.m_V);
                        vec3 Ed = Direct(Py, Ny);
                        if (EmitterDirectExternal()) Ed += EmitterIrradianceAt(Py, Ny, h);
                        Lr = Emission / max(GF(5u), 1e-6) + Albedo * (1.0 / gi_pi_v) * (Ed + Ei);     // (the emission as the surface shows it: SurfaceAt's carries the GI's emissive scale)
                        Class = refl_hit_v; Dist = Hit.m_T;
                    }
                    else
                    {
                        Lr = AtlasSample(EnvAtlas, vec2(0.0), kEnvAtlas, L, Rough) * Push.m_EnvIntensity;
                        Class = refl_miss_v;
                    }
                    const float Conf = 1.0 - smoothstep(0.5 * Push.m_MaxRough, Push.m_MaxRough, Rough);     // (B: sharp receivers; glossy ones fade to the probes until the reconstruction)
                    Rec.x = packHalf2x16(Lr.rg);
                    Rec.y = packHalf2x16(vec2(Lr.b, Conf));
                    Rec.z = packHalf2x16(vec2(Dist, w));
                    Rec.w |= Class << 24;
                }
                else if (Hit.m_Inst >= 0) atomicAdd(s_Hits, 1u);
            }
        }
    }
    if (bIn && Push.m_Mode == 0u) Records[i] = Rec;
    barrier();
    if (gl_LocalInvocationIndex == 0u)
    {
        atomicAdd(Stats[Push.m_Slot * 8u + 0u], s_Rays);
        atomicAdd(Stats[Push.m_Slot * 8u + 1u], s_Hits);
        atomicAdd(Stats[Push.m_Slot * 8u + 2u], s_Visits);
        atomicAdd(Stats[Push.m_Slot * 8u + 3u], s_Planar);
        atomicAdd(Stats[Push.m_Slot * 8u + 4u], s_PlanarRough);
    }
}
