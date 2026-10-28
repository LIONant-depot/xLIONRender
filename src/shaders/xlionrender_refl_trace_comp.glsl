#version 450
// Reflections (documentation/reflections_strategy.md sec. 2-6, milestone B): one reflection ray per texel of the thin G-buffer (half
// size: its mapped normal, roughness, depth), a GGX visible normal (Heitz 2018), the nearest hit in the GI's BVH (opaque rigid geometry)
// and the radiance toward the receiver: D(y) = L_e + albedo / pi (E_direct + E_indirect) fresh at the hit with the GI's own model
// (xlionrender_gi_pass_common.glsl: Direct - the sun and the lights, a BVH shadow ray each -, Indirect - the surfels' read -, SurfaceAt -
// the capture's albedo and emission -, Unsupported - its fallback); a miss: the sky (the environment atlas at the receiver's roughness).
// S(y, -w), the hit's own specular: T1 restricted - each light's GGX highlight toward the receiver with the light's own shadow ray (the
// capture's F0 and roughness per triangle); its indirect specular is not evaluated (labelled: a metal at a hit shows its highlights only).
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
struct planar_t { mat4 C2W; vec4 Info; uvec4 Level[8]; uvec4 nBoxes; vec4 Box[128]; };      // (Box: the unrepresented drawn things' world boxes, lo / hi: instance::refl_boxes_v)                                  // the planar view's clip -> its eye's centred world; x: the half view's pixels a unit at distance 1, y: the pyramid's levels; per level: its base, w, h
layout(std430, set = 0, binding = 9) readonly buffer planar_b  { planar_t PL[]; };        // (a ring slot each)
// The hits to texture (xlionrender_refl_hits_comp.glsl), a list: Hits[0] its counts (x, y: by the frame's parity - the hits pass clears
// the next frame's), then three a hit - x: the instance, y: the triangle, z, w: the barycentrics u, v; then E = (E_direct + E_indirect) / pi (rgb), the
// hit's own specular S (rgb) as halves, the ray cone's width at the hit and the receiver's selection weight (halves). The record's U is
// the per-triangle capture's (Le + albedo E + S); then its key (the half size's W x H texels, then the full size's sharp ones, rows padded
// to even) and its place (the hits pass's). The hits pass replaces the record's U with the material's own texels where it shaded them. Only the
// receivers whose reflection shows (Push.m_Tex: their visible reflectance - F0 through the split sum's DFG, Karis' fit - times their
// sharpness): a white dielectric shows 4% of what it reflects, its hits keep the capture's
layout(std430, set = 0, binding = 10) buffer hits_b { uvec4 Hits[]; };
layout(set = 1, binding = 0) uniform sampler2D GColor;     // xy: the octahedral normal (x 0.5 + 0.5), z: roughness
layout(set = 1, binding = 1) uniform sampler2D GDepth;
layout(set = 1, binding = 2) uniform sampler2D EnvAtlas;   // the sky's prefiltered atlas (xlionrender_environment.glsl EnvSample)
layout(set = 1, binding = 3) uniform sampler2D Planar;     // the planar view (instance::renderPlanar), this size, x mirrored; alpha: drawn (m_bPlanar)
layout(set = 1, binding = 4) uniform sampler2D PlanarDepth;
layout(set = 1, binding = 5) uniform sampler2D GMotion;    // w: the motion's flag and the surface's F0 luminance (1 + F0, or -F0)

// m_SharpMax: the receivers sharper than this are traced at full size (0: none); m_Full: this dispatch is that one - the full size, its
// records' first (x, y, z; x 0: the half-size dispatch)
// m_Tex: x the receivers' roughness up to which their hits get their material's own shading (fading over its last quarter), y the visible
// reflectance from which (scene::m_ReflectionTextureMaxRoughness, m_ReflectionTextureMinReflectance), w the list's capacity; x 0: none
layout(push_constant) uniform push_t { uint m_Mode, m_Slot; int m_TlasRoot, m_PrimOffset; vec3 m_Eye; uint m_Frame; float m_EnvIntensity, m_MaxRough, m_bPlanar, m_SharpMax; vec4 m_Plane; ivec4 m_Full; vec4 m_Tex; } Push;

const uint refl_hit_v = 1u, refl_miss_v = 2u, refl_planar_v = 3u, refl_none_v = 0u;    // the hit class (bits 24-27): HIT_WORLD_SUPPORTED, MISS_WORLD, HIT_PLANAR_SUPPORTED, nothing traced

shared uint s_Rays, s_Hits, s_Visits, s_Planar, s_PlanarRough, s_Incomplete, s_Textured;
shared uvec2 s_Pair[64];       // (the full-size dispatch: two pixels a record)

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

// Does the ray's prefix (O, D, up to TMax) cross a drawn thing the GI's BVH does not hold (its box)? PREFIX_INCOMPLETE: a one-sided
// certificate - a box is conservative for what it holds, so a ray that crosses none of them has seen all there is
bool PrefixIncomplete(vec3 O, vec3 D, float TMax)
{
    const vec3 Inv = 1.0 / mix(D, vec3(1e-12), lessThan(abs(D), vec3(1e-12)));
    for (uint b = 0u; b < PL[Push.m_Slot].nBoxes.x; ++b)
    {
        const vec3 t0 = (PL[Push.m_Slot].Box[2u * b].xyz - O) * Inv, t1 = (PL[Push.m_Slot].Box[2u * b + 1u].xyz - O) * Inv;
        const vec3 tn = min(t0, t1), tf = max(t0, t1);
        const float a = max(max(tn.x, tn.y), max(tn.z, 0.0)), z = min(min(tf.x, tf.y), min(tf.z, TMax));
        if (a <= z) return true;
    }
    return false;
}

// The GI's Direct (xlionrender_gi_pass_common.glsl: the sun and its lights, a shadow ray each) with each light's GGX highlight toward V
// (mb_standard_pbr.frag's: D, correlated Smith, Schlick) for a surface of F0 and roughness Rough: E the irradiance, S the radiance
void DirectAndSpecular(vec3 P, vec3 N, vec3 V, vec3 F0, float Rough, bool bSpec, out vec3 E, out vec3 S)
{
    const uint Lt = GD[2];
    E = vec3(0.0); S = vec3(0.0);
    const vec3  O  = P + N * 1e-3;
    const float a  = max(Rough, 0.045) * max(Rough, 0.045), a2 = a * a;
    const float NdotV = max(dot(N, V), 1e-4);
    const vec3 SunD = GV(Lt), SunC = GV(Lt + 3u);
    for (int k = -1; k < int(GD[3]); ++k)
    {
        vec3 L, C; float d = 1e30;
        if (k < 0)
        {
            if ((SunC.x + SunC.y + SunC.z) <= 0.0) continue;
            L = -SunD; C = SunC;
        }
        else
        {
            const uint b = Lt + 57u + 12u * uint(k);
            const vec3 Dv = GV(b) - P;
            d = length(Dv);
            const float s = d / GF(b + 6u);
            if (s >= 1.0 || d < 1e-6) continue;
            L = Dv / d;
            const float Co = GF(b + 10u);
            const float Cone = Co > -1.5 ? smoothstep(Co, max(GF(b + 11u), Co + 1e-6), dot(-L, GV(b + 7u))) : 1.0;
            if (Cone <= 0.0) continue;
            C = GV(b + 3u) * ((1.0 - s * s * s * s) / (1.0 + 2.0 * s * s) * Cone);
        }
        const float c = dot(N, L);
        if (c <= 0.0 || Occluded(O, L, k < 0 ? 1e30 : d - 2e-3)) continue;
        E += C * c;
        if (!bSpec) continue;
        const vec3  H  = normalize(L + V);
        const float NdotH = max(dot(N, H), 0.0), Dd = NdotH * NdotH * (a2 - 1.0) + 1.0;
        const float Vis = 0.5 / max(c * sqrt(NdotV * NdotV * (1.0 - a2) + a2) + NdotV * sqrt(c * c * (1.0 - a2) + a2), 1e-6);
        const vec3  F  = F0 + (1.0 - F0) * pow(clamp(1.0 - dot(L, H), 0.0, 1.0), 5.0);
        S += C * c * F * (a2 / (gi_pi_v * Dd * Dd) * Vis);
    }
}

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
    if (gl_LocalInvocationIndex == 0u) { s_Rays = 0u; s_Hits = 0u; s_Visits = 0u; s_Planar = 0u; s_PlanarRough = 0u; s_Incomplete = 0u; s_Textured = 0u; }
    barrier();
    const bool  bFull = Push.m_Full.x > 0;
    const ivec2 Size = bFull ? Push.m_Full.xy : ivec2(F[Push.m_Slot].Params.xy);
    const ivec2 p    = ivec2(gl_GlobalInvocationID.xy);
    const bool  bIn  = p.x < Size.x && p.y < Size.y;
    const uint  i    = uint(Size.x * Size.y) + uint(p.y * Size.x + p.x);
    const float z    = bIn ? texelFetch(GDepth, p, 0).r : 1.0;
    uvec4 Rec = uvec4(0u);
    const uint Key = bFull ? uint(F[Push.m_Slot].Params.x * F[Push.m_Slot].Params.y) + uint(p.y * ((Size.x + 1) & ~1) + p.x) : uint(p.y * Size.x + p.x);
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
        const bool  bPlanarHere = Push.m_Mode == 0u && Push.m_bPlanar > 0.5 && Rough < Push.m_MaxRough && abs(PlaneDist) < 0.01 + 0.002 * length(P) && dot(N, Push.m_Plane.xyz) > 0.95;
        const bool  bSharpHere  = Push.m_Mode == 0u && !bPlanarHere && Rough < Push.m_SharpMax;      // (traced at full size)
        if (bFull && !bSharpHere) { }      // (the full-size dispatch: the sharp ones only; the rest: none - the half-size records serve them)
        else if (!bFull && bSharpHere) { }  // (the half-size dispatch leaves the sharp ones to the full-size one)
        else if (bPlanarHere)
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
            const vec3  Lt  = Push.m_Mode >= 4u ? vec3(0.0, 0.0, 1.0) : reflect(-Ve, H);
            if (Lt.z > 0.0)
            {
                // (mode 4, measurement: the camera's own ray to the texel - the BVH against the raster: a hit within 1% of the depth counts)
                const vec3  L = Push.m_Mode >= 4u ? normalize(P) : normalize(Lt.x * T + Lt.y * B + Lt.z * N);
                const float w = (1.0 + Lambda(Ve.z, a)) / (1.0 + Lambda(Ve.z, a) + Lambda(Lt.z, a));     // G2 / G1
                const vec3  O = Push.m_Mode >= 4u ? Push.m_Eye : P + Push.m_Eye + N * (0.002 * length(P) + 0.001);
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
                        g_HitGridRead = true;
                        if (!Indirect(Hit.m_Inst, Hit.m_Tri, Hit.m_U, Hit.m_V, Ei)) Ei = Unsupported(Hit.m_Inst, Hit.m_Tri, Hit.m_U, Hit.m_V);
                        const bool bSpec = (GD[R + 47u] & 128u) != 0u;      // (its triangles' specular: world::SetTriangleSpecular)
                        const uint Sb = bSpec ? GD[R + 48u] + 4u * uint(Hit.m_Tri) : 0u;
                        vec3 Ed, Sy;
                        DirectAndSpecular(Py, Ny, -L, bSpec ? GV(Sb) : vec3(0.04), bSpec ? GF(Sb + 3u) : 1.0, bSpec, Ed, Sy);
                        if (EmitterDirectExternal()) Ed += EmitterIrradianceAt(Py, Ny, h);
                        // the hit's own reflection of its surroundings, fully rough (Unreal's EnvBRDFApproxFullyRough at Lumen's hits: the
                        // specular color x 0.45 as more diffuse): a metal hit (albedo ~0) was black in a mirror - only its lights' highlights
                        Sy += (bSpec ? GV(Sb) : vec3(0.04)) * 0.45 * (1.0 / gi_pi_v) * (Ed + Ei);
                        Lr = Emission / max(GF(5u), 1e-6) + Albedo * (1.0 / gi_pi_v) * (Ed + Ei) + Sy;     // (the emission as the surface shows it: SurfaceAt's carries the GI's emissive scale)
                        // the receiver's visible reflectance (its F0's luminance from the G-buffer's motion w; the split sum's directional
                        // albedo, Karis' fit) times its sharpness: the hits worth the material's own texels
                        const float Mw = texelFetch(GMotion, p, 0).w, F0l = Mw > 0.5 ? Mw - 1.0 : -Mw;
                        const vec4  c0 = vec4(-1.0, -0.0275, -0.572, 0.022), c1 = vec4(1.0, 0.0425, 1.04, -0.04);
                        const vec4  rr = Rough * c0 + c1;
                        const float a4 = min(rr.x * rr.x, exp2(-9.28 * Ve.z)) * rr.x + rr.y;
                        const vec2  AB = vec2(-1.04, 1.04) * a4 + rr.zw;
                        const float Sel = (F0l * AB.x + AB.y) * (1.0 - smoothstep(0.75 * Push.m_Tex.x, Push.m_Tex.x, Rough));
                        const uint  Par = uint(Push.m_Tex.z) & 1u;          // (the frame's parity: the hits pass's count)
                        const uint  At  = Push.m_Tex.x > 0.0 && Sel >= 0.75 * Push.m_Tex.y ? atomicAdd(Hits[0][Par], 1u) : 0xFFFFFFFFu;
                        if (At != 0xFFFFFFFFu) atomicAdd(s_Textured, 1u);
                        if (At < uint(Push.m_Tex.w))
                        {
                            // the ray cone (Amanatides; Ray Tracing Gems ch. 20): the texel's angle, twice the surface's spread (its normal's
                            // change to the next texels: a convex mirror's minification) and the lobe's, times the distance, from the
                            // texel's own width at the receiver
                            const vec4  P1 = F[Push.m_Slot].C2W * vec4(Ndc + vec2(2.0 / float(Size.x), 0.0), z, 1.0);
                            const float Px = acos(clamp(dot(normalize(P1.xyz / P1.w), P / length(P)), -1.0, 1.0));
                            const vec3  Nx = OctDecode(texelFetch(GColor, min(p + ivec2(1, 0), Size - 1), 0).xy * 2.0 - 1.0);
                            const vec3  Ny = OctDecode(texelFetch(GColor, min(p + ivec2(0, 1), Size - 1), 0).xy * 2.0 - 1.0);
                            const float Beta = 0.5 * acos(clamp(min(dot(N, Nx), dot(N, Ny)), -1.0, 1.0));
                            const float Cone = Px * length(P) + (Px + 2.0 * Beta + a) * Hit.m_T;
                            Hits[1u + 3u * At]  = uvec4(uint(Hit.m_Inst), uint(Hit.m_Tri), floatBitsToUint(Hit.m_U), floatBitsToUint(Hit.m_V));
                            const vec3  E = (Ed + Ei) * (1.0 / gi_pi_v);
                            Hits[2u + 3u * At]  = uvec4(packHalf2x16(E.rg), packHalf2x16(vec2(E.b, Sy.r)), packHalf2x16(Sy.gb), packHalf2x16(vec2(Cone, Sel)));
                            Hits[3u + 3u * At]  = uvec4(Key, 0xFFFFFFFFu, 0u, 0u);
                        }
                        Class = refl_hit_v; Dist = Hit.m_T;
                    }
                    else
                    {
                        Lr = AtlasSample(EnvAtlas, vec2(0.0), kEnvAtlas, L, 0.0) * Push.m_EnvIntensity;      // (the sky along the sampled direction: the lobe is the sampling's, not the atlas')
                        Class = refl_miss_v;
                    }
                    const float Conf = 1.0 - smoothstep(0.5 * Push.m_MaxRough, Push.m_MaxRough, Rough);     // (B: sharp receivers; glossy ones fade to the probes until the reconstruction)
                    Rec.x = packHalf2x16(Lr.rg);
                    Rec.y = packHalf2x16(vec2(Lr.b, Conf));
                    Rec.z = packHalf2x16(vec2(Dist, w));
                    Rec.w |= Class << 24;
                    if (PrefixIncomplete(O, L, Hit.m_Inst >= 0 ? Hit.m_T : 1e30)) { Rec.w |= 1u << 28; atomicAdd(s_Incomplete, 1u); }     // (bit 28: PREFIX_INCOMPLETE)
                }
                else if (Push.m_Mode >= 4u)
                {
                    // (mode 4: the hit's point from its barycentrics against the raster's: within 1% of the distance a hit; s_Incomplete:
                    // the hits of the view's centre, |ndc| < 0.2)
                    float Dy = 1e30;
                    if (Hit.m_Inst >= 0)
                    {
                        const uint  R = InstRec(Hit.m_Inst), M = MeshOf(R);
                        const uvec3 Tr = MTri(M, Hit.m_Tri);
                        const float b0 = 1.0 - Hit.m_U - Hit.m_V;
                        Dy = length(ToWorld(R, MPos(M, Tr.x) * b0 + MPos(M, Tr.y) * Hit.m_U + MPos(M, Tr.z) * Hit.m_V) - Push.m_Eye);

                    }
                    if (abs(Dy - length(P)) < 0.01 * length(P)) { atomicAdd(s_Hits, 1u); if (all(lessThan(abs(Ndc), vec2(0.2)))) atomicAdd(s_Incomplete, 1u); }
                    // (and as the record, for the reflection debug view: green agrees, red the BVH nearer, blue farther / missed)
                    const float e = (Dy - length(P)) / length(P);
                    const float Mw = texelFetch(GMotion, p, 0).w;
                    const vec3  C = Push.m_Mode == 7u ? vec3(Mw > 0.5 ? Mw - 1.0 : -Mw, Mw > 0.5 ? 1.0 : 0.0, abs(Mw) < 1e-6 ? 1.0 : 0.0)   // (7: the G-buffer's F0 luminance (red), its motion flag (green), none (blue))
                                  : Push.m_Mode == 6u ? vec3(max(dot(N, -normalize(P)), 0.0))                                      // (6: N.V - a sphere's bright spot at its centre)
                                  : Push.m_Mode == 5u ? N * 0.5 + 0.5                                                              // (5: the G-buffer's normal the trace uses)
                                  : vec3(clamp(-e * 10.0, 0.0, 1.0), abs(e) < 0.01 ? 1.0 : 0.0, clamp(e * 10.0, 0.0, 1.0));    // (red / blue: 10 x the error)
                    Rec.x = packHalf2x16(C.rg); Rec.y = packHalf2x16(vec2(C.b, 1.0)); Rec.z = packHalf2x16(vec2(0.0, 1.0)); Rec.w |= refl_hit_v << 24;
                }
                else if (Hit.m_Inst >= 0) atomicAdd(s_Hits, 1u);
            }
        }
    }
    if (!bFull && bIn && (Push.m_Mode == 0u || Push.m_Mode >= 4u)) Records[i] = Rec;
    if (bFull)          // two pixels a record: rgb and, its sign the confidence, the view depth (the materials match it)
    {
        const float Depth = unpackHalf2x16(Rec.w & 0xFFFFu).x;
        const float Conf  = unpackHalf2x16(Rec.y).y;
        s_Pair[gl_LocalInvocationIndex] = Conf > 0.0 ? uvec2(Rec.x, packHalf2x16(vec2(unpackHalf2x16(Rec.y).x, Depth))) : uvec2(0u);
    }
    barrier();
    if (bFull && bIn && Push.m_Mode == 0u && (p.x & 1) == 0)
    {
        const uint  k = uint(p.y * ((Size.x + 1) & ~1) + p.x);     // (rows padded to even: a record never spans two)
        const uvec2 b = p.x + 1 < Size.x ? s_Pair[gl_LocalInvocationIndex + 1u] : uvec2(0u);
        Records[uint(Push.m_Full.z) + k / 2u] = uvec4(s_Pair[gl_LocalInvocationIndex], b);
    }
    if (gl_LocalInvocationIndex == 0u)
    {
        atomicAdd(Stats[Push.m_Slot * 8u + 0u], s_Rays);
        atomicAdd(Stats[Push.m_Slot * 8u + 1u], s_Hits);
        atomicAdd(Stats[Push.m_Slot * 8u + 2u], s_Visits);
        atomicAdd(Stats[Push.m_Slot * 8u + 3u], s_Planar);
        atomicAdd(Stats[Push.m_Slot * 8u + 4u], s_PlanarRough);
        atomicAdd(Stats[Push.m_Slot * 8u + 5u], s_Incomplete);
        atomicAdd(Stats[Push.m_Slot * 8u + (bFull ? 7u : 6u)], s_Textured);     // (the textured hits: the half size's, the full size's)
    }
}
