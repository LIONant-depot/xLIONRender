#version 450
// Reflections, milestone A (documentation/reflections_strategy.md sec. 11, 12): the world trace's cost measured before anything is built
// on it. One reflection ray per texel of the screen tail's thin G-buffer (half size: its octahedral normal, roughness, depth), a GGX
// visible normal (Heitz 2018), the nearest hit in the GI's BVH (GICast: opaque rigid geometry). Push.m_Mode (instance::m_ReflTraceBench):
// 1 mirror rays (coherent), 2 roughness 0.5 everywhere (incoherent), 3 the materials' roughness. It writes only its counters per ring
// slot (rays, hits, the BVH's visits) - the benchmark, and the skeleton of the pass graph's pass 5
layout(local_size_x = 8, local_size_y = 8) in;

#include "xlionrender_octahedral.glsl"

struct frame_t    { mat4 C2W; mat4 Reproject; vec4 Params; };      // (xlionrender_tail_denoise_comp.glsl's: C2W to the camera-centred world)
struct GINode     { vec4 m_B[6]; ivec4 m_Child; };
struct GITri4     { vec4 m_V0[3]; vec4 m_E1[3]; vec4 m_E2[3]; ivec4 m_Tri; };
struct GIInstance { vec4 m_W2L[3]; ivec4 m_Info; };

layout(std430, set = 0, binding = 0) readonly buffer frame_b     { frame_t    F[]; };
layout(std430, set = 0, binding = 1) readonly buffer nodes_b     { GINode     GIBvhNodes[]; };
layout(std430, set = 0, binding = 2) readonly buffer tri4_b      { GITri4     GIBvhTri4[]; };
layout(std430, set = 0, binding = 3) readonly buffer instances_b { GIInstance GIBvhInstances[]; };
layout(std430, set = 0, binding = 4) readonly buffer tlas_b      { int        GIBvhTlasLeaves[]; };
layout(std430, set = 0, binding = 5)          buffer stats_b     { uint       Stats[]; };     // 4 per ring slot: rays, hits, visits, 0
layout(set = 1, binding = 0) uniform sampler2D GColor;      // xy: the octahedral normal (x 0.5 + 0.5), z: roughness
layout(set = 1, binding = 1) uniform sampler2D GDepth;

layout(push_constant) uniform push_t { uint m_Mode, m_Slot; int m_TlasRoot, m_PrimOffset; vec3 m_Eye; uint m_Frame; } Push;

#include "xlionrender_gi_bvh.glsl"

shared uint s_Rays, s_Hits, s_Visits;

uint Hash(uint x) { x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16; return x; }

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

void main()
{
    if (gl_LocalInvocationIndex == 0u) { s_Rays = 0u; s_Hits = 0u; s_Visits = 0u; }
    barrier();
    const ivec2 Size = ivec2(F[Push.m_Slot].Params.xy);
    const ivec2 p    = ivec2(gl_GlobalInvocationID.xy);
    const float z    = p.x < Size.x && p.y < Size.y ? texelFetch(GDepth, p, 0).r : 1.0;
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
        const vec3  V   = normalize(-P);
        const vec3  T   = normalize(abs(N.z) < 0.999 ? cross(vec3(0.0, 0.0, 1.0), N) : cross(vec3(1.0, 0.0, 0.0), N));
        const vec3  B   = cross(N, T);
        const vec3  Ve  = vec3(dot(V, T), dot(V, B), max(dot(V, N), 1e-4));
        const uint  h   = Hash(uint(p.x) * 73856093u ^ uint(p.y) * 19349663u ^ Push.m_Frame * 83492791u);
        const vec2  u   = vec2(float(h & 0xFFFFu), float(h >> 16)) / 65536.0;
        const float a   = max(Rough, 0.045) * max(Rough, 0.045);
        const vec3  H   = Push.m_Mode == 1u ? vec3(0.0, 0.0, 1.0) : SampleVNDF(Ve, a, u);
        const vec3  Lt  = reflect(-Ve, H);
        if (Lt.z > 0.0)
        {
            const vec3 L = normalize(Lt.x * T + Lt.y * B + Lt.z * N);
            const gi_hit Hit = GICast(P + Push.m_Eye + N * (0.002 * length(P) + 0.001), L, 1e30);
            atomicAdd(s_Rays, 1u);
            if (Hit.m_Inst >= 0) atomicAdd(s_Hits, 1u);
            atomicAdd(s_Visits, uint(g_GIBvhVisits));
        }
    }
    barrier();
    if (gl_LocalInvocationIndex == 0u)
    {
        atomicAdd(Stats[Push.m_Slot * 4u + 0u], s_Rays);
        atomicAdd(Stats[Push.m_Slot * 4u + 1u], s_Hits);
        atomicAdd(Stats[Push.m_Slot * 4u + 2u], s_Visits);
    }
}
