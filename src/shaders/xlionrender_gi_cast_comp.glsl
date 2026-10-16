#version 450
// Part II, G1 (documentation/gi_dynamic_master_plan.md): the nearest hit of each ray of a buffer (GICast), against gi::world::Cast.
// Rays: origin and TMax (w), direction. Hits: t, u, v; the world instance (-1: none) and the mesh's triangle
layout(local_size_x = 64) in;

struct GINode     { vec4 m_B[6]; ivec4 m_Child; };
struct GITri4     { vec4 m_V0[3]; vec4 m_E1[3]; vec4 m_E2[3]; ivec4 m_Tri; };
struct GIInstance { vec4 m_W2L[3]; ivec4 m_Info; };
struct GIRay      { vec4 m_O; vec4 m_D; };
struct GIHit      { vec4 m_TUV; ivec4 m_Ids; };

layout(std430, set = 0, binding = 0) readonly  buffer nodes_b     { GINode     GIBvhNodes[]; };
layout(std430, set = 0, binding = 1) readonly  buffer tri4_b      { GITri4     GIBvhTri4[]; };
layout(std430, set = 0, binding = 2) readonly  buffer instances_b { GIInstance GIBvhInstances[]; };
layout(std430, set = 0, binding = 3) readonly  buffer tlas_b      { int        GIBvhTlasLeaves[]; };
layout(std430, set = 0, binding = 4) readonly  buffer rays_b      { GIRay      GIRays[]; };
layout(std430, set = 0, binding = 5) writeonly buffer hits_b      { GIHit      GIHits[]; };

layout(push_constant) uniform push_t { int m_nRays, m_TlasRoot, m_PrimOffset, m_Pad; } Push;

#include "xlionrender_gi_bvh.glsl"

void main()
{
    const int i = int(gl_GlobalInvocationID.x);
    if (i >= Push.m_nRays) return;
    g_GIBvhTlasRoot   = Push.m_TlasRoot;
    g_GIBvhPrimOffset = Push.m_PrimOffset;
    const gi_hit H = GICast(GIRays[i].m_O.xyz, GIRays[i].m_D.xyz, GIRays[i].m_O.w);
    GIHits[i].m_TUV = vec4(H.m_T, H.m_U, H.m_V, 0.0);
    GIHits[i].m_Ids = ivec4(H.m_Inst, H.m_Tri, 0, 0);
}
