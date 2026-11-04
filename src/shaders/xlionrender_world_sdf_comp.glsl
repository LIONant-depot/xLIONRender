#version 450
// Builds the world's distance field (xlionrender_world_sdf.glsl; prototype, static, the camera where it was): mode 0, short BVH rays from each cell's
// center, 14 directions: the least distance, capped; mode 1, a flood: the thin things the directions missed reach their neighbours. Unsigned
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

#include "xlionrender_gi_pass_common.glsl"           // (bindings 0-4: the BVH and the GI's data)
layout(std430, set = 0, binding = 5) buffer sdf_b { float WorldSDF[]; };
layout(std430, set = 0, binding = 6) buffer sdf_surf_b { uvec2 WorldSDFSurf[]; };      // per cell, the nearest surface: x: albedo, sqrt, RGB8, 255 in A where there is one; y: the normal, octahedral 16 16

layout(push_constant) uniform push_t { vec3 m_Center; float m_Cell0; int m_Cascade, m_Slice, m_Mode, m_TlasRoot, m_PrimOffset; float m_MaxCells; } Push;

const int  dim_v = 128;
const vec3 dirs_v[14] = vec3[14](vec3(1, 0, 0), vec3(-1, 0, 0), vec3(0, 1, 0), vec3(0, -1, 0), vec3(0, 0, 1), vec3(0, 0, -1),
    vec3(0.57735, 0.57735, 0.57735), vec3(-0.57735, 0.57735, 0.57735), vec3(0.57735, -0.57735, 0.57735), vec3(-0.57735, -0.57735, 0.57735),
    vec3(0.57735, 0.57735, -0.57735), vec3(-0.57735, 0.57735, -0.57735), vec3(0.57735, -0.57735, -0.57735), vec3(-0.57735, -0.57735, -0.57735));

vec2 Oct(vec3 n) { n /= abs(n.x) + abs(n.y) + abs(n.z); return n.z >= 0.0 ? n.xy : (1.0 - abs(n.yx)) * vec2(n.x >= 0.0 ? 1.0 : -1.0, n.y >= 0.0 ? 1.0 : -1.0); }

int At(ivec3 i) { return 4 + Push.m_Cascade * dim_v * dim_v * dim_v + i.z * dim_v * dim_v + i.y * dim_v + i.x; }

void main()
{
    if (Push.m_Cascade == 0 && Push.m_Mode == 0 && gl_GlobalInvocationID == uvec3(0u)) { WorldSDF[0] = Push.m_Center.x; WorldSDF[1] = Push.m_Center.y; WorldSDF[2] = Push.m_Center.z; WorldSDF[3] = Push.m_Cell0; }
    const ivec3 i    = ivec3(gl_GlobalInvocationID) + ivec3(0, 0, Push.m_Slice);
    const float Cell = Push.m_Cell0 * float(1 << Push.m_Cascade);
    const float MaxD = Push.m_MaxCells * Cell;
    if (i.z >= dim_v) return;
    if (Push.m_Mode == 0)
    {
        g_GIBvhTlasRoot   = Push.m_TlasRoot;
        g_GIBvhPrimOffset = Push.m_PrimOffset;
        g_GIBvhSeeThrough = true;
        const vec3 O = Push.m_Center + (vec3(i) + 0.5 - float(dim_v) * 0.5) * Cell;
        float d = MaxD;
        gi_hit Best; Best.m_Inst = -1; Best.m_Tri = -1; Best.m_U = 0.0; Best.m_V = 0.0; Best.m_T = MaxD;
        for (int k = 0; k < 14; ++k) { const gi_hit H = GICast(O, dirs_v[k], d); if (H.m_Inst >= 0 && H.m_T < d) { d = H.m_T; Best = H; } }
        WorldSDF[At(i)] = d;
        uvec2 Surf = uvec2(0u);
        if (Best.m_Inst >= 0)
        {
            const uint  R = InstRec(Best.m_Inst), M = MeshOf(R);
            const uvec3 Tr = MTri(M, Best.m_Tri);
            const float b0 = 1.0 - Best.m_U - Best.m_V;
            const vec3  N = normalize(NormalToWorld(R, MNrm(M, Tr.x) * b0 + MNrm(M, Tr.y) * Best.m_U + MNrm(M, Tr.z) * Best.m_V));
            vec3 Albedo, Emission;
            SurfaceAt(Best.m_Inst, Best.m_Tri, Best.m_U, Best.m_V, Albedo, Emission);
            const uvec3 c = uvec3(sqrt(clamp(Albedo, vec3(0.0), vec3(1.0))) * 255.0 + 0.5);
            const uvec2 o = uvec2((Oct(N) * 0.5 + 0.5) * 65535.0 + 0.5);
            Surf = uvec2(c.x | (c.y << 8) | (c.z << 16) | (255u << 24), o.x | (o.y << 16));
        }
        WorldSDFSurf[At(i) - 4] = Surf;
    }
    else
    {
        float d = WorldSDF[At(i)];
        for (int a = 0; a < 3; ++a) for (int s = -1; s <= 1; s += 2)
        {
            ivec3 n = i; n[a] += s;
            if (n[a] >= 0 && n[a] < dim_v) d = min(d, WorldSDF[At(n)] + Cell);
        }
        WorldSDF[At(i)] = d;
    }
}
