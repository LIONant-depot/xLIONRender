// The GI's rays on the GPU (documentation/gi_dynamic_master_plan.md, Part II, G1): the same walk as gi::world::Cast over the structures
// gi::world::PackGPU writes. Every mesh's BVH4 nodes and the TLAS's are in one array (a child code >= 0 a node, < 0 ~ a leaf: for a
// mesh its 4-triangle row, for the TLAS its entry in the TLAS leaves); per TLAS primitive the instance's world-to-mesh rows, its mesh's
// root code and its world id. The includer declares the buffers with these names (std430):
//   GINode     { vec4 m_B[6]; ivec4 m_Child; }          GIBvhNodes[]
//   GITri4     { vec4 m_V0[3], m_E1[3], m_E2[3]; ivec4 m_Tri; }   GIBvhTri4[]
//   GIInstance { vec4 m_W2L[3]; ivec4 m_Info; }         GIBvhInstances[]       (m_Info.x: root code, .y: world instance)
//   int                                                 GIBvhTlasLeaves[]      (per leaf first, count; the primitives from g_GIBvhPrimOffset)
// and sets g_GIBvhTlasRoot, g_GIBvhPrimOffset

int g_GIBvhTlasRoot   = 0;
int g_GIBvhPrimOffset = 0;

const int gi_bvh_stack_v = 64;

// Which of a node's 4 children the ray enters before TMax (a bit each), and where (TNear): bvh::Box4
int GIBox4(int Node, vec3 O, vec3 Inv, ivec3 Near, float TMax, out vec4 TNear)
{
    const vec4 nx = (GIBvhNodes[Node].m_B[Near.x] - O.x) * Inv.x, fx = (GIBvhNodes[Node].m_B[Near.x ^ 1] - O.x) * Inv.x;
    const vec4 ny = (GIBvhNodes[Node].m_B[Near.y] - O.y) * Inv.y, fy = (GIBvhNodes[Node].m_B[Near.y ^ 1] - O.y) * Inv.y;
    const vec4 nz = (GIBvhNodes[Node].m_B[Near.z] - O.z) * Inv.z, fz = (GIBvhNodes[Node].m_B[Near.z ^ 1] - O.z) * Inv.z;
    TNear = max(max(nx, ny), max(nz, vec4(0.0)));
    const vec4 tf = min(min(fx, fy), min(fz, vec4(TMax)));
    const bvec4 In = lessThanEqual(TNear, tf);
    return (In.x ? 1 : 0) | (In.y ? 2 : 0) | (In.z ? 4 : 0) | (In.w ? 8 : 0);
}

// gi::SafeInverse, as it is: a component under 1e-12 as 1e-12 (its sign dropped: the slab is then all or nothing)
vec3 GISafeInverse(vec3 D)
{
    return vec3(1.0 / (abs(D.x) > 1e-12 ? D.x : 1e-12), 1.0 / (abs(D.y) > 1e-12 ? D.y : 1e-12), 1.0 / (abs(D.z) > 1e-12 ? D.z : 1e-12));
}

// tri4::Intersect: the nearest of a row's 4 triangles before TMax (-1 none)
int GIIntersectTri4(int Row, vec3 O, vec3 D, float TMax, out float Dist, out float U, out float V)
{
    int Best = -1;
    Dist = TMax; U = 0.0; V = 0.0;
    for (int k = 0; k < 4; ++k)
    {
        const vec3 E1 = vec3(GIBvhTri4[Row].m_E1[0][k], GIBvhTri4[Row].m_E1[1][k], GIBvhTri4[Row].m_E1[2][k]);
        const vec3 E2 = vec3(GIBvhTri4[Row].m_E2[0][k], GIBvhTri4[Row].m_E2[1][k], GIBvhTri4[Row].m_E2[2][k]);
        const vec3 P  = cross(D, E2);
        const float det = dot(E1, P);
        if (abs(det) <= 1e-12) continue;
        const float inv = 1.0 / det;
        const vec3 Tv = O - vec3(GIBvhTri4[Row].m_V0[0][k], GIBvhTri4[Row].m_V0[1][k], GIBvhTri4[Row].m_V0[2][k]);
        const float u = dot(Tv, P) * inv;
        const vec3 Q = cross(Tv, E1);
        const float v = dot(D, Q) * inv;
        const float t = dot(E2, Q) * inv;
        if (u >= 0.0 && v >= 0.0 && u + v <= 1.0 && t > 1e-5 && t < Dist) { Best = k; Dist = t; U = u; V = v; }
    }
    return Best;
}

// One BVH from Root, the nearest first; a mesh's leaves are triangle rows, the TLAS's are instances (bTlas). The nearest hit before
// TMax: its instance (Inst), its triangle (Tri: the mesh's), U, V. A mesh is walked with the ray in its space (t is the same parameter)
struct gi_hit { float m_T, m_U, m_V; int m_Inst, m_Tri; };

void GIWalkMesh(int Root, vec3 O, vec3 D, int Inst, inout gi_hit H)
{
    const vec3  Inv  = GISafeInverse(D);
    const ivec3 Near = ivec3(Inv.x < 0.0 ? 1 : 0, Inv.y < 0.0 ? 3 : 2, Inv.z < 0.0 ? 5 : 4);
    int   Code[gi_bvh_stack_v]; float TS[gi_bvh_stack_v];
    int n = 0;
    Code[n] = Root; TS[n] = 0.0; ++n;
    while (n > 0)
    {
        --n;
        const int c = Code[n];
        if (TS[n] > H.m_T) continue;
        if (c < 0)
        {
            float d, u, v;
            const int k = GIIntersectTri4(~c, O, D, H.m_T, d, u, v);
            if (k >= 0) { H.m_T = d; H.m_U = u; H.m_V = v; H.m_Inst = Inst; H.m_Tri = GIBvhTri4[~c].m_Tri[k]; }
            continue;
        }
        vec4 TNear;
        int Mask = GIBox4(c, O, Inv, Near, H.m_T, TNear);
        // push far .. near (the nearest last: popped first)
        int Kids[4]; float KT[4]; int h = 0;
        for (int k = 0; k < 4; ++k)
            if ((Mask & (1 << k)) != 0)
            {
                int j = h++;
                for (; j > 0 && KT[j - 1] < TNear[k]; --j) { Kids[j] = Kids[j - 1]; KT[j] = KT[j - 1]; }
                Kids[j] = GIBvhNodes[c].m_Child[k]; KT[j] = TNear[k];
            }
        for (int k = 0; k < h && n < gi_bvh_stack_v; ++k) { Code[n] = Kids[k]; TS[n] = KT[k]; ++n; }
    }
}

gi_hit GICast(vec3 O, vec3 D, float TMax)
{
    gi_hit H; H.m_T = TMax; H.m_U = 0.0; H.m_V = 0.0; H.m_Inst = -1; H.m_Tri = -1;
    const vec3  Inv  = GISafeInverse(D);
    const ivec3 Near = ivec3(Inv.x < 0.0 ? 1 : 0, Inv.y < 0.0 ? 3 : 2, Inv.z < 0.0 ? 5 : 4);
    int   Code[gi_bvh_stack_v]; float TS[gi_bvh_stack_v];
    int n = 0;
    Code[n] = g_GIBvhTlasRoot; TS[n] = 0.0; ++n;
    while (n > 0)
    {
        --n;
        const int c = Code[n];
        if (TS[n] > H.m_T) continue;
        if (c < 0)
        {
            const int First = GIBvhTlasLeaves[2 * (~c)], Count = GIBvhTlasLeaves[2 * (~c) + 1];
            for (int k = 0; k < Count; ++k)
            {
                const int p = GIBvhTlasLeaves[g_GIBvhPrimOffset + First + k];
                const vec4 R0 = GIBvhInstances[p].m_W2L[0], R1 = GIBvhInstances[p].m_W2L[1], R2 = GIBvhInstances[p].m_W2L[2];
                const vec3 MO = vec3(dot(R0.xyz, O) + R0.w, dot(R1.xyz, O) + R1.w, dot(R2.xyz, O) + R2.w);
                const vec3 MD = vec3(dot(R0.xyz, D), dot(R1.xyz, D), dot(R2.xyz, D));
                GIWalkMesh(GIBvhInstances[p].m_Info.x, MO, MD, GIBvhInstances[p].m_Info.y, H);
            }
            continue;
        }
        vec4 TNear;
        int Mask = GIBox4(c, O, Inv, Near, H.m_T, TNear);
        int Kids[4]; float KT[4]; int h = 0;
        for (int k = 0; k < 4; ++k)
            if ((Mask & (1 << k)) != 0)
            {
                int j = h++;
                for (; j > 0 && KT[j - 1] < TNear[k]; --j) { Kids[j] = Kids[j - 1]; KT[j] = KT[j - 1]; }
                Kids[j] = GIBvhNodes[c].m_Child[k]; KT[j] = TNear[k];
            }
        for (int k = 0; k < h && n < gi_bvh_stack_v; ++k) { Code[n] = Kids[k]; TS[n] = KT[k]; ++n; }
    }
    return H;
}
