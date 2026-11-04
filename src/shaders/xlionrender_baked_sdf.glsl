// The geometry's baked distance field (features/gi/xlionrender_world_sdf_bake.h, reflections_sdf_plan's contract): it only proves space empty. A ray steps by
// the lower bound, what it can't prove empty goes to the triangles (a short BVH ray), so the result is the BVH's. The includer declares BakedSDF: [0] the cell,
// [1..3] the table's first brick, [4..6] its size, [7] where the bytes start, [8..] the table, then 512 bytes a brick
const uint baked_absent_v = 0xFFFFFFFFu;

float BakedBound(vec3 P, float Cell, ivec3 Org, ivec3 Dim, uint Bytes)
{
    const ivec3 c = ivec3(floor(P / Cell)), b = (c >> 3) - Org;
    float v = 255.0;
    if (all(greaterThanEqual(b, ivec3(0))) && all(lessThan(b, Dim)))
    {
        const uint i = BakedSDF[8u + uint(b.z * Dim.y * Dim.x + b.y * Dim.x + b.x)];
        if (i != baked_absent_v)
        {
            const ivec3 l = c & 7;
            const uint  o = i * 512u + uint(l.z * 64 + l.y * 8 + l.x);
            v = float((BakedSDF[Bytes + (o >> 2)] >> ((o & 3u) * 8u)) & 255u);
        }
    }
    return v / 255.0 * 4.0 * Cell - 0.8661 * Cell;
}

gi_hit BakedCast(vec3 O, vec3 D)
{
    const float Cell = uintBitsToFloat(BakedSDF[0]);
    const ivec3 Org  = ivec3(int(BakedSDF[1]), int(BakedSDF[2]), int(BakedSDF[3])), Dim = ivec3(int(BakedSDF[4]), int(BakedSDF[5]), int(BakedSDF[6]));
    const uint  Bytes = BakedSDF[7];
    gi_hit Miss; Miss.m_T = 1e30; Miss.m_U = 0.0; Miss.m_V = 0.0; Miss.m_Inst = -1; Miss.m_Tri = -1;
    // Only the table's box holds triangles
    const vec3  Inv = GISafeInverse(D), Lo = (vec3(Org) * 8.0 * Cell - O) * Inv, Hi = (vec3(Org + Dim) * 8.0 * Cell - O) * Inv;
    const vec3  A = min(Lo, Hi), B = max(Lo, Hi);
    float t = max(max(A.x, max(A.y, A.z)), 0.0);
    const float End = min(B.x, min(B.y, B.z));
    const float Win = 2.0 * Cell;
    for (int s = 0; s < 128; ++s)
    {
        if (t >= End) return Miss;
        const float d = BakedBound(O + D * t, Cell, Org, Dim, Bytes);
        if (d > 0.0) { t += d; continue; }
        gi_hit E = GICast(O + D * t, D, Win);
        if (E.m_Inst >= 0) { E.m_T += t; return E; }
        t += Win;
    }
    gi_hit E = GICast(O + D * t, D, 1e30);           // (out of steps: the BVH finishes it, the same answer)
    if (E.m_Inst >= 0) E.m_T += t;
    return E;
}
