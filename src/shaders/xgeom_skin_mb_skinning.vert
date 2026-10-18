//
// Decodes the fused position stream's packed bone offsets/weights and builds this vertex's blended
// skinning matrix. Mirrors xgeom_skin::geom::vertex::getBoneOffset()/getWeights() exactly (see
// xgeom_skin.h) - same bit layout, same weight-range math - so CPU and GPU decode agree bit-for-bit.
//
// Deliberately plain 32-bit uint arithmetic throughout (no 64-bit integer extension): the packed
// value's only field that straddles the 32/48-bit split (Weight0, bits 28-35) is reassembled from a
// low nibble in `lo32` and a high nibble in `hi16` - see the bit-layout comment in xgeom_skin.h.
//
// The cluster's bone addressing (xgeom_skin.h cluster_data) is read once and branched on once per vertex, outside the influence loop:
// the same for every vertex of a cluster. BASE_OFFSET: base bone + the 7 bit index; PALETTE: BoneMap[palette start + the 7 bit index].
//
mat4 ComputeSkinMatrix(uint lo32, uint hi16, ClusterData Cluster, uint maxInfluences)
{
    const uint boneOffset0 =  lo32        & 0x7Fu;
    const uint boneOffset1 = (lo32 >>  7) & 0x7Fu;
    const uint boneOffset2 = (lo32 >> 14) & 0x7Fu;
    const uint boneOffset3 = (lo32 >> 21) & 0x7Fu;

    const uint q0 = ((lo32 >> 28) & 0xFu) | ((hi16 & 0xFu) << 4);   // 8 bits, straddles lo32/hi16
    const uint q1 = (hi16 >>  4) & 0x7Fu;                            // 7 bits
    const uint q2 = (hi16 >> 11) & 0x1Fu;                            // 5 bits

    const float w0 = 0.25 + (float(q0) / 255.0)  * 0.75;
    const float w1 =        (float(q1) / 127.0)  * 0.5;
    const float w2 =        (float(q2) / 31.0)   * (1.0 / 3.0);
    const float w3 = max(0.0, 1.0 - (w0 + w1 + w2));

    uint boneIndex[4];
    if ((Cluster.boneAddressing >> 30) == 1u)
    {
        const uint start = Cluster.boneAddressing & 0x3FFFFFFFu;
        boneIndex = uint[4](BoneMapEntry(start + boneOffset0), BoneMapEntry(start + boneOffset1), BoneMapEntry(start + boneOffset2), BoneMapEntry(start + boneOffset3));
    }
    else
    {
        const uint base = uint(Cluster.posScale.w);
        boneIndex = uint[4](base + boneOffset0, base + boneOffset1, base + boneOffset2, base + boneOffset3);
    }
    const float rawWeight[4]  = float[4](w0, w1, w2, w3);

    // Weight-truncation skin LOD (Part A/C): at maxInfluences==k, use the first k-1 stored weights
    // as-is and force the k-th USED weight to 1-sum(the rest), so influences always sum to exactly 1
    // regardless of how many are evaluated. At maxInfluences==1, bone 0 gets weight 1.0 outright.
    float usedWeight[4] = float[4](0.0, 0.0, 0.0, 0.0);
    float sumSoFar = 0.0;
    for (uint i = 0u; i < maxInfluences; ++i)
    {
        const bool isLast = (i == maxInfluences - 1u);
        usedWeight[i] = isLast ? max(0.0, 1.0 - sumSoFar) : rawWeight[i];
        sumSoFar += usedWeight[i];
    }

    mat4 Skin = mat4(0.0);
    for (uint i = 0u; i < maxInfluences; ++i)
        Skin += boneMatrix[boneIndex[i]] * usedWeight[i];

    return Skin;
}
