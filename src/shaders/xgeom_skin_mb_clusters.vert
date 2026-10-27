//
// Cluster struct
//
struct ClusterData
{
    vec4 posScale;                 // xyz = position scale, w = this cluster's base skeleton bone index
    vec3 posTranslation;           // position translation
    uint boneAddressing;           // bits 31..30: 0 base offset (posScale.w), 1 palette; bits 29..0: the palette's start in BoneMap
    vec4 uvScaleTranslation;       // xy = UV Scale, zw = UV offset
};

//
// Cluster-level uniforms array (bound once, indexed per draw)
//
layout(std430, set = 1, binding = 0) buffer ClusterBuffer
{
    ClusterData cluster[];  // Unsized array of structs
};

//
// Per-bone skinning matrices for the CURRENT pose (World * InvBindPose per bone), computed CPU-side
// once per draw and indexed as BaseBoneIndex + BoneOffset[i] - see xgeom_skin_mb_skinning.vert.
//
layout(std430, set = 1, binding = 1) buffer BoneMatrixBuffer
{
    mat4 boneMatrix[];
};

//
// The palettes of the palette clusters: uint16 skeleton bone indices, two per 32 bit word (a skin without palettes binds a dummy word)
//
layout(std430, set = 1, binding = 8) readonly buffer BoneMapBuffer
{
    uint boneMapWords[];
};

uint BoneMapEntry(uint i)
{
    return (boneMapWords[i >> 1] >> ((i & 1u) * 16u)) & 0xFFFFu;
}

//
// Push constant for cluster index and skin LOD (updated per draw)
//
layout(push_constant) uniform PushConstants
{
    uint    clusterIndex;   // Index into cluster array
    uint    maxInfluences;  // 1..4 - how many bone weights to actually use (weight-truncation skin LOD)
    uint    instanceIndex;  // xLIONRender: the draw's slot in the instance buffer
    uint    giProbes;       // xLIONRender: 1 the probe grid is bound as GIMesh (the fragment's SurfelGIProbes)
    uint    materialBase;   // xLIONRender: the fragment's (push_skin_geom)
    uint    prevBones;      // xLIONRender: last frame's pose in boneMatrix from this matrix on (the screen tail's motion); 0: none
} push;
