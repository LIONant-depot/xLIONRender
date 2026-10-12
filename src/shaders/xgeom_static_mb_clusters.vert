//
// Cluster struct
//
struct ClusterData
{
    vec4 posScale;                 // xyz = position scale, w = trash
    vec4 posTranslation;           // xyz = position translation, w = trash
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
// Push constant for cluster index (updated per draw)
//
layout(push_constant) uniform PushConstants
{
    uint    clusterIndex;  // Index into cluster array
} push;

