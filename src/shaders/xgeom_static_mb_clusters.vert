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

#ifdef XLION_RECORDS
// xlionrender_static_geom_gi_records_vert.glsl: the push constants below, one record per draw of an indirect run (its first instance)
struct DrawRecord
{
    uint    clusterIndex;
    uint    instanceIndex;
    uint    giOffset;
    uint    giLevel;
    uint    materialBase;
    int     giVertexDelta;
};
layout(std430, set = 1, binding = 12) readonly buffer DrawRecords
{
    DrawRecord Records[];
};
#define push Records[gl_InstanceIndex]
#else
//
// Push constant for cluster index (updated per draw)
//
layout(push_constant) uniform PushConstants
{
    uint    clusterIndex;  // Index into cluster array
    uint    instanceIndex; // xLIONRender: the draw's slot in the instance buffer
#ifdef XLION_GI
    uint    giOffset;      // xLIONRender surfel GI: the instance's first surfel in GILight
    uint    giLevel;       // its active surfel level (0xFFFFFFFF: no GI)
    uint    materialBase;  // (the fragment's: its MaterialParams base)
    int     giVertexDelta; // GI sections: this cluster's vertex in its section's GIMesh is gl_VertexIndex + this (0: the geometry's own)
#endif
} push;
#endif

