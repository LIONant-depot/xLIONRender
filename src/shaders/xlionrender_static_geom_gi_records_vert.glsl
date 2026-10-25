#version 450
// xlionrender_static_geom_gi_vert.glsl for the GI read at half resolution: each draw's push constants come from a record buffer
// (xgeom_static_mb_clusters.vert XLION_RECORDS, by gl_InstanceIndex), so a run of clusters with the same bindings is one indirect draw
#define XLION_GI
#define XLION_RECORDS
#include "xlionrender_static_geom_body.glsl"
