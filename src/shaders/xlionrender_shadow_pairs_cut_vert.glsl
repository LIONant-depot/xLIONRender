#version 450
// The pairs (xlionrender_shadow_pairs_vert.glsl) for a masked material's cutout: the cluster vertex path's varyings (the material's
// fragment reads its UVs and stops after the cut) with the record's cluster, instance and page - the page placement in
// xlionrender_static_geom_body.glsl XLION_SHADOW_RECORDS. Records at binding 2, the pages at 13 (3 is the lit layout's lights)
#define XLION_SHADOW_RECORDS
#include "xlionrender_static_geom_body.glsl"
