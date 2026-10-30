#version 450
// The depth prepass's G-buffer (gi_gbuffer_gather_plan.md, stage 1): the one geometry pass that also tells the GI what each pixel is -
// its world normal (octahedral, R16G16) and its submesh instance (the instance slot and the cluster, exact in R32G32 floats); the GI's
// read gathers from these instead of drawing the geometry again
#include "mb_varying_definition_full.glsl"

layout(location = 0) in VaryingFull In;
layout(location = 8) flat in uint InInstance;       // the slot (xlionrender_static_geom_body.glsl OutInstance)
layout(location = 15) flat in uint InCluster;       // the cluster (xlionrender_static_geom_body.glsl OutCluster)

layout(location = 0) out vec4 OutHDR;               // (the pass's first target: the scene's HDR, only cleared - the pipeline's mask keeps it)
layout(location = 1) out vec2 OutNormal;
layout(location = 2) out vec4 OutId;

void main()
{
    vec3 n = normalize(In.Normal.xyz);      // (not turned to the viewer: a double-sided back face's GI reads the vertex normal, as the materials' own read)
    vec2 o = n.xy / (abs(n.x) + abs(n.y) + abs(n.z));
    if (n.z < 0.0) o = (1.0 - abs(o.yx)) * vec2(o.x >= 0.0 ? 1.0 : -1.0, o.y >= 0.0 ? 1.0 : -1.0);
    OutHDR    = vec4(0.0);
    OutNormal = o;
    OutId     = vec4(float(InInstance + 1u), gl_FrontFacing ? float(InCluster) : -float(InCluster) - 1.0, length(In.wSpacePosition.xyz), 0.0);     // (x 0: nothing drawn there - the cleared value; y < 0: a back face; z: the distance, exact - the depth buffer's is not, far away)
}
