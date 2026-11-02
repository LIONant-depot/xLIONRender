#version 450
// The G-buffer prepass with what the material tile pass needs (XLION_TILE, material_tiles_plan.md): xlionrender_gprepass_frag.glsl's targets, the material's base in the id's w (+ 1: 0 a masked
// or skinned surface, which stays on the raster path), and the attributes: UV and its gradients, the tangent. As floats for now: the reference; the polar codec after the measurement
#include "mb_varying_definition_full.glsl"

layout(location = 0) in VaryingFull In;
layout(location = 8) flat in uint InInstance;
layout(location = 15) flat in uint InCluster;
layout(location = 16) flat in uint InMaterialBase;  // (xlionrender_static_geom_body.glsl)

layout(location = 0) out vec4 OutHDR;
layout(location = 1) out vec2 OutNormal;
layout(location = 2) out vec4 OutId;
layout(location = 3) out vec4 OutAttr0;             // U, V, dU/dx, dV/dx
layout(location = 4) out vec4 OutAttr1;             // dU/dy, dV/dy, the tangent's octahedral x, y (y + 4: a negative binormal sign)

vec2 Oct(in const vec3 n)
{
    vec2 o = n.xy / (abs(n.x) + abs(n.y) + abs(n.z));
    if (n.z < 0.0) o = (1.0 - abs(o.yx)) * vec2(o.x >= 0.0 ? 1.0 : -1.0, o.y >= 0.0 ? 1.0 : -1.0);
    return o;
}
void main()
{
    OutHDR    = vec4(0.0);
    OutNormal = Oct(normalize(In.Normal.xyz));
    OutId     = vec4(float(InInstance + 1u), gl_FrontFacing ? float(InCluster) : -float(InCluster) - 1.0, length(In.wSpacePosition.xyz), float(InMaterialBase + 1u));
    const vec2 dx = dFdx(In.UV), dy = dFdy(In.UV);
    const vec2 t  = Oct(normalize(In.Tangent.xyz));
    OutAttr0 = vec4(In.UV, dx);
    OutAttr1 = vec4(dy, t.x, t.y + (In.Tangent.w < 0.0 ? 4.0 : 0.0));
}
