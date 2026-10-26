#version 450
// Forward plan, standard material: constant metal/roughness parameters through mb_standard_pbr.frag's PBRLighting.
// Writes scene-linear HDR (no tone mapping here - that is the tone map pass).
#define XLION_NO_GI_PIXEL          // xlionrender_forward_vert.glsl: no mesh surfels (and the pipeline has no GI buffers)
#include "mb_standard_pbr.frag"
#include "xlionrender_editor_grid.glsl"

layout(set = 2, binding = 2) uniform material_uniforms
{
    vec4 BaseColor;
    vec4 Params;            // x = metallic, y = roughness, z = AO, w = F0
    vec4 Emissive;          // w: 1 the editors' ground grid (the albedo its pattern in world XZ)
} UBOMaterial;

layout(location = 0) out vec4 outColor;

void main()
{
    const vec3 C = PBRLighting
    ( vec3(0.0, 0.0, 1.0)           // tangent-space normal (no normal map)
    , UBOMaterial.Emissive.w > 0.5 ? GridAlbedo(In.wSpacePosition.xyz.xz + UBOLight.CameraPos.xz, In.wSpacePosition.xyz) : UBOMaterial.BaseColor.rgb
    , UBOMaterial.Params.z
    , UBOMaterial.Params.w
    , UBOMaterial.Params.y
    , UBOMaterial.Params.x
    , UBOMaterial.Emissive.rgb
    );
    outColor = vec4(C, 1.0);
}
