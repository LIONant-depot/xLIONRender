#version 450
// xLIONRender copy of xgeom_skin.plugin/source/runtime/GeomSkinBasicShader_vert.glsl (unchanged), renamed so its compiled header
// never collides with the plugin's own. The skin vertex path; xlionrender_static_geom_vert.glsl is the rigid one.

#include "xgeom_skin_mb_input_full.vert"
#include "mb_varying_definition_full.glsl"

// Per draw (plans/forward/xlionrender_plan_forward.h ubo_static_geom_mesh): 112 bytes
layout(set=2, binding = 0) uniform MeshUniforms
{
    mat3x4 L2wRows;    // Local space -> camera-centered small world: the affine matrix's 3 rows (each column here is one)
    mat4   w2C;        // Small world -> clip space (projection * view)
} mesh;


// Fragment shader inputs
layout(location = 0) out VaryingFull Out;


void main()
{
    const mb_full_vertex VertData = getVertexData();

    //
    // Transform to all needed spaces
    //
    Out.wSpacePosition      = vec4(VertData.LocalPos * mesh.L2wRows, 1.0);
    Out.ShadowPosition      = vec4(0.5, 0.5, 2.0, 1.0);                 // the editors' single shadow map (LDR materials) only
    gl_Position             = mesh.w2C * Out.wSpacePosition;

    //
    // Set the UVs
    //
    Out.UV = VertData.UV;

    //
    // Handle the Tangent space transforms for MikkTSpace
    //

    // Extract scales from L2w column lengths (handles non-uniform scales in the L2W)
    // However does not handle skew matrices... not used offen
    const mat3 RotMat           = transpose(mat3(mesh.L2wRows));
    vec3 scales = vec3( length(RotMat[0]),
                        length(RotMat[1]),
                        length(RotMat[2]));

    // Scale normals/tangents inversely before forward transform
    const vec3 scaledNormal     = normalize(VertData.Normal.xyz)  / scales;
    const vec3 scaledTangent    = normalize(VertData.Tangent.xyz) / scales;


    Out.Tangent.xyz             = normalize(RotMat * scaledTangent);
    Out.Tangent.w               = VertData.Tangent.w;                   // Binormal signed
    Out.Normal                  = normalize(RotMat * scaledNormal);

    //
    // Save the vertex color
    //
    Out.VertColor = VertData.VertColor;
}

