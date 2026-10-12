#version 450
// xLIONRender copy of xgeom_skin.plugin/source/runtime/GeomSkinBasicShader_vert.glsl (unchanged), renamed so its compiled header
// never collides with the plugin's own. The skin vertex path; xlionrender_static_geom_vert.glsl is the rigid one.

#include "xgeom_skin_mb_input_full.vert"
#include "mb_varying_definition_full.glsl"

// Per view or shadow tile (plans/forward/xlionrender_plan_forward.h ubo_geom_view)
layout(set=2, binding = 0) uniform MeshUniforms
{
    mat4   w2C;            // Small world -> clip space (projection * view)
    vec4   WorldOffset;    // xyz: the instance buffer's origin in that small world
} mesh;

// Per instance (features/instances/xlionrender_instance_slots.h): persistent, indexed by push.instanceIndex
struct InstanceData
{
    vec4   Rows[3];        // Local space -> the instance buffer's origin: the affine matrix's 3 rows
    vec4   Material[3];    // the constant material (xlionrender_geom_standard_frag.glsl reads it)
};
layout(std430, set = 1, binding = 2) readonly buffer InstanceBuffer
{
    InstanceData Instances[];
} instances;


// Fragment shader inputs
layout(location = 0) out VaryingFull Out;
layout(location = 8) flat out uint OutInstance;     // the slot, for fragment shaders that read the instance record (geom standard)


void main()
{
    const mb_full_vertex VertData = getVertexData();

    //
    // Transform to all needed spaces
    //
    OutInstance             = push.instanceIndex;
    const InstanceData I    = instances.Instances[push.instanceIndex];
    const mat3x4 L2wRows    = mat3x4(I.Rows[0], I.Rows[1], I.Rows[2]);
    Out.wSpacePosition      = vec4(VertData.LocalPos * L2wRows + mesh.WorldOffset.xyz, 1.0);
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
    const mat3 RotMat           = transpose(mat3(L2wRows));
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

