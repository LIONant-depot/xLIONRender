// The body of xlionrender_static_geom_vert.glsl and xlionrender_static_geom_gi_vert.glsl (XLION_GI: + the surfel GI read).
// xLIONRender copy of xgeom_static.plugin/source/runtime/GeomStaticBasicShader_vert.glsl, renamed so its compiled header
// never collides with the plugin's own in a build that compiles both.

#include "xgeom_static_mb_input_full.vert"
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
layout(std430, set = 1, binding = 1) readonly buffer InstanceBuffer
{
    InstanceData Instances[];
} instances;


// Fragment shader inputs
layout(location = 0) out VaryingFull Out;
layout(location = 8) flat out uint OutInstance;     // the slot, for fragment shaders that read the instance record (geom standard)
layout(location = 9) out vec4 OutGI;                // surfel GI: sum(w E), sum(w), interpolated and divided per pixel (mb_standard_pbr.frag); 0 without GI
layout(location = 10) out vec3 OutGIPos;            // the per-pixel read (mb_standard_pbr.frag): mesh space position,
layout(location = 11) out vec3 OutGINormal;         //   normal,
layout(location = 12) flat out uvec2 OutGIRef;      //   the first surfel in GILight, the GI level (xlionrender_surfel_gi.glsl gi_none_v: none)
layout(location = 13) out vec4 OutGITangent;        //   and the tangent (w: the binormal's sign): the normal-mapped normal in mesh space

#ifdef XLION_GI
#include "xlionrender_surfel_gi.glsl"

// The weighted sum over the vertex's surfels at the instance's level (world space distances: any scale)
vec4 SurfelGI(in const vec3 LocalPos, in const vec3 LocalNormal, in const mat3x4 L2wRows, in const float MeanScale)
{
    if (push.giLevel == gi_none_v) return vec4(0.0);
    const uint Level = push.giLevel & 0xFFu, LevelBase = GIMesh[Level];
    const uint P0 = GIMesh[8u + 2u * uint(gl_VertexIndex)], P1 = GIMesh[9u + 2u * uint(gl_VertexIndex)];
    const uint Idx[4] = uint[4](P0 & 0xFFFFu, P0 >> 16, P1 & 0xFFFFu, P1 >> 16);
    uint Seen[4] = uint[4](0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu);
    vec4 Sum = vec4(0.0);
    for (int k = 0; k < 4; ++k)
    {
        if (Idx[k] == 0xFFFFu) continue;
        uint s = Idx[k];
        if (Level == 1u)
        {
            s = GIMesh[GIMesh[0] + 8u * s + 7u];            // the parent
            bool bSeen = false;
            for (int j = 0; j < k; ++j) bSeen = bSeen || Seen[j] == s;
            if (bSeen) continue;                            // two children of one parent
        }
        Seen[k] = s;
        const uint  b      = LevelBase + 8u * s;
        const vec3  Center = vec3(uintBitsToFloat(GIMesh[b]), uintBitsToFloat(GIMesh[b + 1u]), uintBitsToFloat(GIMesh[b + 2u]));
        const float Radius = uintBitsToFloat(GIMesh[b + 3u]);
        const vec3  Normal = vec3(uintBitsToFloat(GIMesh[b + 4u]), uintBitsToFloat(GIMesh[b + 5u]), uintBitsToFloat(GIMesh[b + 6u]));
        const float c      = dot(Normal, LocalNormal);
        if (c < 0.5) continue;
        const float d      = length(vec4(LocalPos - Center, 0.0) * L2wRows) / (2.0 * Radius * MeanScale);
        const float f      = max(0.0, 1.0 - d);
        const float w      = (f * f * c + 1e-6) * GIValidity(push.giOffset + s) * GIVisibility(LocalPos, Center, Normal, Radius, push.giOffset + s);
        Sum += vec4(GIBasisRead(push.giOffset + s, Normal, LocalNormal) * w, w);
    }
    return Sum;
}
#endif


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

#ifdef XLION_GI
    OutGI       = SurfelGI(VertData.LocalPos.xyz, normalize(VertData.Normal.xyz), L2wRows, (scales.x + scales.y + scales.z) / 3.0);
    OutGIRef    = uvec2(push.giOffset, push.giLevel);
#else
    OutGI       = vec4(0.0);
    OutGIRef    = uvec2(0u, 0xFFFFFFFFu);
#endif
    OutGIPos    = VertData.LocalPos.xyz;
    OutGINormal = VertData.Normal.xyz;
    OutGITangent = VertData.Tangent;
}

