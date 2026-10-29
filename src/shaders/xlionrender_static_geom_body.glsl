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
    mat4   PrevW2C;        // last frame's (the screen tail's motion: details::ubo_geom_view), its origin's offset (w: 1 there is one)
    vec4   PrevWorldOffset;
} mesh;

// Per instance (features/instances/xlionrender_instance_slots.h): persistent, indexed by push.instanceIndex
struct InstanceData
{
    vec4   Rows[3];        // Local space -> the instance buffer's origin: the affine matrix's 3 rows
    vec4   Material[3];    // the constant material (xlionrender_geom_standard_frag.glsl reads it)
    vec4   PrevRows[3];    // last frame's transform (the screen tail's motion: instance_slots)
};
layout(std430, set = 1, binding = 1) readonly buffer InstanceBuffer
{
    InstanceData Instances[];
} instances;


// Fragment shader inputs
invariant gl_Position;                              // the depth prepass and the shading compute it alike (renderScenePass: LESS_OR_EQUAL)
layout(location = 0) out VaryingFull Out;
layout(location = 6) out vec2 OutUV1;                // the second UVs (mb_varying_definition_full.glsl)
layout(location = 8) flat out uint OutInstance;     // the slot, for fragment shaders that read the instance record (geom standard)
layout(location = 9) out vec4 OutGI;                // surfel GI: sum(w E), sum(w), interpolated and divided per pixel (mb_standard_pbr.frag); 0 without GI
layout(location = 10) out vec3 OutGIPos;            // the per-pixel read (mb_standard_pbr.frag): mesh space position,
layout(location = 11) out vec3 OutGINormal;         //   normal,
layout(location = 12) flat out uvec2 OutGIRef;      //   the first surfel in GILight, the GI level (xlionrender_surfel_gi.glsl gi_none_v: none)
layout(location = 13) out vec4 OutGITangent;        //   and the tangent (w: the binormal's sign): the normal-mapped normal in mesh space
layout(location = 14) out vec4 OutPrevClip;         // last frame's clip position (the screen tail's motion: the G-buffer variants read it; w 0: none)
layout(location = 15) flat out uint OutCluster;      // the cluster drawn (the G-buffer prepass: the submesh instance, xlionrender_gprepass_frag.glsl)

#ifdef XLION_GI
#include "xlionrender_surfel_gi.glsl"

// The weighted sum over the vertex's surfels at the instance's level (world space distances: any scale)
vec4 SurfelGI(in const vec3 LocalPos, in const vec3 LocalNormal, in const mat3x4 L2wRows, in const float MeanScale)
{
    if (push.giLevel == gi_none_v || push.giLevel == gi_probes_v || push.giLevel == gi_screen_v) return vec4(0.0);     // (no surfels: the probe grid; read at half resolution: the fragment upsamples)
    g_GIMoments = (push.giLevel & gi_no_moments_v) == 0u;
    const uint Level = push.giLevel & 0xFFu, LevelBase = GIMesh[Level];
    const uint GIVertex = uint(int(gl_VertexIndex) + push.giVertexDelta);       // (GI sections: the vertex in its section's mesh)
    const uint P0 = GIMesh[8u + 2u * GIVertex], P1 = GIMesh[9u + 2u * GIVertex];
    const uint Idx[4] = uint[4](P0 & 0xFFFFu, P0 >> 16, P1 & 0xFFFFu, P1 >> 16);
    uint Seen[4] = uint[4](0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu);
    vec4 Sum = vec4(0.0);
    for (int k = 0; k < 4; ++k)
    {
        if (Idx[k] == 0xFFFFu) continue;
        uint s = Idx[k];
        if (Level == 1u)
        {
            s = GIMesh[GIMesh[0] + 8u * s + 7u] & 0xFFFFu;  // the parent (bit 31: a patch surfel)
            bool bSeen = false;
            for (int j = 0; j < k; ++j) bSeen = bSeen || Seen[j] == s;
            if (bSeen) continue;                            // two children of one parent
        }
        Seen[k] = s;
        const uint  b      = LevelBase + 8u * s;
        const vec3  Center = vec3(uintBitsToFloat(GIMesh[b]), uintBitsToFloat(GIMesh[b + 1u]), uintBitsToFloat(GIMesh[b + 2u]));
        const float Radius = uintBitsToFloat(GIMesh[b + 3u]);
        const vec3  Normal = vec3(uintBitsToFloat(GIMesh[b + 4u]), uintBitsToFloat(GIMesh[b + 5u]), uintBitsToFloat(GIMesh[b + 6u]));
        const float c      = dot(Normal, LocalNormal) ;
        if (c < 0.5) continue;
        const float cn     = smoothstep(0.5, 0.75, c);         // fades in (no line where smoothed normals sweep through 0.5)
        const float d      = length(vec4(LocalPos - Center, 0.0) * L2wRows) / (2.0 * Radius * MeanScale);
        const float f      = max(0.0, 1.0 - d);
        const float w      = (GIReadKernel(f) * c * cn + 1e-6) * GIValidity(push.giOffset + s) * GIVisibility(LocalPos, Center, Normal, Radius, push.giOffset + s);
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
    OutPrevClip             = mesh.PrevWorldOffset.w > 0.5 ? mesh.PrevW2C * vec4(VertData.LocalPos * mat3x4(I.PrevRows[0], I.PrevRows[1], I.PrevRows[2]) + mesh.PrevWorldOffset.xyz, 1.0) : vec4(0.0);
    Out.ShadowPosition      = vec4(0.5, 0.5, 2.0, 1.0);                 // the editors' single shadow map (LDR materials) only
    gl_Position             = mesh.w2C * Out.wSpacePosition;

    //
    // Set the UVs
    //
    Out.UV = VertData.UV;
    OutUV1 = VertData.UV1;

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

    OutCluster  = push.clusterIndex;      // (the G-buffer prepass: both of its vertex shaders, the GI one and not)
#ifdef XLION_GI
    OutGI       = SurfelGI(VertData.LocalPos.xyz, normalize(VertData.Normal.xyz), L2wRows, (scales.x + scales.y + scales.z) / 3.0);
    OutGIRef    = push.giLevel == gi_probes_v || push.giLevel == gi_screen_v ? uvec2(0u, push.giLevel) : uvec2(GIDebugCode(push.giLevel) == 4u ? push.clusterIndex : push.giOffset, push.giLevel);     // SURFACE_COLOR: the cluster (SurfelGITriangleColor)
#else
    OutGI       = vec4(0.0);
    OutGIRef    = uvec2(0u, 0xFFFFFFFFu);
#endif
    OutGIPos    = OutGIRef.y == 0xFFFFFFFEu ? Out.wSpacePosition.xyz : VertData.LocalPos.xyz;     // (the probe grid's read: the eye's world)
    OutGINormal = VertData.Normal.xyz;
    OutGITangent = VertData.Tangent;
}

