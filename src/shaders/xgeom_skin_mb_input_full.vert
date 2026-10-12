
#include "xgeom_skin_mb_clusters.vert"
#include "xgeom_skin_mb_skinning.vert"
#include "mb_input_definition_full.vert"

//
// Vertex inputs - stream 0 is the fused position+skin buffer (xgeom_skin::geom::vertex, 12 bytes),
// stream 1 is vertex_extras (12 bytes)
//
layout(location = 0) in ivec3 in_Pos;          // xyz = compressed pos (read as R16G16B16A16_SINT, w = m_Packed[0..1], unused)
// m_Packed is read as 16 bit words at offsets 6 and 8 of the vertex: D3D12 (Mesa's Vulkan driver under WSL) needs every attribute's offset
// to be a multiple of min(4, its size), so the 4 bytes at offset 6 it used to read as R8G8B8A8_UINT made the pipeline fail.
layout(location = 1) in uint  in_PackedLo;     // m_Packed[0..1] (R16_UINT) - bone offsets 0-1, low bits of bone offset 2
layout(location = 2) in uvec2 in_PackedHi;     // m_Packed[2..5] (R16G16_UINT) - x: rest of the bone offsets + low bits of Weight0, y: Weight0 high bits, Weight1, Weight2
layout(location = 3) in uvec2 in_UV;           // compressed UV (R16G16_UNORM)
layout(location = 4) in uvec2 in_OctNormal;    // full-precision oct normal (R16G16_UINT)
layout(location = 5) in uint  in_OctTangentX;  // full-precision oct tangent X (R16_UINT)
layout(location = 6) in uint  in_OctTangentY_Sign; // oct tangent Y (15 bits) + binormal sign (bit 15) (R16_UINT)

//
// Fast oct-encoded normal/tangent decode
//
vec3 oct_decode(vec2 e)
{
    e = e * 2.0 - 1.0;                          // [0,1] -> [-1,1]
    vec3 v = vec3(e.xy, 1.0 - abs(e.x) - abs(e.y));
    if (v.z < 0.0)
    {
        v.xy = (1.0 - abs(v.yx)) * sign(v.xy);
    }
    return normalize(v);
}

//
// Gets the vertex local position, deformed by skinning
//
mb_full_vertex getVertexData()
{
    mb_full_vertex Data;

    // Select cluster by push constant index
    ClusterData selectedCluster = cluster[push.clusterIndex];
    const uint  baseBoneIndex   = uint(selectedCluster.posScale.w);

    //
    // Decode compressed position from int16 to [-1,1], then to local space
    //
    const vec3 norm_pos = (vec3(in_Pos) + 32768.0) / 32767.5 - 1.0;
    const vec4 bindPos  = vec4(norm_pos * selectedCluster.posScale.xyz + selectedCluster.posTranslation.xyz, 1.0);

    //
    // Decode UV and apply cluster scaling/offset
    //
    const vec2 norm_uv = vec2(in_UV) / 65535.0;
    Data.UV             = norm_uv * selectedCluster.uvScaleTranslation.xy + selectedCluster.uvScaleTranslation.zw;

    //
    // Decode Tangent/Binormal/Normal - full precision, self-contained in vertex_extras
    //
    const uint  signBit         = (in_OctTangentY_Sign >> 15u) & 1u;
    const float binormal_sign   = (signBit == 0u) ? 1.0 : -1.0;

    const vec2 enc_normal  = vec2(in_OctNormal) / 65535.0;
    const vec2 enc_tangent = vec2(float(in_OctTangentX) / 65535.0, float(in_OctTangentY_Sign & 0x7FFFu) / 32767.0);

    const vec3 bindTangent = oct_decode(enc_tangent);
    const vec3 bindNormal  = oct_decode(enc_normal);

    //
    // Build this vertex's skin matrix and deform position/normal/tangent from bind pose
    //
    const uint lo32 = in_PackedLo | (in_PackedHi.x << 16);
    const mat4 Skin = ComputeSkinMatrix(lo32, in_PackedHi.y, baseBoneIndex, push.maxInfluences);
    const mat3 SkinRot = mat3(Skin);

    Data.LocalPos    = Skin * bindPos;
    Data.Tangent.xyz = normalize(SkinRot * bindTangent);
    Data.Tangent.w   = binormal_sign;
    Data.Normal      = normalize(SkinRot * bindNormal);
    Data.VertColor   = vec4(1.,1.,1.,1.);

    return Data;
}
