
#include "xgeom_static_mb_clusters.vert"
#include "mb_input_definition_full.vert"

//
// Vertex inputs
//
layout(location = 0) in ivec4 in_PosExtra;      // xyz = compressed pos (R16G16B16_SINT), w = extra 16 bits
layout(location = 1) in uvec2 in_UV;            // compressed UV (R16G16_UNORM)
layout(location = 2) in uvec4 in_OctNormTan;    // high 8 bits of oct normal/tangent (R8G8B8A8_UINT)

//
// Fast oct-encoded normal/tangent decode (12-bit N, 12/11-bit T)
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
// Gets the vertex local position
//
mb_full_vertex getVertexData()
{
    mb_full_vertex Data;

    // Select cluster by push constant index
    ClusterData selectedCluster = cluster[push.clusterIndex];

    //
    // Decode compressed position from int16 to [-1,1]
    //
    const vec3 norm_pos         = (vec3(in_PosExtra.xyz) + 32768.0) / 32767.5 - 1.0;
    Data.LocalPos               = vec4(norm_pos * selectedCluster.posScale.xyz + selectedCluster.posTranslation.xyz, 1.0);

    //
    // Decode UV and apply cluster scaling/offset
    //
    const vec2 norm_uv          = vec2(in_UV) / 65535.0;
    Data.UV                     = norm_uv * selectedCluster.uvScaleTranslation.xy + selectedCluster.uvScaleTranslation.zw;

    //
    // Decode Tangent Binormal Normal
    //

    // Extract extra_bits
    const uint extra_bits       = uint(in_PosExtra.w);

    // Binormal sign: bit 15 of extra_bits (0:+1, 1:-1)
    const uint  sign_bit        = (extra_bits >> 15u) & 1u;
    const float binormal_sign   = (sign_bit == 0u) ? 1.0 : -1.0;

    // Remaining 15 bits (0-14) for low-precision components
    // N_x_low: 4 bits (bits 0-3)
    // N_y_low: 4 bits (bits 4-7)
    // T_x_low: 4 bits (bits 8-11)
    // T_y_low: 3 bits (bits 12-14)
    const uvec4 lows            = (uvec4(extra_bits) >> uvec4(0u, 4u, 8u, 12u)) & uvec4(0xFu, 0xFu, 0xFu, 7u);

    // Combine high 8 bits (from vertex attr) + low bits
    const uvec4 combined        = (in_OctNormTan << uvec4(4u, 4u, 4u, 3u)) | lows;

    // Decode oct-encoded normal (12 bit) and tangent (12/11 bit)
    const vec2 enc_normal       = vec2(combined.xy) / 4095.0;
    const vec2 enc_tangent      = vec2(combined.z / 4095.0, combined.w / 2047.0);

    Data.Tangent.xyz            = oct_decode(enc_tangent);
    Data.Tangent.w              = binormal_sign;
    Data.Normal                 = oct_decode(enc_normal);
    Data.VertColor              = vec4(1.,1.,1.,1.);

    return Data;
}