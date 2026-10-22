
// The stored vertex colors are 8 bit sRGB (the optional COLOR0 stream): the shaders work in linear
vec4 mb_DecodeVertexColor(vec4 SRGB)
{
    return vec4(mix(SRGB.rgb / 12.92, pow((SRGB.rgb + 0.055) / 1.055, vec3(2.4)), step(0.04045, SRGB.rgb)), SRGB.a);
}

struct mb_full_vertex
{
    vec4 VertColor;         // Vertex color white if none
    vec4 LocalPos;          // In local space ready to be transform
    vec4 Tangent;           // Tangent Local Space in (X,Y,Z) W contains the sign of the binormal
    vec3 Normal;            // Normal Local Space
    vec2 UV;                // Texture Coordinates 
};

