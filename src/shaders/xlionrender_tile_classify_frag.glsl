#version 450
// The tile pass's classification (XLION_TILE_CLASSIFY, A5.2): each pixel's material graph as its depth - (graph + 1) / 65536 - so a graph's
// full-screen draw tested EQUAL (xlionrender_tile_vert.glsl) runs only on its own pixels: the others are rejected before their fragment
// runs (it would discard them: xlionrender_tile_pixel.glsl TilePixelSetup). No tile material there (nothing, masked, skinned, no block):
// the far depth, which no graph's draw is at. The same test as TilePixelSetup's, on the same data
layout(binding = 3) uniform sampler2D TileId;
layout(std430, set = 1, binding = 9) readonly buffer MaterialParams { vec4 P[]; } g_Mat;

void main()
{
    const vec4 Id = texelFetch(TileId, ivec2(gl_FragCoord.xy), 0);
    float z = 1.0;
    if (Id.w >= 0.5)
    {
        const uint Base = uint(Id.w) - 1u;
        if (Base != 0u)
        {
            const uint Graph = floatBitsToUint(g_Mat.P[Base - 1u].x);
            if (Graph < 65535u) z = float(Graph + 1u) / 65536.0;
        }
    }
    gl_FragDepth = z;
}
