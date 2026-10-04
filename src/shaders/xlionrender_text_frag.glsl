#version 450
#extension GL_ARB_separate_shader_objects : enable

// The atlas of the Font resource, whatever its encoding (the font editor's own shader reads it the same way):
//   MTSDF (0): rgb is the multi-channel distance field (the median of the three is the distance)
//   SDF   (1): r is the distance
//   BITMAP(2): r is the coverage of a glyph rasterized at a fixed size (not a distance field: no range, no derivative scale)
layout(binding = 0) uniform sampler2D uAtlas;

layout(std140, push_constant) uniform PushConstants
{
    mat4 W2C;
    vec4 View;
    vec4 Font;      // x: the pixel range of the distance field, y: the output type of the font
} pc;

layout(location = 0) in  vec2 inUV;
layout(location = 1) in  vec4 inColor;
layout(location = 0) out vec4 outFragColor;

float median3(float r, float g, float b)
{
    return max(min(r, g), min(max(r, g), b));
}

void main()
{
    const vec4 Texel = texture(uAtlas, inUV);
    float Coverage;
    if (pc.Font.y > 1.5)
    {
        Coverage = Texel.r;
    }
    else
    {
        const float Distance = pc.Font.y < 0.5 ? median3(Texel.r, Texel.g, Texel.b) : Texel.r;
        // how many screen pixels the range of the field covers right now (it changes with distance and angle: the edge is as sharp at any size)
        const vec2  UnitRange     = vec2(pc.Font.x) / vec2(textureSize(uAtlas, 0));
        const vec2  ScreenTexSize = vec2(1.0) / fwidth(inUV);
        const float PixelRange    = max(0.5 * dot(UnitRange, ScreenTexSize), 1.0);
        Coverage = clamp(PixelRange * (Distance - 0.5) + 0.5, 0.0, 1.0);
    }
    const float Alpha = inColor.a * Coverage;
    if (Alpha < 0.02) discard;
    outFragColor = vec4(inColor.rgb, Alpha);
}
