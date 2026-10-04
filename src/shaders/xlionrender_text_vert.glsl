#version 450
#extension GL_ARB_separate_shader_objects : enable

// One instance is one glyph (xlionrender_text_renderer.h): the four corners of a unit quad come from stream 0, everything else from stream 1 (one record per glyph, the data of its label
// repeated in each of its glyphs, so a whole batch of labels is a single draw).
layout(location = 0) in vec2 aCorner;   // (0,0) (1,0) (1,1) (0,1)
layout(location = 1) in vec4 aRect;     // the glyph in em units: min xy, max zw (y up, the origin of the entity at the anchor of the text)
layout(location = 2) in vec4 aUV;       // the atlas rectangle: the uv of the min corner xy and of the max corner zw
layout(location = 3) in vec4 aPos;      // xyz: the origin of the text in the world, w: how big one em is (world units, or pixels in screen size mode)
layout(location = 4) in vec4 aAxisX;    // xyz: the direction the text reads in, as long as the scale of the entity (entity orientation); w: the orientation (0 entity, 1 camera, 2 upright camera)
layout(location = 5) in vec4 aAxisY;    // xyz: the direction of up, as long as the scale of the entity;                                    w: the size mode (0 world, 1 screen pixels)
layout(location = 6) in vec4 aColor;    // rgba

// Both stages declare the identical block (this codebase's push-constant convention).
layout(std140, push_constant) uniform PushConstants
{
    mat4 W2C;
    vec4 View;      // xy: the viewport in pixels
    vec4 Font;      // x: the pixel range of the distance field, y: the output type of the font (0 MTSDF, 1 SDF, 2 BITMAP)
} pc;

layout(location = 0) out vec2 outUV;
layout(location = 1) out vec4 outColor;

// The projection of the host puts the +y of the screen where the rows of W2C say; if a future camera flips it, this one constant is the place (text that reads upside down in camera mode).
const float kScreenUpSign = -1.0;

void main()
{
    const mat4 M = pc.W2C;
    const vec3 Row0 = vec3(M[0][0], M[1][0], M[2][0]);     // the direction of the screen's x in the world (as long as the projection scales it)
    const vec3 Row1 = vec3(M[0][1], M[1][1], M[2][1]);     // the same for the screen's y
    const vec3 Row3 = vec3(M[0][3], M[1][3], M[2][3]);     // the distance along the view direction

    const vec3  Origin = aPos.xyz;
    const float Depth  = dot(Row3, Origin) + M[3][3];

    vec3 Right = aAxisX.xyz;
    vec3 Up    = aAxisY.xyz;
    const float ScaleX = length(Right);
    const float ScaleY = length(Up);
    if (aAxisX.w > 0.5)                                    // faces the camera
    {
        Right = normalize(Row0);
        Up    = kScreenUpSign * normalize(Row1);
        if (aAxisX.w > 1.5)                                // but stays upright: turns around the vertical axis only
        {
            Right = normalize(vec3(Right.x, 0.0, Right.z));
            Up    = vec3(0.0, 1.0, 0.0);
        }
        Right *= ScaleX;
        Up    *= ScaleY;
    }

    float Em = aPos.w;
    if (aAxisY.w > 0.5)                                    // screen size: Em pixels, wherever the text is
        Em *= 2.0 * max(Depth, 0.0001) / (pc.View.y * length(Row1));

    const vec2 E = mix(aRect.xy, aRect.zw, aCorner);
    const vec3 World = Origin + (Right * E.x + Up * E.y) * Em;

    gl_Position = pc.W2C * vec4(World, 1.0);
    outUV       = mix(aUV.xy, aUV.zw, aCorner);
    outColor    = aColor;
}
