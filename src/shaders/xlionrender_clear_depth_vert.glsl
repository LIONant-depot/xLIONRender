#version 450
// Clears one shadow-atlas tile to the far depth: the fullscreen triangle at z = 1, drawn with the tile's viewport, depth test ALWAYS.
// (xGPU render passes clear whole attachments; a tile of the atlas is cleared by drawing.)
layout(location = 0) in vec2 in_Pos;

void main()
{
    gl_Position = vec4(in_Pos, 1.0, 1.0);
}
