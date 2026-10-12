#version 450
// Fullscreen triangle (-1,-1) (3,-1) (-1,3); NDC y = -1 is the target's first row, which UV v = 0 samples.
layout(location = 0) in vec2 in_Pos;
layout(location = 0) out vec2 out_UV;

void main()
{
    out_UV      = in_Pos * 0.5 + 0.5;
    gl_Position = vec4(in_Pos, 0.0, 1.0);
}
