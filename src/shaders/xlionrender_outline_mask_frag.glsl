#version 450
// The selection's mask (the editor's selection outline, xlionrender_outline_composite_frag.glsl): the selected object's own triangles, drawn with
// the vertex paths of the casters (rigid, skin) into an RGBA8 target with a depth buffer of its own, so the nearest fragment of the object is
// the one that stays. R: the selection covers the pixel. G: that nearest fragment is in plain sight (not behind what the scene drew).
layout(binding = 0) uniform sampler2D SceneDepth;

layout(location = 0) out vec4 outMask;

void main()
{
    const float Scene = texelFetch(SceneDepth, ivec2(gl_FragCoord.xy), 0).r;
    // The same surface at the same depth must count as in sight: a tolerance relative to how far it is (depth crowds toward 1 far away)
    const float Vis = gl_FragCoord.z <= Scene + 5.0e-4 * (1.0 - Scene) + 1.0e-6 ? 1.0 : 0.0;
    outMask = vec4(1.0, Vis, 0.0, 1.0);
}
