#version 450
// The selection's outline over the tone-mapped image (the editor's selection outline). The mask (xlionrender_outline_mask_frag.glsl): R the
// selection covers the pixel, G its nearest fragment is in plain sight. Drawn with alpha blending, only over the object's projected bounds.
//  - Outside the selection: a ring of taps around the pixel (8 at the radius, 4 inside at half of it) gives the fraction of it that is selection;
//    that fraction, through a smoothstep, is the outline's opacity: a soft anti-aliased edge of the radius' thickness, which a hard
//    neighbour compare (one pixel) or a box dilate (many taps) do not give. The part of the selection in plain sight makes a solid edge, the
//    part behind the scene a ghost edge (Unreal's dim hidden outline, CryEngine's ghost alpha).
//  - Inside the selection: nothing where it is in plain sight (the object keeps its own look); a faint hatched tint where it is hidden.
layout(binding = 0) uniform sampler2D Mask;

layout(push_constant) uniform PushConstants
{
    vec4 Color;     // rgb the outline's color, a its opacity
    vec4 Params;    // x radius (px), y the hidden part's opacity, z the hidden fill's opacity
} P;

layout(location = 0) out vec4 outColor;

float Cover(ivec2 C, ivec2 Offset, out float Vis)
{
    const vec2 M = texelFetch(Mask, C + Offset, 0).rg;
    Vis = M.r * M.g;
    return M.r;
}

void main()
{
    const ivec2 C  = ivec2(gl_FragCoord.xy);
    const vec2  M0 = texelFetch(Mask, C, 0).rg;

    if (M0.r > 0.5)
    {
        // The selection's own pixels: only the hidden ones are tinted (a 2 x 2 hatch)
        const float Hatch = (((C.x >> 1) + (C.y >> 1)) & 1) == 0 ? 1.0 : 0.45;
        outColor = M0.g > 0.5 ? vec4(0.0) : vec4(P.Color.rgb, P.Params.z * Hatch * P.Color.a);
        return;
    }

    const float R = max(P.Params.x, 1.0);
    const vec2  Ring8[8] = vec2[8](vec2(1, 0), vec2(0.7071, 0.7071), vec2(0, 1), vec2(-0.7071, 0.7071), vec2(-1, 0), vec2(-0.7071, -0.7071), vec2(0, -1), vec2(0.7071, -0.7071));
    float Any = 0.0, Vis = 0.0;
    for (int i = 0; i < 8; ++i)
    {
        float V;
        Any += Cover(C, ivec2(round(Ring8[i] * R)), V);
        Vis += V;
    }
    for (int i = 1; i < 8; i += 2)         // the inner ring: the diagonals at half the radius
    {
        float V;
        Any += Cover(C, ivec2(round(Ring8[i] * (0.5 * R))), V);
        Vis += V;
    }
    if (Any <= 0.0) { outColor = vec4(0.0); return; }

    const float Edge   = 1.0 / 12.0;
    const float aVis   = smoothstep(0.0, 0.34, Vis * Edge);
    const float aHid   = smoothstep(0.0, 0.34, (Any - Vis) * Edge) * P.Params.y;
    outColor = vec4(P.Color.rgb, max(aVis, aHid) * P.Color.a);
}
