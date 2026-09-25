#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal; // Kept for the existing mesh vertex layout.

layout(std140, push_constant) uniform PushConstants
{
    mat4 L2C;
    vec4 ViewportAndRadius;  // xy viewport px; z outline radius px; w max fractional expansion e.g. 0.08
    vec4 Color;
    vec4 Bounds;             // xy projected OBB center viewport-local px; zw half-extents px
    vec4 Axis;               // xy normalized OBB long axis in viewport px; zw unused
} uniforms;

void main()
{
    vec4 clip = uniforms.L2C * vec4(inPosition, 1.0);
    if (clip.w > 0.00001)
    {
        vec2 viewportPx = uniforms.ViewportAndRadius.xy;
        float radiusPx   = uniforms.ViewportAndRadius.z;
        float maxScale   = uniforms.ViewportAndRadius.w;
        vec2 centerPx    = uniforms.Bounds.xy;
        vec2 halfSizePx  = max(uniforms.Bounds.zw, vec2(0.00001));
        vec2 axisX       = normalize(uniforms.Axis.xy);
        vec2 axisY       = vec2(-axisX.y, axisX.x);

        vec2 positionPx = (clip.xy / clip.w * 0.5 + 0.5) * viewportPx;
        vec2 deltaPx    = positionPx - centerPx;

        float scaleX = 1.0 + min(radiusPx / halfSizePx.x, maxScale);
        float scaleY = 1.0 + min(radiusPx / halfSizePx.y, maxScale);

        vec2 expandedPx = centerPx
                        + axisX * dot(deltaPx, axisX) * scaleX
                        + axisY * dot(deltaPx, axisY) * scaleY;

        clip.xy = (expandedPx / viewportPx * 2.0 - 1.0) * clip.w;
    }
    gl_Position = clip;
}