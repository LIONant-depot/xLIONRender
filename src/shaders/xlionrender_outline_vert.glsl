#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;

// Both stages declare the identical block - this codebase's own push-constant convention requires
// it (every stage that touches the range needs the same struct or nothing rasterizes).
layout(std140, push_constant) uniform PushConstants
{
    mat4 L2C;
    vec4 ViewportAndRadius; // x width, y height, z radius (pixels), w unused
    vec4 Color;
} uniforms;

// Expands each vertex along its OWN surface normal, measured as a fixed pixel distance on screen -
// not toward/away from an object pivot. Radial (vertex-pivot) expansion pushes corners but barely
// moves a long flat face's middle (a wall's side gets almost no outline); normal expansion gives
// every point on the surface the same outline width regardless of the mesh's shape. L2C's own
// rotation part doubles as the normal matrix here since every object in this DLL is translation-only
// (see xlionrender_system.h's own comment) - revisit with a proper inverse-transpose if that changes.
void main()
{
    vec4 clip = uniforms.L2C * vec4(inPosition, 1.0);

    const float width  = uniforms.ViewportAndRadius.x;
    const float height = uniforms.ViewportAndRadius.y;
    const float radius = uniforms.ViewportAndRadius.z;

    if (clip.w > 0.00001 && width > 0.0 && height > 0.0 && radius > 0.0)
    {
        vec4 clipNormal = uniforms.L2C * vec4(inNormal, 0.0);
        if (length(clipNormal.xy) > 0.00001)
        {
            vec2 dirPixels    = normalize(clipNormal.xy) * vec2(width, height);
            vec2 offsetPixels = normalize(dirPixels) * radius;
            vec2 offsetNDC    = offsetPixels * vec2(2.0 / width, 2.0 / height);

            // Multiplying by clip.w makes the offset survive perspective division as the requested
            // fixed screen-space pixel displacement, regardless of the vertex's depth/distance.
            clip.xy += offsetNDC * clip.w;
        }
    }

    gl_Position = clip;
}
