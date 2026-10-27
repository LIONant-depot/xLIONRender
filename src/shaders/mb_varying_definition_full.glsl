#ifndef MB_VARYING_DEFINITION_FULL_GLSL
#define MB_VARYING_DEFINITION_FULL_GLSL
struct VaryingFull
{
    vec4 wSpacePosition;    // Position in small world
    vec4 ShadowPosition;    // Position in shadow texture-clip-space
    vec4 VertColor;         // Color of the vertices
    vec4 Tangent;           // Tangent in world XYZ, Binormal Sign in w
    vec3 Normal;            // Normal in world XYZ
    vec2 UV;                // Final texture coordinates
};
// Location 6: vec2 UV1, the second texture coordinates (the optional UV1 stream, else UV0). The vertex shaders write it
// (out vec2 OutUV1); a fragment shader that reads it declares layout(location = 6) in vec2 InUV1.
#endif
