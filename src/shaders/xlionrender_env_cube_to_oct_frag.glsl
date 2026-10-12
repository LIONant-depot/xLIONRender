#version 450
// A reflection probe's 6 captured faces (each a scene pass of 90 degrees, FaceW2C's basis - features/shadows/xlionrender_shadow_local.h)
// into the environment source's level 0: an octahedral map with folded borders. The pyramid and the prefilter then run as for the sky.
#include "xlionrender_environment_atlas.glsl"

layout(binding = 0) uniform sampler2D SamplerFacePX;        // +X -X +Y -Y +Z -Z
layout(binding = 1) uniform sampler2D SamplerFaceNX;
layout(binding = 2) uniform sampler2D SamplerFacePY;
layout(binding = 3) uniform sampler2D SamplerFaceNY;
layout(binding = 4) uniform sampler2D SamplerFacePZ;
layout(binding = 5) uniform sampler2D SamplerFaceNZ;

layout(push_constant) uniform cube_to_oct
{
    vec4 Params;        // x: the source's size
} P;

layout(location = 0) out vec4 outColor;

void main()
{
    const float Size  = P.Params.x;
    const float Inner = Size - 2.0 * kEnvBorder;
    const vec2  UV    = (gl_FragCoord.xy - kEnvBorder) / Inner * 2.0 - 1.0;
    const vec3  D     = OctDecode(OctFold(UV));
    const vec3  A     = abs(D);

    int  Face;
    vec3 F;
    if      (A.x >= A.y && A.x >= A.z) { Face = D.x > 0.0 ? 0 : 1; F = vec3(sign(D.x), 0.0, 0.0); }
    else if (A.y >= A.z)               { Face = D.y > 0.0 ? 2 : 3; F = vec3(0.0, sign(D.y), 0.0); }
    else                               { Face = D.z > 0.0 ? 4 : 5; F = vec3(0.0, 0.0, sign(D.z)); }
    const vec3 Up = abs(F.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    const vec3 R  = normalize(cross(Up, F));
    const vec3 U  = cross(F, R);
    const float d = dot(D, F);
    const vec2  FaceUV = vec2(dot(D, R), dot(D, U)) / d * 0.5 + 0.5;

    vec3 C;
    switch (Face)
    {
    case 0:  C = textureLod(SamplerFacePX, FaceUV, 0.0).rgb; break;
    case 1:  C = textureLod(SamplerFaceNX, FaceUV, 0.0).rgb; break;
    case 2:  C = textureLod(SamplerFacePY, FaceUV, 0.0).rgb; break;
    case 3:  C = textureLod(SamplerFaceNY, FaceUV, 0.0).rgb; break;
    case 4:  C = textureLod(SamplerFacePZ, FaceUV, 0.0).rgb; break;
    default: C = textureLod(SamplerFaceNZ, FaceUV, 0.0).rgb; break;
    }
    outColor = vec4(C, 1.0);
}
