#version 450
// Writes one local light's shadow map into the atlas (features/shadows/xlionrender_shadow_local.h) from the casters' depth in the
// scratch: drawn as a fullscreen triangle over the map's tile (viewport), depth test ALWAYS. Every texel gets the light's linear
// distance to the nearest caster in its direction, / radius (1 = nothing).
//  - point light: the tile is octahedral; each direction reads the cube face it falls in (6 faces in the scratch)
//  - spot light: the tile is the light's frustum (1 face)
// Both with a border of local_border_v texels: the octahedral one folded over the edges, the spot's clamped.
#include "xlionrender_octahedral.glsl"

layout(binding = 0) uniform sampler2D SamplerScratch;

layout(push_constant) uniform remap
{
    ivec4   Tile;           // x, y: the tile's origin in the atlas (texels), z: its size, w: face size in the scratch
    vec4    Params;         // x: near, y: far (= radius), z: tan of the half field of view (spot), w: 1 point, 0 spot
    ivec4   Faces[3];       // the faces' origins in the scratch: xy, zw pairs in the order +X -X +Y -Y +Z -Z (spot: the first)
} P;

const float kBorder = 2.0;

void main()
{
    const float Inner = float(P.Tile.z) - 2.0 * kBorder;
    const vec2  UV    = ((gl_FragCoord.xy - vec2(P.Tile.xy)) - kBorder) / Inner * 2.0 - 1.0;
    const float Near  = P.Params.x, Far = P.Params.y;

    vec2  XY;           // the face's NDC
    float Ray;          // radial distance / distance along the face axis
    ivec2 Origin;
    if (P.Params.w > 0.5)
    {
        const vec3 D = OctDecode(OctFold(UV));
        const vec3 A = abs(D);
        int  Face;
        vec3 F;
        if      (A.x >= A.y && A.x >= A.z) { Face = D.x > 0.0 ? 0 : 1; F = vec3(sign(D.x), 0.0, 0.0); }
        else if (A.y >= A.z)               { Face = D.y > 0.0 ? 2 : 3; F = vec3(0.0, sign(D.y), 0.0); }
        else                               { Face = D.z > 0.0 ? 4 : 5; F = vec3(0.0, 0.0, sign(D.z)); }
        const vec3 Up = abs(F.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
        const vec3 R  = normalize(cross(Up, F));
        const vec3 U  = cross(F, R);
        const float d = dot(D, F);
        XY     = vec2(dot(D, R), dot(D, U)) / d;
        Ray    = 1.0 / d;
        const ivec4 Pair = P.Faces[Face / 2];
        Origin = (Face % 2) == 0 ? Pair.xy : Pair.zw;
    }
    else
    {
        XY     = clamp(UV, vec2(-1.0), vec2(1.0));
        Ray    = length(vec3(XY * P.Params.z, 1.0));
        Origin = P.Faces[0].xy;
    }

    const int   FaceSize = P.Tile.w;
    const ivec2 Texel    = Origin + clamp(ivec2(floor((XY * 0.5 + 0.5) * float(FaceSize))), ivec2(0), ivec2(FaceSize - 1));
    const float Z        = texelFetch(SamplerScratch, Texel, 0).r;
    if (Z >= 1.0) { gl_FragDepth = 1.0; return; }

    const float AlongAxis = Far * Near / (Far - Z * (Far - Near));       // the face's z = Far/(Far-Near) - Far*Near/((Far-Near) d)
    gl_FragDepth = clamp(AlongAxis * Ray / Far, 0.0, 1.0);
}
