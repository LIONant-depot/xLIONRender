#ifndef XLIONRENDER_OCTAHEDRAL_GLSL
#define XLIONRENDER_OCTAHEDRAL_GLSL
// Octahedral mapping of directions to [-1,1]^2 (Cigolle et al. 2014), oriented with down (-Y) at the center and the seams up: a
// light's shadow on the ground below it gets the best texels, the folds land on the ceiling. Shared by the point light shadow
// maps (features/shadows/xlionrender_shadow_local.h) - and meant for the reflection probes.

vec2 OctSignNotZero(in const vec2 V) { return vec2(V.x >= 0.0 ? 1.0 : -1.0, V.y >= 0.0 ? 1.0 : -1.0); }

vec2 OctEncode(in const vec3 N)
{
    vec3 E = vec3(N.x, N.z, -N.y);                          // -Y becomes the octahedron's +Z (the center)
    E /= abs(E.x) + abs(E.y) + abs(E.z);
    return E.z >= 0.0 ? E.xy : (1.0 - abs(E.yx)) * OctSignNotZero(E.xy);
}

vec3 OctDecode(in const vec2 P)
{
    vec3 E = vec3(P, 1.0 - abs(P.x) - abs(P.y));
    if (E.z < 0.0) E.xy = (1.0 - abs(E.yx)) * OctSignNotZero(E.xy);
    E = normalize(E);
    return vec3(E.x, -E.z, E.y);
}

// A point past the square's edge -> the point it continues to (the map folds over each edge, mirrored along it): for the borders
vec2 OctFold(in vec2 P)
{
    if (abs(P.x) > 1.0) { P.x = (P.x > 0.0 ? 2.0 : -2.0) - P.x; P.y = -P.y; }
    if (abs(P.y) > 1.0) { P.y = (P.y > 0.0 ? 2.0 : -2.0) - P.y; P.x = -P.x; }
    return P;
}

#endif
