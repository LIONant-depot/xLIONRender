// Surfel GI (features/gi/xlionrender_surfel_gi.h): the mesh's surfel data and this frame's surfel irradiance, read per vertex
// (xlionrender_static_geom_body.glsl) and per pixel (mb_standard_pbr.frag).
//  GIMesh: a header of 8 uints: [0] level 0 surfels' first uint, [1] level 1's, [2] the read grid's table, [3] its mask (0: no grid),
//          [4] its cell size (float bits), [5] its lists, [6] the table's longest probe; from 8: per vertex 2 uints (4 x 16-bit level 0 surfel indices, 0xFFFF none);
//          per surfel 8 floats: center xyz, radius, normal xyz, parent index (uint bits; level 0); the grid (surfel_mesh::BuildReadGrid):
//          per table entry 4 uints { key low, key high (0xFFFFFFFF: empty), first, count }, the lists of level 0 surfel indices
//  GILight: per surfel of the instance's active level: rgb irradiance, a validity (0: buried)
//  GIMoment: per GILight surfel its depth moments (gi::MomentTexel): 16 hemi-octahedral texels of the mean and mean^2 distance its rays
//          travel (mesh units, capped at its reach)
#ifndef XLIONRENDER_SURFEL_GI_GLSL
#define XLIONRENDER_SURFEL_GI_GLSL

layout(std430, set = 1, binding = 5) readonly buffer GIMeshBuffer  { uint GIMesh[]; };
layout(std430, set = 1, binding = 6) readonly buffer GILightBuffer { vec4 GILight[]; };
layout(std430, set = 1, binding = 7) readonly buffer GIMomentBuffer { uint GIMoment[]; };    // per GILight surfel 16 texels: half mean, half mean^2

// The GI reference a vertex shader passes on (giLevel of the push constants): the level in the low 8 bits, gi_per_pixel_v when the
// pixels may read the grid, 0xFFFFFFFF: no GI
const uint gi_none_v      = 0xFFFFFFFFu;
const uint gi_per_pixel_v = 0x100u;
const uint gi_debug_shift_v = 9u;        // 2 bits: 1 the surfels (each pixel the color of its dominant surfel), 2 the GI light only (scene::gi_settings::m_Debug)

vec4 g_GIDebug = vec4(0.0);             // w > 0: the debug view's color replaces the lit one (GIDebugOr)
vec3 GIDebugOr(in const vec3 Lit) { return g_GIDebug.w > 0.0 ? g_GIDebug.rgb : Lit; }

vec3 GISurfelCenter(in const uint b) { return vec3(uintBitsToFloat(GIMesh[b]), uintBitsToFloat(GIMesh[b + 1u]), uintBitsToFloat(GIMesh[b + 2u])); }
vec3 GISurfelNormal(in const uint b) { return vec3(uintBitsToFloat(GIMesh[b + 4u]), uintBitsToFloat(GIMesh[b + 5u]), uintBitsToFloat(GIMesh[b + 6u])); }

// The anti-leak terms of a read (Wicked Engine's surfel GI), times the falloff and normal weight:
//  - the plane distance: a point off the surfel's plane (across a corner, the far side of a seam) fades within 0.7 of its reach
//  - the depth moments: a point farther than the surfel's rays travel toward it is probably behind something (Chebyshev, cubed)
float GIVisibility(in const vec3 P, in const vec3 Center, in const vec3 Normal, in const float Radius, in const uint LightIndex)
{
    const vec3  D     = P - Center;
    const float Reach = 2.0 * Radius;
    float       w     = clamp(1.0 - abs(dot(D, Normal)) / (0.7 * Reach), 0.0, 1.0);
    const float d     = length(D);
    if (d < 1e-5 * Reach || w <= 0.0) return w;
    const vec3  T  = normalize(cross(abs(Normal.x) < 0.9 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0), Normal)), B = cross(Normal, T);
    const vec3  Dd = D / d;
    const float x  = dot(Dd, T), y = dot(Dd, B), z = max(dot(Dd, Normal), 0.0);
    const float k  = 1.0 / max(abs(x) + abs(y) + z, 1e-6);
    const ivec2 t  = clamp(ivec2((vec2((x + y) * k, (x - y) * k) * 0.5 + 0.5) * 4.0), ivec2(0), ivec2(3));
    const vec2  M  = unpackHalf2x16(GIMoment[LightIndex * 16u + uint(t.y * 4 + t.x)]);
    if (d > M.x)
    {
        const float Var = abs(M.x * M.x - M.y);
        w *= pow(Var / (Var + (d - M.x) * (d - M.x) + 1e-12), 3.0);
    }
    return w;
}

// The read grid's list for the cell of the surface point P (mesh space, its normal N): false when P is off the surface's cells, or there
// is no grid. The cell is the one of P lifted 5% of a cell along N (surfel_mesh::ReadCellAt): a floor at a multiple of the cell size lies
// on a boundary, and the interpolated position falls a hair to either side of it
bool GIReadCell(in const vec3 P, in const vec3 N, out uint First, out uint Count)
{
    First = 0u; Count = 0u;
    const uint Mask = GIMesh[3];
    if (Mask == 0u) return false;
    const float Cell  = uintBitsToFloat(GIMesh[4]);
    const ivec3 C     = ivec3(floor((P + N * (0.05 * Cell)) / Cell));
    const uvec3 U     = uvec3(C) & 0x1FFFFFu;                       // surfel_grid::Key: 21 bits per axis
    const uint  KeyLo = U.z | (U.y << 21);
    const uint  KeyHi = (U.y >> 11) | (U.x << 10);
    uint h = ((uint(C.x) * 73856093u) ^ (uint(C.y) * 19349663u) ^ (uint(C.z) * 83492791u)) & Mask;
    for (uint n = 0u; n < GIMesh[6]; ++n, h = (h + 1u) & Mask)        // [6]: the table's longest probe
    {
        const uint e = GIMesh[2] + 4u * h;
        if (GIMesh[e + 1u] == 0xFFFFFFFFu) return false;
        if (GIMesh[e] == KeyLo && GIMesh[e + 1u] == KeyHi) { First = GIMesh[e + 2u]; Count = GIMesh[e + 3u]; return true; }
    }
    return false;
}

// The debug view of the surfels: the level 0 surfel with the most weight at P (0xFFFFFFFF: none reaches it), and a color of its own
uint SurfelGIDominant(in const vec3 P, in const vec3 N)
{
    uint First, Count, Best = 0xFFFFFFFFu;
    float BestW = 0.0;
    GIReadCell(P, N, First, Count);
    for (uint i = 0u; i < Count; ++i)
    {
        const uint  s = GIMesh[GIMesh[5] + First + i];
        const uint  b = GIMesh[0] + 8u * s;
        const float c = dot(GISurfelNormal(b), N);
        const float f = max(0.0, 1.0 - length(P - GISurfelCenter(b)) / (2.0 * uintBitsToFloat(GIMesh[b + 3u])));
        const float w = c < 0.5 ? 0.0 : f * f * c;
        if (w > BestW) { BestW = w; Best = s; }
    }
    return Best;
}
vec3 SurfelDebugColor(in const uint s)
{
    uint h = s * 2654435761u; h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
    return vec3(float(h & 255u), float((h >> 8) & 255u), float((h >> 16) & 255u)) / 255.0 * 0.85 + 0.15;
}

// sum(w E), sum(w) of the level 0 surfels that reach a point (mesh space position and normal): its cell of the read grid lists them.
// The weight of the vertex read (falloff over 2 radii squared, times the normal match, times the validity). 0: no grid, or off the
// surface's cells (the caller falls back to the vertex read)
vec4 SurfelGIPixel(in const vec3 P, in const vec3 N, in const uint LightOffset, in const uint Level)
{
    uint First, Count;
    if (!GIReadCell(P, N, First, Count)) return vec4(0.0);
    vec4 Sum = vec4(0.0);
    for (uint i = 0u; i < Count; ++i)
    {
        const uint  s = GIMesh[GIMesh[5] + First + i];
        const uint  b = GIMesh[0] + 8u * s;
        const float c = dot(GISurfelNormal(b), N);
        if (c < 0.5) continue;
        const float f = max(0.0, 1.0 - length(P - GISurfelCenter(b)) / (2.0 * uintBitsToFloat(GIMesh[b + 3u])));
        if (f <= 0.0) continue;
        // Level 1 (far): the fine surfels still find what reaches the pixel, their parents give the light (the same coverage at both
        // levels: no pop at the switch)
        const uint  sl = Level == 1u ? GIMesh[b + 7u] : s;
        const uint  bl = Level == 1u ? GIMesh[1] + 8u * sl : b;
        const vec4  E = GILight[LightOffset + sl];
        const float w = f * f * c * E.w * GIVisibility(P, GISurfelCenter(bl), GISurfelNormal(bl), uintBitsToFloat(GIMesh[bl + 3u]), LightOffset + sl);
        Sum += vec4(E.rgb * w, w);
    }
    return Sum;
}

#endif
