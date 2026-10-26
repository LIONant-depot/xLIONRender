// A local light's geometry terms, one definition for the shaders and the CPU (documentation/lighting_strategy.md 6.5, the
// EvaluateLightGeometry part: shared physics, not sampling decisions). Forward shading (xlionrender_local_lights.glsl) and the
// light cells' importance (features/lights/xlionrender_light_cells.h) read it, so they and the references cannot drift apart.
// GLSL and C++ both compile this file (C++: namespace xlionrender::light_eval).
#ifndef XLIONRENDER_LIGHT_EVAL_GLSL
#define XLIONRENDER_LIGHT_EVAL_GLSL

#ifdef __cplusplus
#include <algorithm>
namespace xlionrender::light_eval
{
    inline float smoothstep(float Edge0, float Edge1, float x) noexcept      // GLSL's
    {
        const float t = std::clamp((x - Edge0) / (Edge1 - Edge0), 0.0f, 1.0f);
        return t * t * (3.0f - 2.0f * t);
    }
#define XLIONRENDER_LIGHT_EVAL_FN inline
#else
#define XLIONRENDER_LIGHT_EVAL_FN
#endif

// Scale x the falloff at s = distance / radius, for s in [0, 1]: Scale at the light, 0 at its radius (the callers skip s >= 1). The
// falloff is unitless (Scale 1: the factor on the light's color x intensity; the cells pass the light's power). The main light's
// Attenuate without its area term. Scale is multiplied first, as the cells always did: their importance stays bitwise what it was
XLIONRENDER_LIGHT_EVAL_FN float LightDistanceFalloff(float s, float Scale)
{
    return Scale * (1.0f - s * s * s * s) / (1.0f + 2.0f * s * s);
}

// A spot's cone at CosAngle (the cosine between its axis and the direction to the point): 0 at and beyond the outer angle, 1 inside
// the inner one, smoothstep between. Unitless
XLIONRENDER_LIGHT_EVAL_FN float LightSpotCone(float CosAngle, float CosOuter, float CosInner)
{
    return smoothstep(CosOuter, CosInner, CosAngle);
}

#ifdef __cplusplus
}
#endif
#undef XLIONRENDER_LIGHT_EVAL_FN
#endif
