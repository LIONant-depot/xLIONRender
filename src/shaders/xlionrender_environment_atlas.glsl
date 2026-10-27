// The environment probe's atlas (features/environment, instance::Init creates it): the octahedral sky prefiltered for kEnvLevels
// roughness levels side by side in one RGBA16F texture, each with a kEnvBorder texel border folded over the octahedral edges so
// bilinear filtering never reads another level. Level k is for roughness k / (kEnvLevels - 1).
//   level 0: 256 at (0, 0); 1: 128 at (256, 0); 2: 64 at (256, 128); 3: 32 at (320, 128); 4: 16 at (352, 128); 5: 8 at (368, 128)
#include "xlionrender_octahedral.glsl"

const int   kEnvLevels  = 6;
const float kEnvBorder  = 2.0;
const vec2  kEnvAtlas   = vec2(384.0, 256.0);

// x, y: the level's origin in texels, z: its size
vec3 EnvLevelRect(in const int Level)
{
    if (Level == 0) return vec3(0.0, 0.0, 256.0);
    if (Level == 1) return vec3(256.0, 0.0, 128.0);
    const float Size = 256.0 / float(1 << Level);
    float X = 256.0;
    for (int i = 2; i < Level; ++i) X += 256.0 / float(1 << i);
    return vec3(X, 128.0, Size);
}

// A direction's texture coordinate in a level (bilinear inside its border)
vec2 EnvLevelUV(in const int Level, in const vec3 Dir)
{
    const vec3 R     = EnvLevelRect(Level);
    const float Inner = R.z - 2.0 * kEnvBorder;
    return (R.xy + kEnvBorder + (OctEncode(Dir) * 0.5 + 0.5) * Inner) / kEnvAtlas;
}

// A direction's prefiltered radiance at a roughness: the two levels around it, blended (the sky's atlas, a probe's slot)
vec3 AtlasSample(in const sampler2D Atlas, in const vec2 Origin, in const vec2 AtlasSize, in const vec3 Dir, in const float PerceptualRoughness)
{
    const float L  = clamp(PerceptualRoughness, 0.0, 1.0) * float(kEnvLevels - 1);
    const int   L0 = int(floor(L));
    const int   L1 = min(L0 + 1, kEnvLevels - 1);
    const vec2  S  = kEnvAtlas / AtlasSize;                  // a level's uv in the sky layout -> the atlas' uv
    const vec3  A  = textureLod(Atlas, Origin / AtlasSize + EnvLevelUV(L0, Dir) * S, 0.0).rgb;
    const vec3  B  = textureLod(Atlas, Origin / AtlasSize + EnvLevelUV(L1, Dir) * S, 0.0).rgb;
    return mix(A, B, L - float(L0));
}
