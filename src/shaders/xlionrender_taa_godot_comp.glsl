#version 450
// C22 diagnostic (XLION_TAA_GODOT=1): Godot's TAA resolve (servers/rendering/renderer_rd/shaders/effects/taa_resolve.glsl), ported as a drop-in
// for xlionrender_taa_comp.glsl - the same inputs (this frame's HDR and depth, the history, the camera's reprojection), Godot's resolve: Catmull-Rom 9,
// the velocity of the nearest depth in the 3 x 3, variance clipping toward the centre with a box that shrinks with speed, 1/16 blend, Reinhard,
// anti-flicker. If it still shakes, the inputs are at fault; if not, our resolve is. Differences: the velocity is computed here from the depth and
// the camera (camera-only scene: exact), and there is no velocity disocclusion (no last velocity buffer).
//
// Copyright(c) 2016-2022 Panos Karabelas (Spartan Engine), 2020 Joan Fons (Vulkan / Godot), 2025 Jakub Brzyski (dynamic variance). MIT licence:
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and / or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to
// the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions
// of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
layout(local_size_x = 8, local_size_y = 8) in;
layout(set = 1, binding = 0) uniform sampler2D HDR;
layout(set = 1, binding = 1) uniform sampler2D Depth;
layout(set = 1, binding = 2) uniform sampler2D History;
layout(set = 1, binding = 3) uniform sampler2D Motion;      // (unused here)
layout(set = 2, binding = 0, rgba16f) writeonly uniform image2D Dst;
layout(set = 2, binding = 1, rgba16f) writeonly uniform image2D Dbg;      // (unused here: the layout of xlionrender_taa_comp.glsl)
layout(push_constant) uniform push_t { mat4 m_Reproject; int m_W, m_H; float m_Blend, m_bHistory; vec2 m_Jitter; float m_GammaSlow, m_GammaFast; int m_Debug, m_Pad0, m_Pad1, m_Pad2; vec2 m_PrevJitter; float m_bMotion, m_Pad3; } Push;

#define FLT_MIN 0.00000001
#define FLT_MAX 32767.0
#define RPC_9 0.11111111111
#define RPC_16 0.0625

vec2 Res() { return vec2(Push.m_W, Push.m_H); }
vec3 LoadColor(ivec2 q) { const vec3 c = texelFetch(HDR, clamp(q, ivec2(0), ivec2(Push.m_W, Push.m_H) - 1), 0).rgb; return all(lessThan(c, vec3(1e5))) ? max(c, vec3(0.0)) : vec3(0.0); }
float LoadDepth(ivec2 q) { return texelFetch(Depth, clamp(q, ivec2(0), ivec2(Push.m_W, Push.m_H) - 1), 0).r; }

// Godot's velocity is "uv of last frame minus uv of this frame"; here from the depth and the camera (with the jitter put back, as our TAA)
vec2 Velocity(ivec2 q)
{
    const vec2 uv = (vec2(q) + 0.5) / Res();
    const vec4 Prev = Push.m_Reproject * vec4(uv * 2.0 - 1.0, LoadDepth(q), 1.0);
    if (Prev.w <= 0.0) return vec2(0.0);
    return Prev.xy / Prev.w * 0.5 + 0.5 + Push.m_Jitter - uv;
}

vec3 reinhard(vec3 hdr) { return hdr / (hdr + 1.0); }
vec3 reinhard_inverse(vec3 sdr) { return sdr / (1.0 - sdr); }

vec2 ClosestVelocity3x3(ivec2 p)      // (the velocity of the nearest depth in the 3 x 3: standard depth here, the smallest is the nearest)
{
    float MinD = 1e30; ivec2 MinP = p;
    for (int y = -1; y <= 1; ++y) for (int x = -1; x <= 1; ++x) { const float d = LoadDepth(p + ivec2(x, y)); if (d < MinD) { MinD = d; MinP = p + ivec2(x, y); } }
    return Velocity(MinP);
}

vec3 sample_catmull_rom_9(vec2 uv, vec2 resolution)
{
    // Source: https://gist.github.com/TheRealMJP/c83b8c0f46b63f3a88a5986f4fa982b1 (as Godot's)
    vec2 sample_pos = uv * resolution;
    vec2 texPos1 = floor(sample_pos - 0.5f) + 0.5f;
    vec2 f = sample_pos - texPos1;
    vec2 w0 = f * (-0.5f + f * (1.0f - 0.5f * f));
    vec2 w1 = 1.0f + f * f * (-2.5f + 1.5f * f);
    vec2 w2 = f * (0.5f + f * (2.0f - 1.5f * f));
    vec2 w3 = f * f * (-0.5f + 0.5f * f);
    vec2 w12 = w1 + w2;
    vec2 offset12 = w2 / (w1 + w2);
    vec2 texPos0 = (texPos1 - 1.0f) / resolution;
    vec2 texPos3 = (texPos1 + 2.0f) / resolution;
    vec2 texPos12 = (texPos1 + offset12) / resolution;
    vec3 result = vec3(0.0f);
    result += textureLod(History, vec2(texPos0.x, texPos0.y), 0.0).xyz * w0.x * w0.y;
    result += textureLod(History, vec2(texPos12.x, texPos0.y), 0.0).xyz * w12.x * w0.y;
    result += textureLod(History, vec2(texPos3.x, texPos0.y), 0.0).xyz * w3.x * w0.y;
    result += textureLod(History, vec2(texPos0.x, texPos12.y), 0.0).xyz * w0.x * w12.y;
    result += textureLod(History, vec2(texPos12.x, texPos12.y), 0.0).xyz * w12.x * w12.y;
    result += textureLod(History, vec2(texPos3.x, texPos12.y), 0.0).xyz * w3.x * w12.y;
    result += textureLod(History, vec2(texPos0.x, texPos3.y), 0.0).xyz * w0.x * w3.y;
    result += textureLod(History, vec2(texPos12.x, texPos3.y), 0.0).xyz * w12.x * w3.y;
    result += textureLod(History, vec2(texPos3.x, texPos3.y), 0.0).xyz * w3.x * w3.y;
    return max(result, 0.0f);
}

vec3 clip_aabb(vec3 aabb_min, vec3 aabb_max, vec3 p, vec3 q)
{
    vec3 r = q - p;
    vec3 rmax = (aabb_max - p.xyz);
    vec3 rmin = (aabb_min - p.xyz);
    if (r.x > rmax.x + FLT_MIN) r *= (rmax.x / r.x);
    if (r.y > rmax.y + FLT_MIN) r *= (rmax.y / r.y);
    if (r.z > rmax.z + FLT_MIN) r *= (rmax.z / r.z);
    if (r.x < rmin.x - FLT_MIN) r *= (rmin.x / r.x);
    if (r.y < rmin.y - FLT_MIN) r *= (rmin.y / r.y);
    if (r.z < rmin.z - FLT_MIN) r *= (rmin.z / r.z);
    return p + r;
}

vec3 clip_history_3x3(ivec2 p, vec3 color_history, vec2 velocity_closest)
{
    vec3 color_avg = vec3(0.0), color_avg2 = vec3(0.0);
    for (int y = -1; y <= 1; ++y) for (int x = -1; x <= 1; ++x) { const vec3 s = LoadColor(p + ivec2(x, y)); color_avg += s; color_avg2 += s * s; }
    color_avg *= RPC_9; color_avg2 *= RPC_9;
    // Godot's variance_dynamic: clamp(1.1 * 1080 / height, 0.75, 1.0)
    const float variance_dynamic = clamp(1.1 * 1080.0 / float(Push.m_H), 0.75, 1.0);
    float box_size = mix(0.0f, variance_dynamic, smoothstep(0.02f, 0.0f, length(velocity_closest)));
    vec3 dev = sqrt(abs(color_avg2 - (color_avg * color_avg))) * box_size;
    vec3 color_min = color_avg - dev;
    vec3 color_max = color_avg + dev;
    vec3 color = clip_aabb(color_min, color_max, clamp(color_avg, color_min, color_max), color_history);
    return clamp(color, FLT_MIN, FLT_MAX);
}

float luminance(vec3 color) { return max(dot(color, vec3(0.299f, 0.587f, 0.114f)), 0.0001f); }

void main()
{
    const ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (p.x >= Push.m_W || p.y >= Push.m_H) return;
    const vec4 Here = texelFetch(HDR, p, 0);
    vec3 color_input = LoadColor(p);
    if (Push.m_bHistory < 0.5) { imageStore(Dst, p, vec4(color_input, Here.a)); return; }
    const vec2 uv = (vec2(p) + 0.5) / Res();
    const vec2 velocity = Velocity(p);
    const vec2 uv_reprojected = uv + velocity;
    vec3 color_history = sample_catmull_rom_9(uv_reprojected, Res());
    color_history = clip_history_3x3(p, color_history, ClosestVelocity3x3(p));
    float blend_factor = RPC_16;
    const float factor_screen = any(lessThan(uv_reprojected, vec2(0.0))) || any(greaterThan(uv_reprojected, vec2(1.0))) ? 1.0 : 0.0;
    blend_factor = clamp(blend_factor + factor_screen, 0.0, 1.0);      // (no velocity disocclusion: no last velocity buffer)
    color_history = reinhard(color_history);
    color_input = reinhard(color_input);
    const float lum_color = luminance(color_input), lum_history = luminance(color_history);
    float diff = abs(lum_color - lum_history) / max(lum_color, max(lum_history, 1.001));
    diff = 1.0 - diff; diff = diff * diff;
    blend_factor = mix(0.0, blend_factor, diff);
    const vec3 color_resolved = reinhard_inverse(mix(color_history, color_input, blend_factor));
    imageStore(Dst, p, vec4(color_resolved, Here.a));
}
