// Spherical harmonics order 1 - the GPU copy of features/gi/xlionrender_sh_l1.h (same constants and order; xlionrender_test_sh_l1.cpp checks
// the convention on the CPU). Coefficients of incident radiance in a fixed WORLD basis; irradiance by the clamped-cosine convolution.
const float sh_y00_v = 0.282094792, sh_y1_v = 0.488602512, sh_pi_v = 3.14159265;
const float sh_a0_v = sh_pi_v, sh_a1_v = 2.0 * sh_pi_v / 3.0;
// The basis at a unit direction: Y00, Y1-1 (y), Y10 (z), Y11 (x)
vec4 ShBasis(in const vec3 w) { return vec4(sh_y00_v, sh_y1_v * w.y, sh_y1_v * w.z, sh_y1_v * w.x); }
// One sample of incident radiance L (rgb) from direction w, of K uniform-sphere samples: (4 pi / K) L Y(w) into each channel's 4 coefficients
void ShAddSample(inout vec4 R, inout vec4 G, inout vec4 B, in const vec3 L, in const vec3 w, in const float K)
{
    const vec4 Y = ShBasis(w) * (4.0 * sh_pi_v / K);
    R += L.r * Y; G += L.g * Y; B += L.b * Y;
}
// Irradiance for the unit normal n, clamped once at the end
vec3 ShIrradiance(in const vec4 R, in const vec4 G, in const vec4 B, in const vec3 n)
{
    const vec4 Y = ShBasis(n) * vec4(sh_a0_v, sh_a1_v, sh_a1_v, sh_a1_v);
    return max(vec3(dot(R, Y), dot(G, Y), dot(B, Y)), vec3(0.0));
}
