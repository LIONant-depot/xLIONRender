// The movers' irradiance probes' radiometric contract (dynamic_objects_gi_plan_2026-11-13_v3.md, milestone A), on the CPU:
//  1. a constant environment L gives E = pi L for every normal
//  2. a first-order field L(w) = a + b (w . d) gives E(n) = pi a + (2 pi / 3) b (n . d) - signs, axis order, normalization, convolution
//  3. a fixed world field, the probe's object turned: world-basis coefficients are not rotated, a fixed world normal reads the same
//  4. Monte Carlo over K uniform-sphere rays converges to (1) and (2)
#include "features/gi/xlionrender_sh_l1.h"
#include <cstdio>
#include <random>

namespace sh = xlionrender::sh;
static int g_Fail = 0;
#define CHECK(c, ...) do { if( !(c) ) { ++g_Fail; std::printf("FAIL: " __VA_ARGS__); std::printf("\n"); } } while(0)

static void Dir( std::mt19937& R, float& x, float& y, float& z )        // uniform on the sphere
{
    std::uniform_real_distribution<float> U(0.0f, 1.0f);
    const float u = 2.0f * U(R) - 1.0f, p = 2.0f * sh::pi_v * U(R), s = std::sqrt(1.0f - u * u);
    x = s * std::cos(p); y = s * std::sin(p); z = u;
}

int main()
{
    const float N[6][3] = { {1,0,0}, {-1,0,0}, {0,1,0}, {0,-1,0}, {0,0,1}, {0.577f,0.577f,0.577f} };
    // 1. constant environment, exact coefficients (L00 = 4 pi Y00 L)
    {
        sh::l1 S; S.m_C[0] = 4.0f * sh::pi_v * sh::y00_v * 2.0f;
        for( auto& n : N ) { const float E = sh::Irradiance(S, n[0], n[1], n[2]); CHECK(std::abs(E - sh::pi_v * 2.0f) < 1e-4f, "constant: E %f, want %f", E, sh::pi_v * 2.0f); }
    }
    // 2. first-order field, exact coefficients: L = a + b (w . d), d = (0.6, 0, 0.8)
    {
        const float a = 1.5f, b = 0.7f, d[3] = { 0.6f, 0.0f, 0.8f };
        sh::l1 S; S.m_C[0] = 4.0f * sh::pi_v * sh::y00_v * a;
        // the projection of (w . d) on Y1m is (4 pi / 3) sh::y1_v d_m (the integral of w_i w_j over the sphere is 4 pi / 3 delta_ij)
        S.m_C[1] = 4.0f * sh::pi_v / 3.0f * sh::y1_v * b * d[1]; S.m_C[2] = 4.0f * sh::pi_v / 3.0f * sh::y1_v * b * d[2]; S.m_C[3] = 4.0f * sh::pi_v / 3.0f * sh::y1_v * b * d[0];
        for( auto& n : N )
        {
            const float nd = n[0] * d[0] + n[1] * d[1] + n[2] * d[2];
            const float Want = sh::pi_v * a + 2.0f * sh::pi_v / 3.0f * b * nd, E = sh::Irradiance(S, n[0], n[1], n[2]);
            CHECK(std::abs(E - Want) < 1e-3f, "first order: n (%g %g %g) E %f, want %f", n[0], n[1], n[2], E, Want);
        }
    }
    // 4. Monte Carlo from uniform-sphere rays converges to both
    {
        std::mt19937 R(7); const int K = 200000; const float a = 1.5f, b = 0.7f, d[3] = { 0.6f, 0.0f, 0.8f };
        sh::l1 S;
        for( int k = 0; k < K; ++k ) { float x, y, z; Dir(R, x, y, z); sh::AddSample(S, a + b * (x * d[0] + y * d[1] + z * d[2]), x, y, z, K); }
        for( auto& n : N )
        {
            const float nd = n[0] * d[0] + n[1] * d[1] + n[2] * d[2];
            const float Want = sh::pi_v * a + 2.0f * sh::pi_v / 3.0f * b * nd, E = sh::Irradiance(S, n[0], n[1], n[2]);
            CHECK(std::abs(E - Want) < 0.02f * Want, "monte carlo: n (%g %g %g) E %f, want %f", n[0], n[1], n[2], E, Want);
        }
    }
    // 3. a fixed world field, the object turned through 360 degrees: the world-basis coefficients are what the probe sampled at its
    //    position - turning the object changes the position (here: none, a probe at the object's centre) and not the coefficients
    {
        const float a = 1.0f, b = 0.9f, d[3] = { 1.0f, 0.0f, 0.0f };
        sh::l1 S; S.m_C[0] = 4.0f * sh::pi_v * sh::y00_v * a; S.m_C[3] = 4.0f * sh::pi_v / 3.0f * sh::y1_v * b * d[0];
        const float E0 = sh::Irradiance(S, 1.0f, 0.0f, 0.0f);
        for( int deg = 0; deg < 360; deg += 30 )
        {
            // the object's own normal (local +x) turned by deg about +y: the world normal changes, the field does not
            const float r = float(deg) * sh::pi_v / 180.0f, nx = std::cos(r), nz = -std::sin(r);
            const float E = sh::Irradiance(S, nx, 0.0f, nz), Want = sh::pi_v * a + 2.0f * sh::pi_v / 3.0f * b * nx;
            CHECK(std::abs(E - std::max(Want, 0.0f)) < 1e-3f, "rotation %d: E %f, want %f", deg, E, Want);
        }
        CHECK(std::abs(sh::Irradiance(S, 1.0f, 0.0f, 0.0f) - E0) < 1e-6f, "a fixed world normal reads the same");
    }
    std::printf(g_Fail ? "FAIL: %d\n" : "PASS\n", g_Fail);
    return g_Fail ? 1 : 0;
}
