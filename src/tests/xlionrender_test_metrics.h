// The image metrics of documentation/reflections_strategy.md sec. 11 (RGB, linear HDR, 3 floats per pixel; a mask picks the region):
//  - RMAE = mean |a - r| / max(mean r, Floor): dark pixels included; Floor an absolute radiance (a leak into a dark region counts)
//  - DarkLeak = over the pixels with lum(r) < Floor: mean max(0, lum(a) - lum(r)) / Floor (light where the reference has none)
//  - Flicker = mean |E_t - E_t-1| of the error E = a - r (fixed camera: the correspondence is the pixel)
//  - Bias / noise from repeated independent runs: the runs' mean against the reference, and the runs' spread
// SelfTestMetrics() checks each on synthetic images (a test calls it before trusting the numbers)
#ifndef XLIONRENDER_TEST_METRICS_H
#define XLIONRENDER_TEST_METRICS_H
#include <vector>
#include <span>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <algorithm>

namespace xlionrender::test
{
    using rgb_image = std::vector<float>;                  // 3 floats per pixel
    using mask      = std::vector<std::uint8_t>;           // 1 per pixel: in the region

    inline float Lum(const float* p) noexcept { return 0.2126f * p[0] + 0.7152f * p[1] + 0.0722f * p[2]; }

    inline double RMAE(std::span<const float> A, std::span<const float> R, const mask& M, double Floor) noexcept
    {
        double Err = 0, Ref = 0; std::size_t n = 0;
        for (std::size_t i = 0; i < M.size(); ++i)
        {
            if (!M[i]) continue;
            for (int c = 0; c < 3; ++c) { Err += std::abs(double(A[i * 3 + c]) - R[i * 3 + c]); Ref += R[i * 3 + c]; }
            n += 3;
        }
        return n ? (Err / double(n)) / std::max(Ref / double(n), Floor) : 0.0;
    }

    inline double DarkLeak(std::span<const float> A, std::span<const float> R, const mask& M, double Floor) noexcept
    {
        double Leak = 0; std::size_t n = 0;
        for (std::size_t i = 0; i < M.size(); ++i)
        {
            if (!M[i] || Lum(&R[i * 3]) >= Floor) continue;
            Leak += std::max(0.0, double(Lum(&A[i * 3])) - Lum(&R[i * 3]));
            ++n;
        }
        return n ? Leak / double(n) / Floor : 0.0;
    }

    // The temporal change of the error between two frames (fixed camera), relative to the reference's mean
    inline double Flicker(std::span<const float> A1, std::span<const float> R1, std::span<const float> A0, std::span<const float> R0, const mask& M, double Floor) noexcept
    {
        double Ch = 0, Ref = 0; std::size_t n = 0;
        for (std::size_t i = 0; i < M.size(); ++i)
        {
            if (!M[i]) continue;
            for (int c = 0; c < 3; ++c) { const std::size_t k = i * 3 + c; Ch += std::abs((double(A1[k]) - R1[k]) - (double(A0[k]) - R0[k])); Ref += R1[k]; }
            n += 3;
        }
        return n ? (Ch / double(n)) / std::max(Ref / double(n), Floor) : 0.0;
    }

    // Repeated runs: Bias = RMAE of their mean; Noise = the mean per-pixel standard deviation over the same normalization
    struct bias_noise { double m_Bias = 0, m_Noise = 0; };
    inline bias_noise BiasNoise(const std::vector<rgb_image>& Runs, std::span<const float> R, const mask& M, double Floor) noexcept
    {
        rgb_image Mean(R.size(), 0.0f);
        for (const auto& I : Runs) for (std::size_t k = 0; k < Mean.size(); ++k) Mean[k] += I[k] / float(Runs.size());
        double Sd = 0, Ref = 0; std::size_t n = 0;
        for (std::size_t i = 0; i < M.size(); ++i)
        {
            if (!M[i]) continue;
            for (int c = 0; c < 3; ++c)
            {
                const std::size_t k = i * 3 + c;
                double V = 0;
                for (const auto& I : Runs) V += (double(I[k]) - Mean[k]) * (double(I[k]) - Mean[k]);
                Sd += std::sqrt(V / double(std::max<std::size_t>(Runs.size() - 1, 1))); Ref += R[k];
            }
            n += 3;
        }
        return { RMAE(Mean, R, M, Floor), n ? (Sd / double(n)) / std::max(Ref / double(n), Floor) : 0.0 };
    }

    // With XLIONRENDER_TEST_DUMP=<folder>: <folder>/<Name>.ppm (x / (1 + x) per channel, 8 bits)
    inline void DumpRGB(const char* pName, std::span<const float> I, int W, int H) noexcept
    {
        const char* pDir = std::getenv("XLIONRENDER_TEST_DUMP");
        if (pDir == nullptr) return;
        if (FILE* f = std::fopen((std::string(pDir) + "/" + pName + ".ppm").c_str(), "wb"))
        {
            std::fprintf(f, "P6 %d %d 255\n", W, H);
            for (std::size_t k = 0; k < std::size_t(W) * H * 3; ++k) { const float v = std::max(I[k], 0.0f); std::fputc(int(255.0f * v / (1.0f + v)), f); }
            std::fclose(f);
        }
    }

    // Each metric on images whose answer is known; returns the first failure's text (empty: all pass)
    inline std::string SelfTestMetrics() noexcept
    {
        const std::size_t n = 64;
        mask All(n, 1);
        rgb_image R(n * 3), A(n * 3);
        for (std::size_t k = 0; k < R.size(); ++k) R[k] = 0.5f + 0.25f * float(k % 7);
        if (RMAE(R, R, All, 1e-3) != 0) return "RMAE of an image against itself is not 0";
        for (std::size_t k = 0; k < A.size(); ++k) A[k] = R[k] * 1.1f;
        if (std::abs(RMAE(A, R, All, 1e-3) - 0.1) > 1e-5) return "RMAE of a 10% brighter image is not 0.1";
        // A dark region: the floor normalizes it, the leak sees light where the reference has none, not a darker answer
        rgb_image D(n * 3, 0.0f), L(n * 3, 0.0f);
        for (std::size_t k = 0; k < L.size(); ++k) L[k] = 0.002f;
        if (std::abs(RMAE(L, D, All, 0.01) - 0.2) > 1e-5) return "RMAE in a dark region is not normalized by the floor";
        if (std::abs(DarkLeak(L, D, All, 0.01) - 0.2) > 1e-4) return "DarkLeak does not see light in a dark region";
        if (DarkLeak(D, L, All, 0.01) != 0) return "DarkLeak counts a darker answer";
        if (DarkLeak(A, R, All, 0.01) != 0) return "DarkLeak counts bright reference pixels";
        // Flicker: a constant error does not flicker; an error that changes sign does, though both frames' mean error is the same
        if (Flicker(A, R, A, R, All, 1e-3) != 0) return "Flicker of a constant error is not 0";
        rgb_image B(n * 3);
        for (std::size_t k = 0; k < B.size(); ++k) B[k] = R[k] * ((k / 3) % 2 ? 1.1f : 0.9f);
        rgb_image C(n * 3);
        for (std::size_t k = 0; k < C.size(); ++k) C[k] = R[k] * ((k / 3) % 2 ? 0.9f : 1.1f);
        if (Flicker(C, R, B, R, All, 1e-3) < 0.15) return "Flicker misses an error that moves between pixels";
        // Bias vs noise: unbiased noise averages out over runs, a constant offset does not
        std::vector<rgb_image> Noisy, Offset;
        std::uint32_t s = 12345;
        const auto Rand = [&] { s = s * 1664525u + 1013904223u; return float(s >> 8) / float(1u << 24) - 0.5f; };
        for (int r = 0; r < 64; ++r)
        {
            rgb_image I(n * 3), O(n * 3);
            for (std::size_t k = 0; k < I.size(); ++k) { const float e = 0.4f * Rand(); I[k] = R[k] * (1 + e); O[k] = R[k] * (1.05f + e); }
            Noisy.push_back(std::move(I)); Offset.push_back(std::move(O));
        }
        const auto BN = BiasNoise(Noisy, R, All, 1e-3), BO = BiasNoise(Offset, R, All, 1e-3);
        if (!(BN.m_Bias < 0.03 && BN.m_Noise > 0.08)) return "BiasNoise: unbiased noise reads as bias";
        if (!(std::abs(BO.m_Bias - 0.05) < 0.03)) return "BiasNoise misses a 5% offset";
        return {};
    }
}

#endif
