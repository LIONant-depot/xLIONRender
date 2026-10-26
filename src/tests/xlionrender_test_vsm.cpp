// The sun's virtual shadow pages (features/shadows/xlionrender_shadow_vsm.h; documentation/lighting_strategy.md P1a):
//  P1a-0 (CPU): addressing, the table, admission and eviction under pressure, protected and pinned slots, slot reuse and generations
//  (B13), epochs and invalidations; every published table checked against invariant 1 (Validate)
//  P1a-1 (GPU): a ground, spheres and a thin pole under the sun, the pages on:
//   - they settle: every pixel shaded by its own level's pages; the shadow is there; invariant 1 every frame
//   - the cached pages equal a full redraw (no cache, a budget for everything): bitwise
//   - B12 boundaries: against the GI's rays on the same triangles, pixels whose filter crosses a page border disagree no more than the rest
//   - B11 overflow: a small pool - rejections, pixels shaded by a coarser level, none by no level (resolution lost, no hole)
//   - B7 teleport: the recovery per frame (pixels by their own level / a coarser one / none) after a jump to unvisited ground
#include "xlionrender_test_common.h"
#include "features/shadows/xlionrender_shadow_vsm.h"
#include "features/gi/xlionrender_surfel_gi.h"

using namespace xlionrender::test;
using vsm = xlionrender::shadows::vsm;

namespace
{
    bool MakeMesh(xlionrender::instance& Renderer, const xprim_geom::mesh& Prim, xlionrender::mesh_id& Id) noexcept
    {
        std::vector<xlionrender::vertex> Verts(Prim.m_Vertices.size());
        for (std::size_t i = 0; i < Verts.size(); ++i)
        {
            const auto& V = Prim.m_Vertices[i];
            Verts[i] = { { V.m_Position.m_X, V.m_Position.m_Y, V.m_Position.m_Z }, { V.m_Texcoord.m_X, V.m_Texcoord.m_Y }, 0xffffffffu
                       , { V.m_Normal.m_X, V.m_Normal.m_Y, V.m_Normal.m_Z }, { V.m_Tangent.m_X, V.m_Tangent.m_Y, V.m_Tangent.m_Z, V.m_Tangent.m_D } };
        }
        return Renderer.CreateMesh(Id, Verts, Prim.m_Indices) == nullptr;
    }

    xlionrender::gi::surfel_mesh RayMesh(const xprim_geom::mesh& Prim)
    {
        xlionrender::gi::surfel_mesh M;
        for (const auto& V : Prim.m_Vertices) { M.m_Pos.push_back({ V.m_Position.m_X, V.m_Position.m_Y, V.m_Position.m_Z }); M.m_Nrm.push_back({ V.m_Normal.m_X, V.m_Normal.m_Y, V.m_Normal.m_Z }); }
        for (std::size_t i = 0; i + 2 < Prim.m_Indices.size(); i += 3) M.m_Tris.push_back({ int(Prim.m_Indices[i]), int(Prim.m_Indices[i + 1]), int(Prim.m_Indices[i + 2]) });
        M.Bake(0.08f, 0.25f);
        return M;
    }

    // One frame of the CPU side: Update then the table, checked
    bool Frame(vsm& V, std::uint64_t F, std::span<const vsm::key> Requests, const std::vector<xmath::fbbox>& Changed = {}, xmath::fvec3 Camera = { 0, 0, 0 }
              , xmath::fvec3 Dir = { 0.3f, -1, 0.2f }, float Texel = 1.0f / 128)
    {
        V.Update(Dir, Texel, Camera, Changed, true, Requests, F);
        std::vector<std::uint32_t> T;
        V.BuildTable(T);
        return V.Validate(T) && V.m_Stats.m_PinnedNotValid == 0;
    }

    int TableCount(const vsm& V)
    {
        std::vector<std::uint32_t> T;
        V.BuildTable(T);
        int n = 0;
        for (auto u : T) n += ((u & 0xFFFF) != 0) + ((u >> 16) != 0);
        return n;
    }
}

int main()
{
    // Addressing: a page's W2C maps its rect to [-1,1]; keys pack uniquely with negative indices; toroidal entries round trip
    {
        vsm V;
        V.Init({});
        CHECK(Frame(V, 1, {}), "first frame");
        const vsm::key K{ 2, -7, 5 };
        const float P = V.PageSize(2);
        const auto M = V.PageW2C(K);
        const auto Corner = [&](float u, float v)          // the world point at the page's light-space (u, v), on the depth centre
        {
            return V.m_X * ((K.m_X + u) * P) + V.m_Y * ((K.m_Y + v) * P) + V.m_Z * V.m_Levels[2].m_DepthCenter;
        };
        const auto A = M * xmath::fvec4(Corner(0, 0), 1), B = M * xmath::fvec4(Corner(1, 1), 1);
        CHECK(std::abs(A.m_X + 1) < 1e-3f && std::abs(A.m_Y + 1) < 1e-3f && std::abs(B.m_X - 1) < 1e-3f && std::abs(B.m_Y - 1) < 1e-3f && std::abs(A.m_Z - 0.5f) < 1e-3f
             , "PageW2C: (%f %f %f) .. (%f %f)", A.m_X, A.m_Y, A.m_Z, B.m_X, B.m_Y);
        CHECK(vsm::Pack({ 0, -1, 0 }) != vsm::Pack({ 0, 0, -1 }) && vsm::Pack({ 1, 3, 3 }) != vsm::Pack({ 2, 3, 3 }), "Pack collides");
        const auto& Lv = V.m_Levels[0];
        std::vector<int> Seen(std::size_t(V.getSetup().m_Window) * V.getSetup().m_Window, 0);
        for (int y = Lv.m_OriginY; y < Lv.m_OriginY + V.getSetup().m_Window; ++y)
        for (int x = Lv.m_OriginX; x < Lv.m_OriginX + V.getSetup().m_Window; ++x)
        {
            CHECK(V.Unwrap(V.Torus(x), Lv.m_OriginX) == x, "Torus/Unwrap at %d", x);
            ++Seen[std::size_t(V.TableIndex({ 0, x, y }))];
        }
        for (int n : Seen) CHECK(n == 1, "two window pages share a table entry");
    }

    // The net on the first frame: every pinned page rendered and VALID, the table holds exactly them
    vsm V;
    V.Init({});
    const int Pinned = V.getSetup().PinnedSlots();
    CHECK(Frame(V, 1, {}), "net frame");
    CHECK(V.m_Stats.m_Epochs == 1 && V.m_Stats.m_RenderedPinned == Pinned && V.m_Stats.m_ResidentPinned == Pinned && V.m_Stats.m_Rendered == 0
         , "the net: epoch %d, rendered %d of %d", V.m_Stats.m_Epochs, V.m_Stats.m_RenderedPinned, Pinned);
    CHECK(TableCount(V) == Pinned, "table %d entries, not the net's %d", TableCount(V), Pinned);
    CHECK(Frame(V, 2, {}) && V.m_Stats.m_RenderedPinned == 0, "a second frame re-rendered %d net pages", V.m_Stats.m_RenderedPinned);

    // Requests: admitted, rendered within the budget, then reused
    const auto& L0 = V.m_Levels[0];
    std::vector<vsm::key> Req;
    for (int i = 0; i < 100; ++i) Req.push_back({ 0, L0.m_OriginX + i % 32, L0.m_OriginY + i / 32 });
    CHECK(Frame(V, 3, Req), "requests");
    CHECK(V.m_Stats.m_Requested == 100 && V.m_Stats.m_Rendered == V.getSetup().m_Budget && V.m_Stats.m_Deferred == 100 - V.getSetup().m_Budget
         , "budget: requested %d rendered %d deferred %d", V.m_Stats.m_Requested, V.m_Stats.m_Rendered, V.m_Stats.m_Deferred);
    CHECK(Frame(V, 4, Req) && V.m_Stats.m_Reused == V.getSetup().m_Budget && V.m_Stats.m_Rendered == 100 - V.getSetup().m_Budget, "the rest next frame (reused %d rendered %d)", V.m_Stats.m_Reused, V.m_Stats.m_Rendered);
    CHECK(Frame(V, 5, Req) && V.m_Stats.m_Reused == 100 && V.m_Stats.m_Rendered == 0, "settled: reused %d rendered %d", V.m_Stats.m_Reused, V.m_Stats.m_Rendered);

    // A caster change: the pages under it leave the table at once (DIRTY), come back when rendered again
    {
        const vsm::key K = Req[5];
        const float P = V.PageSize(0);
        const xmath::fvec3 Center = V.m_X * ((K.m_X + 0.5f) * P) + V.m_Y * ((K.m_Y + 0.5f) * P);
        const std::vector<xmath::fbbox> Changed{ xmath::fbbox{ Center - xmath::fvec3(0.01f), Center + xmath::fvec3(0.01f) } };
        V.Update({ 0.3f, -1, 0.2f }, 1.0f / 128, { 0, 0, 0 }, Changed, true, {}, 6);       // no requests this frame: not re-rendered
        std::vector<std::uint32_t> T;
        V.BuildTable(T);
        CHECK(V.Validate(T) && !V.IsValid(K) && V.m_Stats.m_Dirtied >= 1, "a changed caster's page stayed VALID (dirtied %d)", V.m_Stats.m_Dirtied);
        CHECK(Frame(V, 7, Req) && V.IsValid(K) && V.m_Stats.m_Rendered >= 1, "the dirty page was not rendered again when requested");
    }

    // Pressure: a small pool. Rejections are counted, the net is never evicted, pages requested lately are never evicted
    {
        vsm S;
        S.Init({ .m_PoolTexels = 2048, .m_Levels = 6, .m_PinnedRadius = 4, .m_Budget = 1000 });         // 256 slots: 64 net, 192 fine
        const int Fine = S.getSetup().Slots() - S.getSetup().PinnedSlots();
        CHECK(Frame(S, 1, {}), "small pool net");
        std::vector<vsm::key> Many;
        const auto& O = S.m_Levels[0];
        for (int i = 0; i < 300; ++i) Many.push_back({ 0, O.m_OriginX + i % 32, O.m_OriginY + (i / 32) % 32 });
        CHECK(Frame(S, 2, Many), "pressure frame");
        CHECK(S.m_Stats.m_Admitted == Fine && S.m_Stats.m_Rejected == 300 - Fine && S.m_Stats.m_ResidentPinned == S.getSetup().PinnedSlots()
             , "pressure: admitted %d rejected %d (fine slots %d), net %d", S.m_Stats.m_Admitted, S.m_Stats.m_Rejected, Fine, S.m_Stats.m_ResidentPinned);
        // The same requests again: every resident one is protected; nothing is evicted for the rejected ones
        CHECK(Frame(S, 3, Many) && S.m_Stats.m_Evicted == 0 && S.m_Stats.m_Reused == Fine, "protected pages were evicted (%d) or not reused (%d)", S.m_Stats.m_Evicted, S.m_Stats.m_Reused);

        // B13, slot reuse: once the old requests are past the protection, new pages take their slots - least recently requested first -,
        // each reused slot's generation changes, the old page is gone and the table never names it
        std::vector<vsm::key> Other;
        for (int i = 0; i < 64; ++i) Other.push_back({ 1, S.m_Levels[1].m_OriginX + i % 32, S.m_Levels[1].m_OriginY + i / 32 });
        std::vector<std::pair<vsm::key, std::uint32_t>> Before;          // the resident pages and their slots' generations
        for (const auto& K : Many) if (const int s = S.Find(K); s >= 0) Before.push_back({ K, S.Generation(s) });
        for (std::uint64_t f = 4; f < 4 + std::uint64_t(S.getSetup().m_ProtectFrames); ++f)
            CHECK(Frame(S, f, Other) && S.m_Stats.m_Evicted == 0, "frame %llu: %d pages evicted while still protected", (unsigned long long)f, S.m_Stats.m_Evicted);
        CHECK(Frame(S, 4 + std::uint64_t(S.getSetup().m_ProtectFrames) + 1, Other), "reuse frame");
        int nGone = 0;
        for (const auto& [K, G] : Before) nGone += S.Find(K) < 0;
        int nAdmitted = 0;
        for (const auto& K : Other) nAdmitted += S.IsValid(K);
        CHECK(S.m_Stats.m_Evicted == nGone && nAdmitted == int(Other.size()) && nGone == nAdmitted, "slot reuse: evicted %d, old pages gone %d, new pages valid %d of %zu"
             , S.m_Stats.m_Evicted, nGone, nAdmitted, Other.size());
        for (const auto& K : Other) CHECK(S.Generation(S.Find(K)) > 1, "a reused slot kept its generation");
    }

    // A new light direction: a new epoch, every page goes, the net is rendered again; requests decoded with an older epoch are dropped
    {
        const auto Old = V.Snapshot();
        CHECK(Frame(V, 20, {}, {}, { 0, 0, 0 }, { -0.5f, -1, 0.1f }), "epoch frame");
        CHECK(V.m_Stats.m_Epochs == 1 && V.m_Stats.m_RenderedPinned == Pinned && TableCount(V) == Pinned, "epoch: %d, net rendered %d, table %d", V.m_Stats.m_Epochs, V.m_Stats.m_RenderedPinned, TableCount(V));
        std::vector<std::uint32_t> Bits(std::size_t(V.getSetup().TableUints()) * 2 / 32 + 1, 0xFFFFFFFFu);
        std::vector<vsm::key> Decoded;
        V.DecodeRequests(Bits, Old, Decoded);
        CHECK(Decoded.empty(), "requests of an old epoch decoded (%zu)", Decoded.size());
    }

    // Decoding: a request bit maps to the page of the window that made it, even after the window moved
    {
        const auto S0 = V.Snapshot();
        const auto& Lv = V.m_Levels[1];
        const vsm::key K{ 1, Lv.m_OriginX + 3, Lv.m_OriginY + 30 };
        std::vector<std::uint32_t> Bits(std::size_t(V.getSetup().TableEntries() + 31) / 32, 0u);
        const int i = V.TableIndex(K);
        Bits[std::size_t(i) >> 5] |= 1u << (i & 31);
        CHECK(Frame(V, 21, {}, {}, { 100, 0, 40 }, { -0.5f, -1, 0.1f }), "moved frame");      // the camera moved: the windows moved
        std::vector<vsm::key> Decoded;
        V.DecodeRequests(Bits, S0, Decoded);
        CHECK(Decoded.size() == 1 && Decoded[0] == K, "decoded (%d %d %d), wanted (%d %d %d)", Decoded.empty() ? -1 : Decoded[0].m_Level
             , Decoded.empty() ? 0 : Decoded[0].m_X, Decoded.empty() ? 0 : Decoded[0].m_Y, K.m_Level, K.m_X, K.m_Y);
    }

    // The windows follow the camera: pages left behind go; a far vertical move re-centres the depth ranges (their pages go too)
    {
        vsm W;
        W.Init({});
        CHECK(Frame(W, 1, {}), "w0");
        std::vector<vsm::key> R{ { 0, W.m_Levels[0].m_OriginX, W.m_Levels[0].m_OriginY } };
        CHECK(Frame(W, 2, R) && W.IsValid(R[0]), "w1");
        CHECK(Frame(W, 3, {}, {}, { 1000, 0, 1000 }) && W.Find(R[0]) < 0 && W.m_Stats.m_Evicted >= 1, "a page out of its window stayed");
        std::vector<vsm::key> R2{ { 0, W.m_Levels[0].m_OriginX + 5, W.m_Levels[0].m_OriginY + 5 } };
        CHECK(Frame(W, 4, R2, {}, { 1000, 0, 1000 }) && W.IsValid(R2[0]), "w2");
        CHECK(Frame(W, 5, {}, {}, xmath::fvec3{ 1000, 0, 1000 } + W.m_Z * (W.DepthHalfRange(0) * 0.75f)) && W.Find(R2[0]) < 0, "a page kept a moved depth range");
    }

    std::printf("P1a-0 (CPU): addressing, net, budget, dirty pages, pressure, slot reuse (B13), epochs, decoding, windows: ok\n");

    //------------------------------------------------------------------------------------------------------------------------------
    // P1a-1 (GPU)
    app App;
    CHECK(App.Init(), "Init");
    App.m_bQuietTimes = true;
    auto& Renderer = App.m_Renderer;
    auto& Device   = App.m_Device;

    const auto SpherePrim = MakeSphere();
    const auto BoxPrim    = xprim_geom::cube::Generate(1, 1, 1, 1, xprim_geom::float3{ 1, 1, 1 });
    xlionrender::mesh_id Sphere, Box;
    CHECK(MakeMesh(Renderer, SpherePrim, Sphere), "sphere");
    CHECK(MakeMesh(Renderer, BoxPrim, Box), "box");

    constexpr int size_v = 256;
    xlionrender::render_view View;
    CHECK(!Renderer.CreateView(View, { .m_Width = size_v, .m_Height = size_v }), "CreateView");
    xgpu::tools::view Camera;
    Camera.setFov(xmath::radian{ 1.1f });
    Camera.setViewport({ 0, 0, size_v, size_v });
    Camera.LookAt(xmath::fvec3{ 0, 6.0f, -9.0f }, xmath::fvec3{ 0, 0, 1.0f });

    // A wide ground (draw 0), a grid of spheres, a thin pole (its shadow is a thin line across pages), a cluster far away (the teleport's)
    xlionrender::scene Scene;
    Scene.m_Draws.push_back({ .m_Mesh = Box, .m_L2W = xmath::fmat4::fromTranslation({ 30, -0.55f, 20 }) * xmath::fmat4::fromScale({ 120.0f, 0.1f, 100.0f }) });
    for (int z = 0; z < 4; ++z) for (int x = 0; x < 5; ++x)
        Scene.m_Draws.push_back({ .m_Mesh = Sphere, .m_L2W = xmath::fmat4::fromTranslation({ -4.0f + 2.0f * x, 0.0f, -2.0f + 2.5f * z }) });
    Scene.m_Draws.push_back({ .m_Mesh = Box, .m_L2W = xmath::fmat4::fromTranslation({ 1.2f, 0.45f, 0.6f }) * xmath::fmat4::fromScale({ 0.04f, 2.0f, 0.04f }) });
    for (int k = 0; k < 11; ++k)            // the far view's: a row of spheres and poles out to 60 m
    {
        Scene.m_Draws.push_back({ .m_Mesh = Sphere, .m_L2W = xmath::fmat4::fromTranslation({ (k % 2) ? 2.0f : -2.0f, 0.0f, 10.0f + 5.0f * k }) * xmath::fmat4::fromScale({ 1.5f, 1.5f, 1.5f }) });
        Scene.m_Draws.push_back({ .m_Mesh = Box, .m_L2W = xmath::fmat4::fromTranslation({ (k % 2) ? -1.0f : 1.0f, 0.95f, 12.0f + 5.0f * k }) * xmath::fmat4::fromScale({ 0.15f, 3.0f, 0.15f }) });
    }
    for (int z = 0; z < 3; ++z) for (int x = 0; x < 3; ++x)
        Scene.m_Draws.push_back({ .m_Mesh = Sphere, .m_L2W = xmath::fmat4::fromTranslation({ 58.0f + 2.0f * x, 0.0f, 0.0f + 2.5f * z }) });
    Scene.m_Sun     = { .m_Direction = { 0.3f, -1.0f, 0.2f }, .m_Intensity = 3, .m_ShadowTileSize = 4.0f };     // (the Police level's)
    Scene.m_Ambient = { 0, 0, 0 };          // a pixel is the sun's light only (its visibility: shadowed / unshadowed)
    Renderer.m_bVirtualShadows = true;

    const auto Settle = [&](int n)
    {
        for (int i = 0; i < n; ++i) if (!App.RenderFrame(Scene, View, Camera) || !Renderer.validateVSM()) return false;
        return true;
    };
    const auto Pct = [](std::uint32_t a, std::uint32_t b) { return b ? 100.0 * double(a) / double(b) : 0.0; };

    // Unshadowed, then the pages settled (a budget for everything, as the full redraw below needs)
    std::vector<float> Unlit, Lit, Clip;
    const float Bias = std::getenv("XLION_VSM_TEST_BIAS") ? float(std::atof(std::getenv("XLION_VSM_TEST_BIAS"))) : vsm::setup{}.m_ResolutionBias;
    Renderer.configureVSM({ .m_Budget = 100000, .m_ResolutionBias = Bias });
    Scene.m_Sun.m_bCastShadows = false;
    CHECK(App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, Unlit), "render unshadowed");
    Scene.m_Sun.m_bCastShadows = true;
    CHECK(Settle(12), "settle (invariant 1 failed on a frame)");
    CHECK(App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, Lit), "render settled");
    {
        const auto& R = Renderer.getVSMReport();
        const auto& C = R.m_Counters;
        std::printf("settled: %u pixels - by their level %.2f%%, coarser %.2f%%, none %.2f%%, crossing a page border %.2f%%; pages: requested %d, resident %d + net %d; request pass %.3f ms\n"
                   , C[0], Pct(C[1], C[0]), Pct(C[2], C[0]), Pct(C[3], C[0]), Pct(C[4], C[0]), R.m_Pages.m_Requested, R.m_Pages.m_ResidentFine, R.m_Pages.m_ResidentPinned, R.m_RequestGpuMs);
        std::printf("levels:"); for (int l = 0; l < 12; ++l) if (C[5 + l]) std::printf(" L%d %.1f%%", l, Pct(C[5 + l], C[0])); std::printf("\n");
        std::printf("log2(footprint / texel):"); for (int b = 0; b < 8; ++b) if (C[17 + b]) std::printf(" [%d] %.1f%%", b - 4, Pct(C[17 + b], C[0])); std::printf("\n");
        CHECK(R.m_bCounters && C[0] > 10000, "the request pass counted %u pixels", C[0]);
        CHECK(C[3] == 0 && C[2] == 0, "not settled: %u pixels by a coarser level, %u by none", C[2], C[3]);
        int nDarker = 0;
        for (std::size_t i = 0; i < Lit.size(); ++i)
        {
            CHECK(std::isfinite(Lit[i]) && Lit[i] <= Unlit[i] * 1.001f + 1e-4f, "texel %zu brighter with shadows (%f -> %f)", i, Unlit[i], Lit[i]);
            nDarker += Lit[i] < Unlit[i] * 0.6f;
        }
        // The clipmap (the rollback) on the same view: the pages shadow about as much (finer, so not the same pixels)
        Renderer.m_bVirtualShadows = false;
        CHECK(App.RenderFrame(Scene, View, Camera) && App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, Clip), "render the clipmap");
        Renderer.m_bVirtualShadows = true;
        CHECK(Settle(12), "settle again after the clipmap");
        int nClip = 0;
        for (std::size_t i = 0; i < Clip.size(); ++i) nClip += Clip[i] < Unlit[i] * 0.6f;
        std::printf("shadowed pixels: pages %d, clipmap %d\n", nDarker, nClip);
        CHECK(nDarker > 200 && nDarker > nClip * 3 / 4 && nDarker < nClip * 4 / 3 + 100, "pages shadow %d pixels, the clipmap %d", nDarker, nClip);
    }

    // The cached pages equal a full redraw (every resident page rendered again this frame)
    {
        std::vector<float> Full;
        Renderer.m_bShadowCache = false;
        CHECK(App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, Full) && Renderer.validateVSM(), "render without the cache");
        Renderer.m_bShadowCache = true;
        int nDiff = 0;
        for (std::size_t i = 0; i < Full.size(); ++i) nDiff += Full[i] != Lit[i];
        std::printf("full redraw: %d pixels differ from the cached pages (%d pages rendered)\n", nDiff, Renderer.getVSMReport().m_Pages.m_Rendered + Renderer.getVSMReport().m_Pages.m_RenderedPinned);
        CHECK(nDiff == 0, "the cached pages differ from a full redraw on %d pixels", nDiff);

        // Invariant 2's path: every tap through the table (as across a page border) gives the same image as the one-page fast path
        std::vector<float> Taps;
        Renderer.m_bVSMTapTranslation = true;
        CHECK(App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, Taps), "render with every tap translated");
        Renderer.m_bVSMTapTranslation = false;
        int nTapDiff = 0;
        for (std::size_t i = 0; i < Taps.size(); ++i) nTapDiff += Taps[i] != Lit[i];
        std::printf("per-tap translation: %d pixels differ from the one-page path\n", nTapDiff);
        CHECK(nTapDiff == 0, "the border path differs from the one-page path on %d pixels", nTapDiff);
    }

    // Against the GI's rays on the same triangles, per pixel, the pages and the clipmap: disagreement overall and at the shadow edges, the
    // texel against the pixel. On the near view, B12: the pixels whose filter crosses a page border (found as the request pass finds
    // them: the footprint from the neighbours' hits, the level, the 4 x 4 taps) disagree no more than the others
    const auto RaySphere = RayMesh(SpherePrim), RayBox = RayMesh(BoxPrim);
    xlionrender::gi::world World;
    for (const auto& D : Scene.m_Draws) World.Add(D.m_Mesh.m_Index == Sphere.m_Index ? RaySphere : RayBox, D.m_L2W, { 0.5f, 0.5f, 0.5f });
    const auto Measure = [&](const char* pName, const std::vector<float>& Unlit, const std::vector<float>& Lit, const std::vector<float>& Clip, bool bNear, int S) -> int
    {
        const xmath::fmat4 C2W = Camera.getC2W();
        const xmath::fvec3 Eye = Camera.getPosition(), Forward = Camera.getForward();
        std::vector<xmath::fvec3> Hit(std::size_t(S) * S), Nrm(Hit.size());
        std::vector<char> bHit(Hit.size(), 0);
        for (int y = 0; y < S; ++y) for (int x = 0; x < S; ++x)
        {
            const float nx = 2.0f * (float(x) + 0.5f) / S - 1.0f, ny = 2.0f * (float(y) + 0.5f) / S - 1.0f;   // (the light reference's mapping)
            const auto At = [&](float z) { const xmath::fvec4 P = C2W * xmath::fvec4{ nx, ny, z, 1.0f }; return xmath::fvec3{ P.m_X, P.m_Y, P.m_Z } * (1.0f / P.m_W); };
            auto D = (At(0.6f) - At(0.2f)).NormalizeCopy();
            if (D.Dot(Forward) < 0) D = D * -1.0f;
            const auto H = World.Cast(Eye, D);
            const std::size_t i = std::size_t(y) * S + x;
            if (H.m_Inst < 0) continue;
            bHit[i] = 1; Hit[i] = World.HitPosition(H); Nrm[i] = World.HitNormal(H).NormalizeCopy();
        }
        const auto& V = Renderer.getVSM();
        const float t0 = V.Level0Texel();
        const xmath::fvec3 ToSun = Scene.m_Sun.m_Direction.NormalizeCopy() * -1.0f;
        std::vector<float> Ray(Hit.size(), 1.0f);            // the rays' visibility per pixel (a shadow edge: a neighbour's differs)
        for (std::size_t i = 0; i < Hit.size(); ++i) if (bHit[i]) Ray[i] = World.Occluded(Hit[i] + Nrm[i] * 0.01f, ToSun, 1e4f) ? 0.0f : 1.0f;
        const xmath::fvec3 CamL = V.ToLight(Eye);           // (the clipmap's basis is the pages': the same construction)
        int nB = 0, nBD = 0, nO = 0, nOD = 0, nMiss = 0;
        int nPix = 0, nDisV = 0, nDisC = 0, nEdge = 0, nEdgeV = 0, nEdgeC = 0;
        int nFitV = 0, nFitC = 0, nUnderV = 0, nUnderC = 0;     // texel within [pixel/2, pixel]; texel larger than the pixel
        int nInDarkV = 0, nInLeakV = 0, nInDarkC = 0, nInLeakC = 0;  // away from the edges: shadowed where the rays see the sun (acne), lit where they do not (leaks)
        for (int y = 1; y < S - 1; ++y) for (int x = 1; x < S - 1; ++x)
        {
            const std::size_t i = std::size_t(y) * S + x;
            if (Unlit[i] <= 1e-4f) continue;
            if (!bHit[i]) { ++nMiss; continue; }
            const auto L = V.ToLight(Hit[i]);
            const auto Axis = [&](std::size_t a, std::size_t b)
            {
                float f = 1e30f;
                for (const std::size_t j : { a, b }) if (bHit[j]) { const auto Q = V.ToLight(Hit[j]); f = std::min(f, std::sqrt((Q.m_X - L.m_X) * (Q.m_X - L.m_X) + (Q.m_Y - L.m_Y) * (Q.m_Y - L.m_Y))); }
                return f < 1e29f ? f : 0.0f;
            };
            float Foot = std::max(Axis(i - 1, i + 1), Axis(i - S, i + S));
            if (Foot <= 0) Foot = t0;
            const int   Level = std::clamp(int(std::floor(std::log2(Foot / t0) + V.getSetup().m_ResolutionBias)), 0, V.getSetup().m_Levels - 1);
            const float Texel = t0 * float(1 << Level);
            const int   LoX = int(std::floor(L.m_X / Texel - 0.5f)) - 1, LoY = int(std::floor(L.m_Y / Texel - 0.5f)) - 1;
            const auto  Pg = [](int t) { return int(std::floor(float(t) / float(vsm::page_texels_v))); };
            const bool  bCross = Pg(LoX) != Pg(LoX + 3) || Pg(LoY) != Pg(LoY + 3);
            const float VMap = std::clamp(Lit[i] / Unlit[i], 0.0f, 1.0f);
            const float VRay = Ray[i];
            const bool  bDis = std::abs(VMap - VRay) > 0.5f;
            if (bCross) { ++nB; nBD += bDis; } else { ++nO; nOD += bDis; }

            // Against the clipmap (the rollback) on the same pixels: disagreement overall and at the shadow edges, texel against pixel
            const float VClip = std::clamp(Clip[i] / Unlit[i], 0.0f, 1.0f);
            const bool  bDisC = std::abs(VClip - VRay) > 0.5f;
            const bool  bEdge = Ray[i - 1] != VRay || Ray[i + 1] != VRay || Ray[i - S] != VRay || Ray[i + S] != VRay;
            ++nPix; nDisV += bDis; nDisC += bDisC;
            if (!bEdge) { nInDarkV += bDis && VMap < VRay; nInLeakV += bDis && VMap > VRay; nInDarkC += bDisC && VClip < VRay; nInLeakC += bDisC && VClip > VRay; }
            if (bEdge) { ++nEdge; nEdgeV += bDis; nEdgeC += bDisC; }
            float TexelC = Scene.m_Sun.m_ShadowTileSize * 8.0f / 512.0f;        // past the clipmap's levels: the coarsest's texel
            for (int Lc = 0; Lc < xlionrender::shadows::clipmap_levels_v; ++Lc)
            {
                const float T = Scene.m_Sun.m_ShadowTileSize * float(1 << Lc);
                const int   ox = int(std::floor(CamL.m_X / T)) - 2, oy = int(std::floor(CamL.m_Y / T)) - 2;
                const int   tx = int(std::floor(L.m_X / T)) - ox, ty = int(std::floor(L.m_Y / T)) - oy;
                if (tx >= 0 && tx < 4 && ty >= 0 && ty < 4) { TexelC = T / 512.0f; break; }
            }
            const float rV = std::log2(Foot / Texel), rC = std::log2(Foot / TexelC);
            nFitV += rV >= 0 && rV < 1; nFitC += rC >= 0 && rC < 1;
            nUnderV += rV < 0; nUnderC += rC < 0;
        }
        std::printf("%s: against the rays: pages disagree on %.2f%% of %d lit pixels (%.2f%% of %d at shadow edges), the clipmap %.2f%% (%.2f%% at edges)\n"
                   , pName, Pct(nDisV, nPix), nPix, Pct(nEdgeV, nEdge), nEdge, Pct(nDisC, nPix), Pct(nEdgeC, nEdge));
        std::printf("%s: texel against pixel: pages %.1f%% within [pixel/2, pixel], %.1f%% larger than the pixel; clipmap %.1f%% within, %.1f%% larger\n"
                   , pName, Pct(nFitV, nPix), Pct(nUnderV, nPix), Pct(nFitC, nPix), Pct(nUnderC, nPix));
        std::printf("%s: away from the edges - pages %d dark (acne) %d lit (leaks), clipmap %d dark %d lit\n", pName, nInDarkV, nInLeakV, nInDarkC, nInLeakC);
        CHECK(nMiss * 100 < nPix, "the pixel to ray mapping is off (%d missed)", nMiss);
        CHECK(nDisV <= nDisC && nEdgeV <= nEdgeC, "%s: the pages disagree with the rays more than the clipmap (%d vs %d, edges %d vs %d)", pName, nDisV, nDisC, nEdgeV, nEdgeC);
        CHECK(nUnderV <= nUnderC, "%s: more of the pages' texels are larger than the pixel than the clipmap's (%d vs %d)", pName, nUnderV, nUnderC);
        if (!bNear) return 0;
        const double DB = Pct(nBD, nB), DO = Pct(nOD, nO);
        std::printf("B12 boundaries: %d pixels cross a page border, %.2f%% disagree with the rays; %d do not, %.2f%% disagree (%d lit pixels the rays missed)\n", nB, DB, nO, DO, nMiss);
        CHECK(nB > 200, "only %d pixels cross a page border", nB);
        CHECK(DB <= DO * 1.5 + 1.0, "pixels across page borders disagree more (%.2f%%) than the others (%.2f%%)", DB, DO);
        return 0;
    };
    // The comparisons at a Police-like pixel density (a 1536^2 view: at 256^2 the pixels are larger than the clipmap's finest texels)
    constexpr int hires_v = 1536;
    xlionrender::render_view Hi;
    CHECK(!Renderer.CreateView(Hi, { .m_Width = hires_v, .m_Height = hires_v }), "CreateView hi");
    const auto Triple = [&](std::vector<float>& U, std::vector<float>& C, std::vector<float>& P) -> bool
    {
        Scene.m_Sun.m_bCastShadows = false;
        if (!App.RenderFrame(Scene, Hi, Camera) || !ReadHDR(Device, Hi, U)) return false;
        Scene.m_Sun.m_bCastShadows = true;
        Renderer.m_bVirtualShadows = false;
        if (!App.RenderFrame(Scene, Hi, Camera) || !App.RenderFrame(Scene, Hi, Camera) || !ReadHDR(Device, Hi, C)) return false;
        Renderer.m_bVirtualShadows = true;
        for (int k = 0; k < 12; ++k) if (!App.RenderFrame(Scene, Hi, Camera) || !Renderer.validateVSM()) return false;
        return App.RenderFrame(Scene, Hi, Camera) && ReadHDR(Device, Hi, P);
    };
    {
        std::vector<float> HUnlit, HClip, HLit;
        CHECK(Triple(HUnlit, HClip, HLit), "near view at 1536");
        const auto& C = Renderer.getVSMReport().m_Counters;
        std::printf("near view (1536): %u pixels by their level %.2f%%; pages requested %d; request pass %.3f ms\n", C[0], Pct(C[1], C[0]), Renderer.getVSMReport().m_Pages.m_Requested, Renderer.getVSMReport().m_RequestGpuMs);
        if (Measure("near view (1536)", HUnlit, HLit, HClip, true, hires_v)) return 1;
    }

    // The far view: the ground to 60 m (Tomas: the texels too large on screen)
    {
        std::vector<float> FUnlit, FClip, FLit;
        Camera.LookAt(xmath::fvec3{ 0, 2.5f, -6.0f }, xmath::fvec3{ 0, 0, 40.0f });
        CHECK(Triple(FUnlit, FClip, FLit), "far view at 1536");
        const auto& C = Renderer.getVSMReport().m_Counters;
        std::printf("far view (1536): %u pixels by their level %.2f%%; pages requested %d; request pass %.3f ms\n", C[0], Pct(C[1], C[0]), Renderer.getVSMReport().m_Pages.m_Requested, Renderer.getVSMReport().m_RequestGpuMs);
        if (Measure("far view (1536)", FUnlit, FLit, FClip, false, hires_v)) return 1;
        Camera.LookAt(xmath::fvec3{ 0, 6.0f, -9.0f }, xmath::fvec3{ 0, 0, 1.0f });
        Renderer.DestroyView(Hi);
        CHECK(Settle(12), "settle back on the near view");
    }


    // Temporal stability on a still scene: the settled frames are identical
    {
        std::vector<float> A, B;
        CHECK(App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, A) && App.RenderFrame(Scene, View, Camera) && ReadHDR(Device, View, B), "still frames");
        int nDiff = 0;
        for (std::size_t i = 0; i < A.size(); ++i) nDiff += A[i] != B[i];
        CHECK(nDiff == 0, "a still scene changes between frames on %d pixels", nDiff);
    }

    // Invariant 4 and B5 (a stress result): the sun turning 1.5 degree a frame about the vertical (~0.6 degree between the directions, over setup::m_EpochDegrees) - each frame a new epoch, every
    // page invalid at once, the net rendered again first (whatever the budget), the fine pages by demand; never a hole
    {
        const xmath::fvec3 Dir0 = Scene.m_Sun.m_Direction;
        const auto Turn = [&](float Degrees)
        {
            const float a = Degrees * 3.14159265f / 180.0f;
            Scene.m_Sun.m_Direction = { std::cos(a) * Dir0.m_X + std::sin(a) * Dir0.m_Z, Dir0.m_Y, -std::sin(a) * Dir0.m_X + std::cos(a) * Dir0.m_Z };
        };

        // Under the threshold the epoch stays (its direction, the mismatch reported)
        Turn(0.1f);
        CHECK(App.RenderFrame(Scene, View, Camera) && Renderer.validateVSM(), "sub-threshold turn (invariant 1)");
        CHECK(Renderer.getVSMReport().m_Pages.m_Epochs == 0, "a 0.1 degree turn opened an epoch");
        std::printf("B5 sub-threshold turn 0.1 degree: epoch kept, mismatch %.3f degree\n", Renderer.getVSMReport().m_Pages.m_EpochMismatchDegrees);
        // (the turn is about the vertical: the directions are 0.1 degree times the horizontal part apart)
        const float Expected = std::acos(std::clamp(Scene.m_Sun.m_Direction.NormalizeCopy().Dot(Dir0.NormalizeCopy()), -1.0f, 1.0f)) * 180.0f / 3.14159265f;
        CHECK(std::abs(Renderer.getVSMReport().m_Pages.m_EpochMismatchDegrees - Expected) < 0.01f, "the mismatch reads %.3f, not %.3f", Renderer.getVSMReport().m_Pages.m_EpochMismatchDegrees, Expected);

        const int nNet = Renderer.getVSM().getSetup().PinnedSlots();
        std::printf("B5 moving sun 1.5 degree yaw a frame (frame: pages rendered fine + net, pixels by their level / coarser / none %%):");
        for (int f = 0; f < 12; ++f)
        {
            const float a = (0.1f + 1.5f * float(f + 1)) * 3.14159265f / 180.0f;
            Scene.m_Sun.m_Direction = { std::cos(a) * Dir0.m_X + std::sin(a) * Dir0.m_Z, Dir0.m_Y, -std::sin(a) * Dir0.m_X + std::cos(a) * Dir0.m_Z };
            CHECK(App.RenderFrame(Scene, View, Camera) && Renderer.validateVSM(), "moving sun frame %d (invariant 1)", f);
            const auto& R = Renderer.getVSMReport();
            CHECK(R.m_Pages.m_Epochs == 1 && R.m_Pages.m_RenderedPinned == nNet && R.m_Pages.m_PinnedNotValid == 0, "frame %d: epoch %d, net rendered %d of %d", f, R.m_Pages.m_Epochs, R.m_Pages.m_RenderedPinned, nNet);
            const auto& C = R.m_Counters;
            std::printf(" [%d] %d+%d %.0f/%.0f/%.0f", f, R.m_Pages.m_Rendered, R.m_Pages.m_RenderedPinned, Pct(C[1], C[0]), Pct(C[2], C[0]), Pct(C[3], C[0]));
            CHECK(C[3] == 0, "the moving sun left pixels with no level (%u)", C[3]);
        }
        std::printf("\n");
        Scene.m_Sun.m_Direction = Dir0;
        CHECK(Settle(12), "settle after the moving sun");
    }

    // B11 overflow: a pool too small for the view - pages rejected, pixels by a coarser level, none by no level
    {
        Renderer.configureVSM({ .m_PoolTexels = 1664, .m_PinnedRadius = 6, .m_Budget = 100000 });     // 169 slots: 144 net, 25 fine (the view asks for 27)
        CHECK(Settle(12), "overflow settle (invariant 1)");
        const auto& R = Renderer.getVSMReport();
        const auto& C = R.m_Counters;
        std::printf("B11 overflow: rejected %d pages; %u pixels - by their level %.2f%%, coarser %.2f%%, none %.2f%%\n", R.m_Pages.m_Rejected, C[0], Pct(C[1], C[0]), Pct(C[2], C[0]), Pct(C[3], C[0]));
        CHECK(R.m_Pages.m_Rejected > 0 && C[2] > 0, "the small pool did not overflow (rejected %d, coarser %u)", R.m_Pages.m_Rejected, C[2]);
        CHECK(C[3] == 0, "%u pixels had no level at all (a hole)", C[3]);
    }

    // B7 teleport: settled here, then a jump to the far cluster; the counters arrive a ring late (renderShadows reads them back)
    {
        Renderer.configureVSM({});
        CHECK(Settle(12), "teleport settle");
        Camera.LookAt(xmath::fvec3{ 60.0f, 6.0f, -9.0f }, xmath::fvec3{ 60.0f, 0, 1.0f });
        std::printf("B7 teleport (frame after the jump: by their level / coarser / none %%):");
        std::vector<std::array<double, 3>> Curve;
        for (int f = 0; f < 16; ++f)
        {
            CHECK(App.RenderFrame(Scene, View, Camera) && Renderer.validateVSM(), "teleport frame %d (invariant 1)", f);
            if (f >= 3)         // (frame f reports frame f - 3)
            {
                const auto& C = Renderer.getVSMReport().m_Counters;
                Curve.push_back({ Pct(C[1], C[0]), Pct(C[2], C[0]), Pct(C[3], C[0]) });
                std::printf(" [%d] %.0f/%.0f/%.0f", f - 3, Curve.back()[0], Curve.back()[1], Curve.back()[2]);
            }
        }
        std::printf("\n");
        for (const auto& c : Curve) CHECK(c[2] == 0.0, "a frame after the teleport had pixels with no level (%.2f%%)", c[2]);
        CHECK(Curve.back()[0] > 99.0, "not recovered 12 frames after the teleport (%.1f%% by their level)", Curve.back()[0]);
    }

    // A large scene's tile (the static geom preview's is the model's radius / 2: 40 m for Bistro): level 0 stays at 2 mm or finer,
    // and a close view gets texels within its pixels (Tomas: "Texels from sun dont look smaller" - level 0 was the clipmap's / 4, 2 cm)
    {
        Scene.m_Sun.m_ShadowTileSize = 40.0f;
        Camera.LookAt(xmath::fvec3{ 0, 1.5f, -1.0f }, xmath::fvec3{ 0, 0, 0.5f });
        CHECK(Settle(12), "40 m tile settle (invariant 1)");
        const auto& R = Renderer.getVSMReport();
        const auto& C = R.m_Counters;
        const std::uint32_t Larger = C[17] + C[18] + C[19] + C[20];         // log2(footprint / texel) < 0: the texel larger than the pixel
        std::printf("40 m tile, 1.5 m away: level 0 texel %.2f mm, %d levels; pixels by their level %.1f%%, texel larger than the pixel %.2f%%, pages resident %d + net\n"
                   , Renderer.getVSM().Level0Texel() * 1000.0f, Renderer.getVSM().getSetup().m_Levels, Pct(C[1], C[0]), Pct(Larger, C[0]), R.m_Pages.m_ResidentFine);
        CHECK(Renderer.getVSM().Level0Texel() <= vsm::max_level0_texel_v, "level 0 texel %.4f m with a 40 m tile", Renderer.getVSM().Level0Texel());
        CHECK(C[0] > 0 && Pct(C[1], C[0]) > 99.0, "a close view on a 40 m tile: %.1f%% by their level", Pct(C[1], C[0]));
        CHECK(Pct(Larger, C[0]) < 1.0, "a close view on a 40 m tile: %.2f%% of the pixels have texels larger than themselves", Pct(Larger, C[0]));
        Scene.m_Sun.m_ShadowTileSize = 4.0f;
    }

    Renderer.DestroyView(View);
    CHECK(g_nErrors == 0, "%d validation errors", g_nErrors);
    std::printf("PASS\n");
    return 0;
}
