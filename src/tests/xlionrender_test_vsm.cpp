// The sun's virtual shadow pages (features/shadows/xlionrender_shadow_vsm.h; documentation/lighting_strategy.md P1a):
//  P1a-0 (CPU): addressing, the table, admission and eviction under pressure, protected and pinned slots, slot reuse and generations
//  (B13), epochs and invalidations; every published table checked against invariant 1 (Validate)
#include "xlionrender_test_common.h"
#include "features/shadows/xlionrender_shadow_vsm.h"

using namespace xlionrender::test;
using vsm = xlionrender::shadows::vsm;

namespace
{
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
    std::printf("PASS\n");
    return 0;
}
