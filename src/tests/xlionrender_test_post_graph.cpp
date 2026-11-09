// The post chain's plan (render graph P1a, xlionrender_post_graph.h), on the CPU: the default description lowers to exactly what the
// hand-written chain records (xlionrender_plan_forward.cpp, the Node* steps; xLION_Docs/render_graph_post_chain_inventory_2026-11-09.md) -
// scope names, bindings, targets, sizes, dispatches, the render graph's reads and writes, in order - at several sizes, odd ones too;
// a bad description is refused with the node it is about.
#include "xlionrender_post_graph_text.h"
#include <filesystem>
#include <cstring>
#include <cstdio>
#include <cstdlib>

namespace post = xlionrender::post;
namespace rg = xlionrender::rg;

// A description with these nodes and the default parameters
static post::desc With( std::vector<post::node> Nodes ) { auto D = post::desc::Default(); D.m_Nodes = std::move(Nodes); return D; }

#define CHECK(c, what) do { if( !(c) ) { std::printf("FAIL: %s\n", std::string(what).c_str()); std::exit(1); } } while(0)

// The hand-written chain, transcribed (TAA, bloom and auto exposure on): what each step binds, writes and declares
struct legacy_pass { std::string m_Scope; post::image_ref m_Bind[3]; post::image_ref m_Dst; int m_W, m_H; std::vector<post::pass::access> m_Acc; };
static void Legacy( int W, int H, std::vector<legacy_pass>& Inits, std::vector<legacy_pass>& Frame )
{
    using I = post::image; using A = post::pass::access;
    const post::image_ref HDR{ I::HDR }, DEPTH{ I::DEPTH }, CHAIN{ I::CHAIN }, EXPO{ I::EXPOSURE };
    const auto B = [](int i) { return post::image_ref{ I::BLOOM, i }; };
    const auto R = [](post::image_ref r) { return A{ r, rg::usage::SAMPLED, rg::policy::NONE, false }; };
    const auto Wr = [](post::image_ref r, rg::policy P) { return A{ r, rg::usage::STORAGE, P, true }; };
    // NodeTAA: the history images made: each zeroed (post_comp mode 3, T = { HDR, exposure })
    for( int i = 0; i < 2; ++i ) Inits.push_back({ "TAA history init", { HDR, EXPO }, { i ? I::TAA1 : I::TAA0 }, W, H, { R(HDR), Wr({ i ? I::TAA1 : I::TAA0 }, rg::policy::FULL_OVERWRITE) } });
    // NodePostInit: Run(HDR, level i) for the 6 levels, Run(HDR, exposure) - T[1] the exposure, or HDR when the target is the exposure
    for( int i = 0; i < 6; ++i ) { const auto S = post::BloomSize(W, H, i); Inits.push_back({ "post init", { HDR, EXPO }, B(i), S[0], S[1], { R(HDR), R(EXPO), Wr(B(i), rg::policy::FULL_OVERWRITE) } }); }
    Inits.push_back({ "exposure", { HDR, HDR }, EXPO, 1, 1, { R(HDR), R(HDR), Wr(EXPO, rg::policy::FULL_OVERWRITE) } });
    // NodeTAA's frame step
    Frame.push_back({ "TAA", { HDR, DEPTH, { I::TAA_PREV } }, { I::TAA_CUR }, W, H, { R(HDR), R(DEPTH), R({ I::TAA_PREV }), Wr({ I::TAA_CUR }, rg::policy::FULL_OVERWRITE) } });
    const int n = std::clamp(int(std::round(std::log2(float(H) / 48.0f))), 3, 6);
    for( int i = 0; i < n; ++i ) { const auto S = post::BloomSize(W, H, i); const auto Src = i ? B(i - 1) : CHAIN; Frame.push_back({ "bloom down", { Src, EXPO }, B(i), S[0], S[1], { R(Src), R(EXPO), Wr(B(i), rg::policy::FULL_OVERWRITE) } }); }
    Frame.push_back({ "exposure", { CHAIN, HDR }, EXPO, 1, 1, { R(CHAIN), R(HDR), Wr(EXPO, rg::policy::PRESERVE) } });
    for( int i = n - 2; i >= 0; --i ) { const auto S = post::BloomSize(W, H, i); Frame.push_back({ "bloom up", { B(i + 1), EXPO }, B(i), S[0], S[1], { R(B(i + 1)), R(EXPO), R(B(i)), Wr(B(i), rg::policy::PRESERVE) } }); }
    Frame.push_back({ "tone map", { CHAIN, B(0), EXPO }, { I::OUTPUT }, W, H, { R(CHAIN), R(B(0)), R(EXPO), A{ { I::OUTPUT }, rg::usage::COLOR_ATTACHMENT, rg::policy::FULL_OVERWRITE, true } } });
}

static void Same( const std::vector<post::pass>& Plan, const std::vector<legacy_pass>& L, const std::string& What )
{
    CHECK(Plan.size() == L.size(), std::format("{}: {} passes, the hand-written chain {}", What, Plan.size(), L.size()));
    for( std::size_t i = 0; i < L.size(); ++i )
    {
        const auto& p = Plan[i]; const auto& l = L[i];
        const auto Where = std::format("{} pass {} ({})", What, i, p.m_Source);
        CHECK(p.m_Scope == l.m_Scope, Where + ": scope " + p.m_Scope + " vs " + l.m_Scope);
        for( int b = 0; b < 3; ++b ) CHECK(p.m_Bind[b] == l.m_Bind[b], Where + ": a binding");
        CHECK(p.m_Dst == l.m_Dst, Where + ": the target");
        CHECK(p.m_W == l.m_W && p.m_H == l.m_H, std::format("{}: {} x {} vs {} x {}", Where, p.m_W, p.m_H, l.m_W, l.m_H));
        if( p.m_Kernel != post::kernel::TONE_MAP ) CHECK(p.m_GroupsX == (l.m_W + 7) / 8 && p.m_GroupsY == (l.m_H + 7) / 8, Where + ": the dispatch");
        CHECK(p.m_Accesses.size() == l.m_Acc.size(), Where + ": the declared accesses");
        for( std::size_t a = 0; a < l.m_Acc.size(); ++a )
        {
            const auto& x = p.m_Accesses[a]; const auto& y = l.m_Acc[a];
            CHECK(x.m_Ref == y.m_Ref && x.m_Usage == y.m_Usage && x.m_Policy == y.m_Policy && x.m_bWrite == y.m_bWrite, std::format("{}: access {}", Where, a));
        }
    }
}

int main()
{
    for( const auto [W, H] : { std::array{ 1920, 1080 }, std::array{ 3218, 1512 }, std::array{ 64, 64 }, std::array{ 1, 1 }, std::array{ 3840, 2160 } } )
    {
        const auto P = post::Lower(post::desc::Default(), W, H);
        const auto What = std::format("{} x {}", W, H);
        CHECK(P.m_Error.empty(), What + ": " + P.m_Error);
        std::vector<legacy_pass> Inits, Frame; Legacy(W, H, Inits, Frame);
        Same(P.m_Inits, Inits, What + " inits");
        Same(P.m_Passes, Frame, What + " frame");
    }
    // The odd size of the inventory: 5 levels, the last two 100 x 47 and 50 x 23 (made, zeroed; the 6th unused)
    {
        const auto P = post::Lower(post::desc::Default(), 3218, 1512);
        CHECK(P.m_nBloom == 5, "3218 x 1512: 5 bloom levels");
        CHECK(P.m_Passes[5].m_Source == "Bloom, down 4, 100 x 47", "3218 x 1512: the source map names the level and its size: " + P.m_Passes[5].m_Source);
        CHECK(P.m_Inits[7].m_W == 50 && P.m_Inits[7].m_H == 23, "3218 x 1512: the 6th level 50 x 23");
    }
    // Without bloom: no bloom steps (the tone map still binds level 0: zeroed once, its weight 0)
    {
        const auto P = post::Lower(With({ { post::node_type::TAA }, { post::node_type::EXPOSURE }, { post::node_type::TONE_MAP } }), 1920, 1080);
        CHECK(P.m_Error.empty() && P.m_nBloom == 0 && P.m_Passes.size() == 3, "no bloom: TAA, exposure, tone map");
    }
    // Exposure before the bloom (a different image while adapting - the threshold sees this frame's exposure - but valid)
    {
        const auto P = post::Lower(With({ { post::node_type::EXPOSURE }, { post::node_type::BLOOM }, { post::node_type::TONE_MAP } }), 1920, 1080);
        CHECK(P.m_Error.empty() && P.m_Passes[0].m_Scope == "exposure" && P.m_Passes[1].m_Scope == "bloom down" && P.m_Passes.back().m_Scope == "tone map", "exposure first");
        CHECK(P.m_Inits.size() == 7, "no TAA: no TAA history to make");
    }
    // Refused, with the node it is about
    const auto Refused = [](post::desc D, const char* pWhy, const char* pWhat) { const auto P = post::Lower(D, 64, 64); CHECK(P.m_Error.find(pWhy) != std::string::npos && P.m_Passes.empty(), std::string(pWhat) + ": " + P.m_Error); };
    Refused(With({ { post::node_type::BLOOM } }), "no tone map", "no tone map");
    Refused(With({ { post::node_type::TONE_MAP }, { post::node_type::BLOOM } }), "node 0: the tone map must be last", "tone map not last");
    Refused(With({ { post::node_type::BLOOM }, { post::node_type::TAA }, { post::node_type::TONE_MAP } }), "node 1: TAA must come first", "TAA not first");
    Refused(With({ { post::node_type::BLOOM }, { post::node_type::BLOOM }, { post::node_type::TONE_MAP } }), "node 1: a second one", "two blooms");
    // P3: parameters - a node needing one the description lacks, an unknown source, a type mismatch, a duplicate, an empty range
    { auto D = post::desc::Default(); std::erase_if(D.m_Params, [](const post::param& p) { return p.m_Name == "BloomKnee"; }); Refused(D, "node 1: the bloom needs the parameter 'BloomKnee'", "a missing parameter"); }
    { auto D = post::desc::Default(); D.m_Params[1].m_Source = "scene.post.bloom_treshold"; Refused(D, "parameter 'BloomThreshold': no source 'scene.post.bloom_treshold'", "an unknown source"); }
    { auto D = post::desc::Default(); D.m_Params[4].m_Type = post::param_type::FLOAT; Refused(D, "parameter 'AutoExposure': its source 'scene.post.auto_exposure' is a bool", "a type mismatch"); }
    { auto D = post::desc::Default(); D.m_Params.push_back(D.m_Params[0]); Refused(D, "parameter 'Bloom' is declared twice", "a duplicate"); }
    { auto D = post::desc::Default(); D.m_Params[0].m_Min = 2; Refused(D, "parameter 'Bloom': its range 2 .. 1 is empty", "an empty range"); }
    // P3: every push word a plan fills is filled once, from a known parameter
    {
        const auto P = post::Lower(post::desc::Default(), 1920, 1080);
        for( const auto& Pa : P.m_Passes )
            for( std::size_t i = 0; i < Pa.m_Slots.size(); ++i )
            {
                CHECK(Pa.m_Slots[i].m_From != post::slot_from::PARAM || (Pa.m_Slots[i].m_Param >= 0 && Pa.m_Slots[i].m_Param < int(P.m_Params.size())), Pa.m_Source + ": a slot's parameter resolved");
                for( std::size_t j = 0; j < i; ++j ) CHECK(Pa.m_Slots[i].m_Word != Pa.m_Slots[j].m_Word, Pa.m_Source + ": a push word filled twice");
            }
        // The snapshot: the Scene's values by index; out of range clamped and reported once a change
        xlionrender::post_settings S; S.m_BloomThreshold = 0.6f; S.m_Bloom = 3.0f; S.m_bAutoExposure = true;
        std::vector<float> Snap, Noted; int nReports = 0; std::string Last;
        const auto Rep = [&](const std::string& L) { ++nReports; Last = L; };
        post::SnapshotParams(P, S, Snap, Noted, Rep);
        CHECK(Snap[std::size_t(P.m_iBloom)] == 1.0f && nReports == 1 && Last.find("Bloom = 3 is outside 0 .. 1: 1 is used") != std::string::npos, "an out-of-range value clamped and reported: " + Last);
        CHECK(Snap[1] == 0.6f && Snap[std::size_t(P.m_iAutoExposure)] == 1.0f && Snap[6] == 0.115f, "the Scene's values, by index");
        post::SnapshotParams(P, S, Snap, Noted, Rep);
        CHECK(nReports == 1, "the same clamp is not reported again");
        S.m_Bloom = 0.5f; post::SnapshotParams(P, S, Snap, Noted, Rep); S.m_Bloom = 3.0f; post::SnapshotParams(P, S, Snap, Noted, Rep);
        CHECK(nReports == 2, "back in range, then out again: reported again");
        // FillSlots: the bloom's first down step - the words 3..7 of post_comp's push
        struct { int m_W, m_H, m_Mode; float m_A, m_B, m_C, m_D, m_E; } Push{};
        const auto& Down0 = P.m_Passes[1];
        CHECK(Down0.m_Scope == "bloom down" && Down0.m_Level == 0, "pass 1 is the first down step");
        post::FillSlots(Down0, Snap, 0.0f, 2.0f, &Push);
        CHECK(Push.m_A == 1.0f && Push.m_B == 0.6f && Push.m_C == 0.5f && Push.m_D == 2.0f && Push.m_E == 1.0f && Push.m_W == 0, "the first down step's push from the snapshot");
        // Every pass's pushed words against the hand-written chain's formulas (P2's xlionrender_plan_forward.cpp, transcribed)
        S.m_Bloom = 0.4f; S.m_BloomSpread = 0.65f; S.m_ExposureRate = 2.5f; S.m_ExposureKey = 0.2f; S.m_ExposureMin = 0.05f; S.m_ExposureMax = 20.0f; S.m_TAABlend = 0.15f;
        post::SnapshotParams(P, S, Snap, Noted, Rep);
        const float Dt = 1.0f / 30.0f, ViewExposure = 1.5f;
        for( const auto& Pa : P.m_Passes )
        {
            if( Pa.m_Kernel == post::kernel::TAA )
            {
                float W[20] = {}; post::FillSlots(Pa, Snap, Dt, ViewExposure, W);
                CHECK(W[18] == S.m_TAABlend, "TAA: the blend");
                continue;
            }
            if( Pa.m_Kernel == post::kernel::TONE_MAP )
            {
                float W[8] = {}; post::FillSlots(Pa, Snap, Dt, ViewExposure, W);
                CHECK(W[0] == ViewExposure && W[1] == S.m_Bloom && W[2] == 1.0f && W[3] == 0.0f, "tone map: exposure, bloom, auto");
                continue;
            }
            decltype(Push) X{}, L{}; post::FillSlots(Pa, Snap, Dt, ViewExposure, &X);
            if( Pa.m_Push == post::push::BLOOM_DOWN ) { L.m_A = Pa.m_Level == 0 ? 1.0f : 0.0f; L.m_B = S.m_BloomThreshold; L.m_C = S.m_BloomKnee; L.m_D = ViewExposure; L.m_E = S.m_bAutoExposure ? 1.0f : 0.0f; }
            if( Pa.m_Push == post::push::BLOOM_UP ) L.m_A = S.m_BloomSpread;
            if( Pa.m_Push == post::push::EXPOSURE ) { L.m_A = Dt; L.m_B = S.m_ExposureRate; L.m_C = S.m_ExposureKey; L.m_D = S.m_ExposureMin; L.m_E = S.m_ExposureMax; }
            CHECK(!std::memcmp(&X, &L, sizeof(X)), Pa.m_Source + ": the push words the hand-written chain pushed");
        }
        for( const auto& Pa : P.m_Inits ) CHECK(Pa.m_Slots.empty(), Pa.m_Source + ": an initialisation pushes only its fill (0)");
    }
    { const auto P = post::Lower(post::desc::Default(), 0, 1080); CHECK(!P.m_Error.empty(), "a 0-wide view is refused"); }
    // P2: the description as text - written and read back it is the same description, and lowers to the same plan; an unknown type names its row
    {
        const auto Path = (std::filesystem::temp_directory_path() / "xlionrender_test_post_graph.txt").wstring();
        auto D = post::desc::Default(); std::string Problem;
        { xtextfile::stream F; CHECK(!F.Open(false, Path, xtextfile::file_type::TEXT), "the description's file opens for writing"); CHECK(!post::Serialize(F, D, Problem), "written"); }
        post::desc R;
        { xtextfile::stream F; CHECK(!F.Open(true, Path, xtextfile::file_type::TEXT), "the description's file opens for reading"); CHECK(!post::Serialize(F, R, Problem) && Problem.empty(), "read: " + Problem); }
        CHECK(R == D, "read back: the same description");
        const auto A = post::Lower(D, 3218, 1512), B = post::Lower(R, 3218, 1512);
        CHECK(A.m_Passes.size() == B.m_Passes.size(), "the same plan");
        for( std::size_t i = 0; i < A.m_Passes.size(); ++i ) CHECK(A.m_Passes[i].m_Source == B.m_Passes[i].m_Source, "the same plan: " + A.m_Passes[i].m_Source);
        {
            std::FILE* f = _wfopen(Path.c_str(), L"w"); CHECK(f, "a bad file");
            std::fputs("[ PostGraph : 3 ]\n{ Type:s }\n  \"post.taa\"\n  \"post.blom\"\n  \"post.tonemap\"\n", f); std::fclose(f);
            xtextfile::stream F; CHECK(!F.Open(true, Path, xtextfile::file_type::TEXT), "the bad file opens");
            post::desc X; Problem.clear();
            CHECK(!post::Serialize(F, X, Problem) && Problem.find("row 1: no node type 'post.blom'") != std::string::npos && X.m_Nodes.empty(), "an unknown type names its row: " + Problem);
        }
        std::filesystem::remove(Path);
    }
    std::printf("post graph plan: ok\n");
    return 0;
}
