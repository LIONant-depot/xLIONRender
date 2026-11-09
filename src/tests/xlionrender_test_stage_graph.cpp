// Frame stages as data (render graph R1/R2, xlionrender_stage_graph.h). The lowering refuses what does not match (a name not registered,
// the bindings, the push or the size not the kernel's). Then the equivalence, a stage at a time: hand-written (view A) against its
// description (view B), the stage's own outputs read back each frame - equal bit for bit (the lit image is not: two views of the prepass
// path differ by their frame-indexed noise) - and its Frame graph scopes the same accesses. First A against A (both hand-written): if two
// views differ the harness is not trusted. Stages: the light tiles (local lights; XLION_TILE=1 and cluster geometry - the G-buffer
// prepass's depth), the reflections' screen trace and reconstruction (screen reflections).
//   xlionrender_test_stage_graph [frames]
#include "xlionrender_test_common.h"
#include "plans/forward/xlionrender_plan_forward.h"
#include <cstring>
#include <functional>
#include <algorithm>

using namespace xlionrender::test;
namespace stage = xlionrender::stage;

// a scope's access lines (sorted: the stage declares reads, then writes) - empty when the scope is not there
static std::vector<std::string> Scope( const std::string& Text, const char* pName )
{
    std::vector<std::string> Out;
    const auto b = Text.find(std::string("\t") + pName + "\t"); if( b == std::string::npos ) return Out;
    const auto s = Text.rfind('\n', b) + 1;
    const auto e = Text.find("\nS\t", b);
    const std::string T = Text.substr(s, e == std::string::npos ? std::string::npos : e - s);
    for( std::size_t x = 0; x < T.size(); ) { auto y = T.find('\n', x); if( y == std::string::npos ) y = T.size(); Out.push_back(T.substr(x, y - x)); x = y + 1; }
    std::sort(Out.begin() + 1, Out.end());
    return Out;
}

int main( int argc, char** argv )
{
    const int nFrames = argc > 1 ? std::atoi(argv[1]) : 4;

    // the lowering's refusals
    {
        stage::registry R;
        xgpu::compute Fake;
        R.m_Kernels.Add("light_tiles", { &Fake, 2, 1, 0, 80, "instance: m_LightTilesCompute" });
        for( const char* p : { "Lights: light ring", "Lights: the screen tiles' lists" } ) R.m_Buffers.Add(p);
        R.m_Textures.Add("Render target: depth");
        for( const char* p : { "view.width", "view.height", "lights.local_base", "lights.local_count" } ) R.SetU32(R.Value(p), 1);
        { float M[16]{}; R.SetFloats(R.Value("frame.c2w"), M, 16); }
        const auto& D = xlionrender::details::LightTilesStage();
        CHECK(stage::Lower(D, R).m_Error.find("no size 'view.size'") != std::string::npos, "a missing size is refused");
        R.SetSize(R.Value("view.size"), 8, 8);
        CHECK(stage::Lower(D, R).m_Error.empty(), "the light tiles lower: %s", stage::Lower(D, R).m_Error.c_str());
        auto Bad = D; Bad.m_Passes[0].m_Kernel = "nope";
        CHECK(stage::Lower(Bad, R).m_Error.find("no kernel 'nope'") != std::string::npos, "an unknown kernel is refused");
        Bad = D; Bad.m_Passes[0].m_Buffers[0].m_Name = "nope";
        CHECK(stage::Lower(Bad, R).m_Error.find("no buffer 'nope'") != std::string::npos, "an unknown buffer is refused");
        Bad = D; Bad.m_Passes[0].m_Textures.clear();
        CHECK(stage::Lower(Bad, R).m_Error.find("bindings do not match") != std::string::npos, "a missing binding is refused");
        Bad = D; Bad.m_Passes[0].m_Push.push_back({ "", 7u });
        CHECK(stage::Lower(Bad, R).m_Error.find("pushes 84 bytes") != std::string::npos, "a push of the wrong size is refused: %s", stage::Lower(Bad, R).m_Error.c_str());
        Bad = D; Bad.m_Passes[0].m_Push[1].m_Frame = "nope";
        CHECK(stage::Lower(Bad, R).m_Error.find("no frame value 'nope'") != std::string::npos, "an unknown frame value is refused");
        Bad = D; Bad.m_Passes[0].m_Size = "view.width";
        CHECK(stage::Lower(Bad, R).m_Error.find("no size 'view.width'") != std::string::npos, "a size of one word is refused");
        Bad = D; Bad.m_Passes[0].m_Buffers[0].m_DeclareIf = "nope";
        CHECK(stage::Lower(Bad, R).m_Error.find("no frame value 'nope'") != std::string::npos, "an unknown declaration condition is refused");
        std::printf("lowering: refusals as expected\n");
    }

    // the tile pass (and its light tiles) is on with XLION_TILE (read at the first frame)
#ifdef _WIN32
    _putenv_s("XLION_TILE", "1");
#else
    setenv("XLION_TILE", "1", 1);
#endif
    app App;
    CHECK(App.Init(), "Init");
    App.m_bQuietTimes = true;
    auto& Renderer = App.m_Renderer;
    // cluster geometry: the G-buffer prepass lays its depth before the stages (plain meshes are not in it)
    cluster_buffers SphereB, BoxB;
    xlionrender::mesh_id Sphere, Box;
    CHECK(MakeStaticBuffers(App.m_Device, MakeSphere(), 0.5f, SphereB) && !Renderer.CreateClusterGeom(Sphere, SphereB.Geom()), "the sphere");
    CHECK(MakeStaticBuffers(App.m_Device, xprim_geom::cube::Generate(1, 1, 1, 1, xprim_geom::float3{ 1, 1, 1 }), 0.5f, BoxB) && !Renderer.CreateClusterGeom(Box, BoxB.Geom()), "the box");
    xlionrender::scene Base;
    Base.m_Draws = { { .m_Mesh = Box, .m_L2W = xmath::fmat4::fromTranslation({ 0, -0.55f, 0 }) * xmath::fmat4::fromScale({ 10.0f, 0.1f, 10.0f }), .m_Material = { .m_BaseColor = { 0.5f, 0.5f, 0.5f }, .m_Roughness = 0.15f } }
                   , { .m_Mesh = Sphere, .m_L2W = xmath::fmat4::fromTranslation({ -0.4f, 0.3f, 0 }), .m_Material = { .m_BaseColor = { 0.9f, 0.9f, 0.9f }, .m_Metallic = 1, .m_Roughness = 0.1f } }
                   , { .m_Mesh = Box, .m_L2W = xmath::fmat4::fromTranslation({ 0.7f, -0.1f, 0.5f }) * xmath::fmat4::fromScale({ 0.4f, 0.8f, 0.4f }), .m_Material = { .m_BaseColor = { 0.8f, 0.2f, 0.1f } } } };
    Base.m_Sun.m_Intensity = 2.0f;
    Base.m_GI.m_bAO = false;        // (its pattern follows the frame: two views a frame apart differ)
    Base.m_Sun.m_bCastShadows = false;      // (the soft shadows' taps too; the reflections read last frame's lit image)
    Base.m_Post.m_bTAA = false; Base.m_Post.m_Bloom = 0.0f; Base.m_Post.m_bAutoExposure = false;
    const int W = 805, H = 377;

    // a frame, then the stage's output buffers copied into buffers the CPU reads (xGPU reads back textures only)
    using output = std::pair<xgpu::buffer*, int>;      // (a buffer and its entry size: xGPU does not say it)
    std::vector<xgpu::buffer> Copies;
    std::vector<int> CopyBytes;
    const auto Frame = [&]( const xlionrender::scene& S, xlionrender::render_view& V, xgpu::tools::view& Cam, const std::vector<output>& Out, std::vector<std::vector<std::uint32_t>>& Words ) -> bool
    {
        auto& Win = App.m_Window;
        if( !App.m_Instance.ProcessInputEvents() ) return false;
        while( Win.BeginRendering() ) if( !App.m_Instance.ProcessInputEvents() ) return false;
        Renderer.Render(Win, S, V, Cam);
        Copies.resize(std::max(Copies.size(), Out.size())); CopyBytes.resize(Copies.size());
        Words.assign(Out.size(), {});
        std::vector<int> n(Out.size(), 0);
        for( std::size_t i = 0; i < Out.size(); ++i )
        {
            auto* pB = Out[i].first; const int Bytes = Out[i].second;
            if( !pB->m_Private ) continue;
            n[i] = pB->getEntryCount() * Bytes / 4;
            auto& C = Copies[i];
            if( C.m_Private && C.getEntryCount() * CopyBytes[i] != pB->getEntryCount() * Bytes ) App.m_Device.Destroy(std::move(C));
            if( !C.m_Private && App.m_Device.Create(C, { .m_Type = xgpu::buffer::type::STORAGE, .m_Usage = xgpu::buffer::setup::usage::CPU_WRITE_GPU_READ, .m_EntryByteSize = Bytes, .m_EntryCount = pB->getEntryCount(), .m_pName = "test: a stage's output read back" }) ) return false;
            CopyBytes[i] = Bytes;
            const xgpu::buffer::copy Cp{ 0, 0, pB->getEntryCount() };
            Win.CopyBuffer(*pB, C, std::span{ &Cp, 1 });
        }
        { auto Cmd = Win.getCmdBuffer(); }
        Win.PageFlip();
        for( std::size_t i = 0; i < Out.size(); ++i )
        {
            Words[i].assign(std::size_t(n[i]), 0u);
            if( n[i] ) (void)Copies[i].MemoryMap(0, Out[i].first->getEntryCount(), [&]( void* p ) { std::memcpy(Words[i].data(), p, std::size_t(n[i]) * 4); });
        }
        return true;
    };

    struct test_case
    {
        const char*                                                 m_pName;
        xlionrender::scene                                          m_Scene;
        std::vector<const char*>                                    m_Scopes;
        std::function<void(xlionrender::render_view&, bool)>        m_Stage;        // the stage on (true) or hand-written
        std::function<std::vector<output>(xlionrender::render_view&)> m_Outputs;
        std::function<const stage::plan&(xlionrender::render_view&)> m_Plan;
        std::vector<const char*>                                    m_OutNames;
    };
    std::vector<test_case> Cases;
    {
        auto S = Base;
        S.m_LocalLights.push_back({ .m_Type = xlionrender::local_light::type::POINT, .m_Position = { -0.6f, 0.5f, -0.6f }, .m_Radius = 1.2f, .m_Color = { 1, 0.6f, 0.3f }, .m_Intensity = 6.0f, .m_bCastShadows = false });
        S.m_LocalLights.push_back({ .m_Type = xlionrender::local_light::type::SPOT, .m_Position = { 1.2f, 0.9f, -0.5f }, .m_Direction = { 0, -1, 0.5f }, .m_Radius = 1.6f, .m_Color = { 0.3f, 0.6f, 1 }, .m_Intensity = 8.0f, .m_bCastShadows = false });
        Cases.push_back({ "light tiles", S, { "light tiles" }, []( xlionrender::render_view& V, bool b ) { V.m_bStageLightTiles = b; }
                        , []( xlionrender::render_view& V ) { return std::vector<output>{ { &V.m_LightTiles, 4 } }; }
                        , []( xlionrender::render_view& V ) -> const stage::plan& { return V.m_LightTilesPlan; }, { "the tiles' lists" } });
    }
    {
        auto S = Base;
        S.m_bReflectionScreen = true; S.m_bReflectionReconstruct = true;
        Cases.push_back({ "reflection screen trace + reconstruction", S, { "reflections: screen trace", "reflections: temporal", "reflections: spatial" }, []( xlionrender::render_view& V, bool b ) { V.m_bStageReflSSR = V.m_bStageReflRecon = b; }
                        , []( xlionrender::render_view& V ) { return std::vector<output>{ { &V.m_TailResult, 16 }, { &V.m_ReflTemporal, 16 }, { &V.m_ReflHistory, 16 } }; }
                        , []( xlionrender::render_view& V ) -> const stage::plan& { return V.m_ReflReconPlan; }, { "the tail's records", "the temporal result", "the history" } });
    }

    const char* pOnly = std::getenv("XLION_SG_ONLY");
    int nFail = 0;
    for( std::size_t c = 0; c < Cases.size(); ++c )
    {
        auto& Ca = Cases[c];
        if( pOnly && std::atoi(pOnly) != int(c) ) continue;
        for( int Control = 1; Control >= 0; --Control )         // 1: hand-written against itself (the harness's own check), 0: against the stage
        {
            xlionrender::render_view A, B;
            CHECK(!Renderer.CreateView(A, { .m_Width = W, .m_Height = H }) && !Renderer.CreateView(B, { .m_Width = W, .m_Height = H }), "CreateView");
            Ca.m_Stage(A, false);
            Ca.m_Stage(B, !Control);
            std::string Why;
            for( int f = 0; f < nFrames && Why.empty(); ++f )
            {
                xgpu::tools::view Camera;
                Camera.setFov(xmath::radian{ 1.0f });
                Camera.setViewport({ 0, 0, W, H });
                Camera.LookAt(xmath::fvec3{ 0.5f + 0.05f * float(f), 1.2f, -2.5f }, xmath::fvec3{ 0, 0, 0 });
                std::vector<std::vector<std::uint32_t>> LA, LB;
                CHECK(Frame(Ca.m_Scene, A, Camera, Ca.m_Outputs(A), LA), "Render A");
                CHECK(Frame(Ca.m_Scene, B, Camera, Ca.m_Outputs(B), LB), "Render B");
                for( const char* pScope : Ca.m_Scopes )
                {
                    const auto TA = Scope(A.m_Graph.Text(), pScope), TB = Scope(B.m_Graph.Text(), pScope);
                    CHECK(!TA.empty(), "%s: the scope '%s' ran", Ca.m_pName, pScope);
                    if( TA != TB && Why.empty() )
                    {
                        Why = std::format("frame {}: the Frame graph's '{}' differs:", f, pScope);
                        for( const auto& l : TA ) Why += "\n   A " + l;
                        for( const auto& l : TB ) Why += "\n   B " + l;
                    }
                }
                for( std::size_t i = 0; i < LA.size() && Why.empty(); ++i )
                {
                    CHECK(!LA[i].empty(), "%s: %s made", Ca.m_pName, Ca.m_OutNames[i]);
                    if( LA[i] == LB[i] ) continue;
                    std::size_t k = 0; while( k < std::min(LA[i].size(), LB[i].size()) && LA[i][k] == LB[i][k] ) ++k;
                    std::size_t nDiff = 0; for( std::size_t j = 0; j < std::min(LA[i].size(), LB[i].size()); ++j ) nDiff += LA[i][j] != LB[i][j];
                    Why = std::format("frame {}: {} differs ({} vs {} words, {} differ), the first at word {}", f, Ca.m_OutNames[i], LA[i].size(), LB[i].size(), nDiff, k);
                }
                if( f == nFrames - 1 && Control && c == 0 )
                {
                    int nTiles = 0, nLit = 0;
                    for( std::size_t t = 16; t + 16 <= LA[0].size(); t += 16 ) { ++nTiles; nLit += LA[0][t] != 0; }
                    std::printf("light tiles: %d tiles, %d with lights\n", nTiles, nLit);
                    CHECK(nLit > 0 && nLit < nTiles, "some tiles have lights, some not (a test that can tell)");
                }
            }
            const auto& P = Ca.m_Plan(B);
            const char* pWhat = Control ? "hand-written against itself" : "hand-written against the stage";
            if( !Control && (!P.m_Error.empty() || P.m_Passes.empty()) ) { Why = "view B did not run the stage: " + P.m_Error; }
            Renderer.DestroyView(A); Renderer.DestroyView(B);
            if( !Why.empty() )
            {
                std::printf("DIFFERENT: %s, %s - %s\n", Ca.m_pName, pWhat, Why.c_str()); ++nFail;
                if( Control ) { std::printf("  (the harness is not trusted for this stage: the stage against it not run)\n"); break; }
            }
            else std::printf("same: %s, %s, %d frames\n", Ca.m_pName, pWhat, nFrames);
        }
    }
    for( auto& C : Copies ) if( C.m_Private ) App.m_Device.Destroy(std::move(C));
    CHECK(g_nErrors == 0, "%d validation errors", g_nErrors);
    if( nFail ) { std::printf("FAIL: %d\n", nFail); return 1; }
    std::printf("PASS\n");
    return 0;
}
