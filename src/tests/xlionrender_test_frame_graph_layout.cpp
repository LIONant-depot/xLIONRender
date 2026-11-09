// The Level Editor's Frame graph, its data side on the CPU: a frame's render graph declarations (one rendered frame of the forward test
// scene) turned into steps and wires the way LevelEditor_RenderGraphTab.h DrawFrameGraph does, laid out and routed by xgraph - every step
// placed at a finite position, every wire routed (a diagnostic print of the layout).
#include "xlionrender_test_common.h"
#include "dependencies/xgraph/source/xgraph_routes.h"
#include <unordered_map>

using namespace xlionrender::test;

int main()
{
    app App;
    CHECK(App.Init(), "Init");
    App.m_bQuietTimes = true;
    auto& Renderer = App.m_Renderer;
    xlionrender::mesh_id Sphere;
    {
        const auto Prim = MakeSphere();
        std::vector<xlionrender::vertex> Verts(Prim.m_Vertices.size());
        for (std::size_t i = 0; i < Verts.size(); ++i)
        {
            const auto& V = Prim.m_Vertices[i];
            Verts[i] = { { V.m_Position.m_X, V.m_Position.m_Y, V.m_Position.m_Z }, { V.m_Texcoord.m_X, V.m_Texcoord.m_Y }, 0xffffffffu, { V.m_Normal.m_X, V.m_Normal.m_Y, V.m_Normal.m_Z }, { V.m_Tangent.m_X, V.m_Tangent.m_Y, V.m_Tangent.m_Z, V.m_Tangent.m_D } };
        }
        CHECK(!Renderer.CreateMesh(Sphere, Verts, Prim.m_Indices), "CreateMesh");
    }
    xlionrender::scene Scene;
    Scene.m_Draws.push_back({ .m_Mesh = Sphere, .m_Material = { .m_BaseColor = { 0.8f, 0.8f, 0.8f } } });
    Scene.m_Post.m_bTAA = true; Scene.m_Post.m_Bloom = 0.3f; Scene.m_Post.m_bAutoExposure = true;
    xlionrender::render_view View;
    CHECK(!Renderer.CreateView(View, { .m_Width = 320, .m_Height = 200 }), "CreateView");
    xgpu::tools::view C; C.setFov(xmath::radian{ 1.0f }); C.setViewport({ 0, 0, 320, 200 }); C.LookAt(xmath::fvec3{ 0, 0, -2 }, xmath::fvec3{ 0, 0, 0 });
    for (int f = 0; f < 3; ++f) CHECK(App.RenderFrame(Scene, View, C), "Render");

    // the graph's text, parsed as the tab parses it
    struct access { bool m_bWrite; std::string m_Version, m_Res; bool m_bKeeps = false; };      // (Keeps: a preserve / partial write: it reads the version before)
    struct scope { std::string m_Name; std::vector<access> m_A; };
    std::vector<scope> Scopes;
    const std::string Text = View.m_Graph.Text();
    for (std::size_t b = 0; b < Text.size(); )
    {
        std::size_t e = Text.find('\n', b); if (e == std::string::npos) e = Text.size();
        std::vector<std::string> F; for (std::size_t x = b; x <= e; ) { std::size_t y = Text.find('\t', x); if (y == std::string::npos || y > e) y = e; F.push_back(Text.substr(x, y - x)); x = y + 1; }
        if (F[0] == "S" && F.size() >= 4) Scopes.push_back({ F[2], {} });
        else if (F[0] == "A" && F.size() >= 7 && !Scopes.empty()) Scopes.back().m_A.push_back({ F[1] == "1", F[2], F[6], F[1] == "1" && (F[4] == "preserve" || F[4] == "partial") });
        b = e + 1;
    }
    namespace gl = xgraph::layout; namespace gr = xgraph::routes;
    std::vector<gl::edge> Edges;
    std::unordered_map<std::string, std::pair<int, int>> Writer;
    gl::input In(1);
    for (int s = 0; s < static_cast<int>(Scopes.size()); ++s)
    {
        std::vector<std::string> Ins, Outs;
        for (const auto& A : Scopes[s].m_A)
        {
            if ((A.m_bWrite && !A.m_bKeeps) || std::find(Ins.begin(), Ins.end(), A.m_Res) != Ins.end()) continue;
            const std::string V = A.m_bKeeps ? std::to_string(std::atoi(A.m_Version.c_str()) - 1) : A.m_Version;
            const int Row = static_cast<int>(Ins.size()); Ins.push_back(A.m_Res);
            if (auto It = Writer.find(A.m_Res + "#" + V); It != Writer.end()) Edges.push_back({ It->second.first, It->second.second, s, Row, std::hash<std::string>{}(A.m_Res + "#" + V), Edges.size() });
        }
        for (const auto& A : Scopes[s].m_A)
        {
            if (!A.m_bWrite) continue;
            int Row = static_cast<int>(std::find(Outs.begin(), Outs.end(), A.m_Res) - Outs.begin());
            if (Row == static_cast<int>(Outs.size())) Outs.push_back(A.m_Res);
            Writer[A.m_Res + "#" + A.m_Version] = { s, Row };
        }
        In[0].m_Steps.push_back({ static_cast<std::uint64_t>(s + 1), 200.0f, static_cast<int>(std::max<std::size_t>(std::max(Ins.size(), Outs.size()), 1)) });
    }
    const auto R = gr::Build(In, Edges);
    // outputs nothing in the frame reads (the last version of a resource: kept for the next frame, or the frame's output - or a broken link)
    {
        std::unordered_map<std::string, int> Read;
        for (const auto& S : Scopes) for (const auto& A : S.m_A) { if (!A.m_bWrite) ++Read[A.m_Res + "#" + A.m_Version]; else if (A.m_bKeeps) ++Read[A.m_Res + "#" + std::to_string(std::atoi(A.m_Version.c_str()) - 1)]; }
        for (const auto& S : Scopes) for (const auto& A : S.m_A) if (A.m_bWrite && !Read.contains(A.m_Res + "#" + A.m_Version)) std::printf("  unread: %s writes %s v%s\n", S.m_Name.c_str(), A.m_Res.c_str(), A.m_Version.c_str());
    }
    std::printf("%zu scopes, %zu edges, %d columns, width %.0f height %.0f, routes not drawn %d\n", Scopes.size(), Edges.size(), R.m_Layout.m_K, R.m_Layout.m_Width, R.m_Layout.m_Height, R.m_Skipped);
    for (std::size_t i = 0; i < Scopes.size() && i < 8; ++i) std::printf("  %-40s x %.0f y %.0f w %.0f h %.0f\n", Scopes[i].m_Name.c_str(), R.m_Layout.m_Steps[i].m_X, R.m_Layout.m_Steps[i].m_Y, R.m_Layout.m_Steps[i].m_W, R.m_Layout.m_Steps[i].m_H);
    for (const auto& B : R.m_Layout.m_Steps) CHECK(std::isfinite(B.m_X) && std::isfinite(B.m_Y) && B.m_X >= 0 && B.m_Y >= 0, "a step at a finite position");
    Renderer.DestroyView(View);
    std::printf("PASS\n");
    return 0;
}
