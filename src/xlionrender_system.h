#ifndef XLIONRENDER_SYSTEM_H
#define XLIONRENDER_SYSTEM_H
#pragma once

// The render system - compiled entirely inside LIONRender.dll (see xlionrender_plugin_entry.cpp, the
// only place RegisterComponents<primitive>()/RegisterSystems<system>() are called). Same reasoning as
// xlioncore::physics::system: xecs::component::type::info_v<T> is per-binary, so registration and
// every query/entity-iteration call for a component have to be compiled into the same binary - this
// one, not xLION.exe. Queries xlioncore::transform for pose (shared with physics); size is
// Transform.Scale. OnUpdate only ever calls
// SubmitInternal (xlionrender_internal.h) - it does not touch xGPU/cmd_buffer at all; the actual draw
// calls happen later, from the host's own turn, via the exported xlionrender::Draw (xlionrender_api.h)
// - see xlionrender_renderer.h's comment.
//
// Collect() (not OnUpdate) is what actually runs the query - called from Draw every frame regardless
// of Play state (see xlionrender_api.cpp). Editor-created entities only become visible to Search/Foreach
// after xecs::archetype::mgr::UpdateStructuralChanges() flushes their pending pool append - the host
// calls that unconditionally each frame too (xlevel_session.h), not just while Playing.
#include "xlionrender_view.h"
#include "xlionrender_primitive.h"
#include "xlionrender_text.h"
#include "xlionrender_text_layout.h"
#include "xlionrender_text_renderer.h"
#include "xlionrender_internal.h"
#include "xlionrender_scene_renderer.h"
#include "dependencies/xLIONCore/src/resources/xlioncore_resources_api.h"
#include "dependencies/xGPU/source/xGPU.h"
// LIONCore.dll owns (registers) these components; this DLL only needs their types/reflection, so
// they are read import-only (see xscript_registration.h). Without this, every compiler that runs
// the dynamic initializers of the headers' inline g_AutoReg_* variables (clang/gcc always do; MSVC
// happens to drop the unreferenced ones) gives LIONRender its own registration of transform,
// static_tag, the physics components, ... - a second bit per GUID, and whichever copy sorts last
// wins findComponentTypeInfo, leaving LIONCore's own info_v copies (which its systems query)
// pointing at a bit no entity has (Linux: the ground's static_tag was invisible to physics, so it
// became a kinematic body). The few LIONCore types this DLL queries are synced by GUID in
// XecsPlugin_RegisterSystems (xlionrender_plugin_entry.cpp).
#define XSCRIPT_IMPORT_ONLY
#include "dependencies/xLIONCore/src/transform/xlioncore_transform.h"
#include "dependencies/xLIONCore/src/transform/xlioncore_hierarchy.h"
#include "dependencies/xLIONCore/src/tags/xlioncore_tags.h"
#include "dependencies/xLIONCore/src/physics/xlioncore_physics.h"
#undef  XSCRIPT_IMPORT_ONLY
#include "plugins/xscript_module.plugin/source/Runtime/xscript_registration.h"   // registering macro back for anything after
#include "dependencies/xeditor_tools/src/xeditor_tools_picking.h"
#include <cstring>
#include <format>
#include <functional>
#include <memory>
#include <limits>
#include <string>
#include <unordered_map>

namespace xlionrender
{
    struct system : xecs::system::instance
    {
        constexpr static auto typedef_v = xecs::system::type::update{ .m_pName = "Render" };
        using query = std::tuple<xecs::query::must<const xlioncore::transform, const primitive>>;

        system(xecs::game_mgr::instance& GameMgr) noexcept : xecs::system::instance(GameMgr), m_pWorld(&GameMgr) {}

        const void* m_pWorld;   // the world this system belongs to (the key the host names when it draws or picks)

        // The layout of every Text of this world, kept until what it depends on changes: the text, the font, the size, the wrap, the spacing, the alignment, the anchor, and the atlas it reads
        // (never the transform, the camera, the color: those are not in a layout). Entries not seen in a frame are dropped.
        struct cached_text
        {
            std::size_t                     m_Hash      = 0;
            text_layout::result             m_Layout;
            xlioncore::resources::font_view m_Font;
            int                             m_Frame     = 0;
        };
        std::unordered_map<std::uint64_t, cached_text>  m_Texts;
        int                                             m_Frame = 0;
        std::unique_ptr<scene_renderer>                 m_pScene;       // the forward renderer of this world (Meshes, lights, Environment), made the first time it is asked to render
        view_options                                    m_ViewOptions;  // the editor's viewing choices (xRenderEditor::SetViewOptions): on top of the Environment, never saved

        static void Mix(std::size_t& Hash, std::size_t Value) noexcept { Hash ^= Value + 0x9E3779B97F4A7C15ull + (Hash << 6) + (Hash >> 2); }
        static std::size_t Bits(float V) noexcept { std::uint32_t B; std::memcpy(&B, &V, sizeof(B)); return B; }

        // The layout of the Text of an entity: null while its font is not there (no font chosen, not compiled yet, or the texture of its atlas not made yet). Rebuilt only when it has changed.
        const text_layout::result* LayoutOf(std::uint64_t Entity, const text& Tx) noexcept
        {
            if (Tx.m_Font.empty()) return nullptr;
            xlioncore::resources::font_view Font;
            if (!xlioncore::resources::GetFont(Tx.m_Font.m_Instance.m_Value, Font)) return nullptr;

            const auto Dimensions = Font.m_pTexture->getTextureDimensions();
            text_layout::settings Settings;
            Settings.m_WrapEm      = Tx.m_WrapWidth > 0.0f && Tx.m_Size > 0.0f ? Tx.m_WrapWidth / Tx.m_Size : 0.0f;
            Settings.m_LineSpacing = Tx.m_LineSpacing;
            Settings.m_HAlign      = Tx.m_HAlign;
            Settings.m_VAnchor     = Tx.m_VAnchor;
            Settings.m_PixelSize   = Tx.m_SizeMode == text_size_mode::SCREEN ? Tx.m_Size : 0.0f;       // BITMAP: seen at that many pixels, or (in the world) the biggest baked size
            Settings.m_AtlasWidth  = static_cast<int>(Dimensions[0]);
            Settings.m_AtlasHeight = static_cast<int>(Dimensions[1]);

            std::size_t Hash = std::hash<std::wstring_view>{}(Tx.m_Text);
            Mix(Hash, reinterpret_cast<std::size_t>(Font.m_pFont));
            Mix(Hash, Bits(Settings.m_WrapEm));      Mix(Hash, Bits(Settings.m_LineSpacing)); Mix(Hash, Bits(Settings.m_PixelSize));
            Mix(Hash, static_cast<std::size_t>(Settings.m_HAlign)); Mix(Hash, static_cast<std::size_t>(Settings.m_VAnchor));
            Mix(Hash, static_cast<std::size_t>(Settings.m_AtlasWidth)); Mix(Hash, static_cast<std::size_t>(Settings.m_AtlasHeight));

            auto& Entry = m_Texts[Entity];
            Entry.m_Frame = m_Frame;
            if (Entry.m_Hash != Hash || Entry.m_Font.m_pFont != Font.m_pFont)
            {
                text_layout::Layout(*Font.m_pFont, Tx.m_Text, Settings, Entry.m_Layout);
                Entry.m_Hash = Hash;
                Entry.m_Font = Font;
            }
            return &Entry.m_Layout;
        }

        // What the editor and its tests ask: the layout of the Text of an entity as lines of text. False when the entity has no Text.
        bool DescribeText(std::uint64_t Entity, std::string& Out) noexcept
        {
            bool bFound = false;
            const auto Found = findEntity(xecs::component::entity{ Entity }, [&](const text& Tx) noexcept
            {
                bFound = true;
                if (Tx.m_Font.empty()) { Out = "DescribeText: no font is chosen"; return; }
                if (const auto* pLayout = LayoutOf(Entity, Tx))
                    Out = std::format("DescribeText: ok\nLines={}\nQuads={}\nMissing={}\nBounds={:.4f},{:.4f},{:.4f},{:.4f}", pLayout->m_nLines, pLayout->m_Quads.size(), pLayout->m_nMissing
                                     , pLayout->m_Min.m_X, pLayout->m_Min.m_Y, pLayout->m_Max.m_X, pLayout->m_Max.m_Y);
                else
                    Out = "DescribeText: the font is not loaded (it is not compiled yet, or its atlas is not made)";
            });
            (void)Found;
            return bFound;
        }

        void OnCreate(void)  noexcept { SetActiveSystemInternal(m_pWorld, this); }
        void OnDestroy(void) noexcept { SetActiveSystemInternal(m_pWorld, nullptr); }
        void OnUpdate(void)  noexcept {} // required by xECS (else it Foreach's operator()); work is in Collect

        // The axes of the text in the world, as long as the scale of the entity (the shader replaces them when the text faces the camera, keeping the lengths).
        // Screen size mode ignores the scale: the size is in pixels, only the position places the text.
        static void TextAxes(const xlioncore::world_pose& T, const text& Tx, xmath::fvec3& AxisX, xmath::fvec3& AxisY) noexcept
        {
            xmath::fmat4 L2W;
            L2W.setupSRT(T.m_Scale, T.m_Rotation, T.m_Position);
            const xmath::fvec4 X = L2W * xmath::fvec4(1.0f, 0.0f, 0.0f, 0.0f);
            const xmath::fvec4 Y = L2W * xmath::fvec4(0.0f, 1.0f, 0.0f, 0.0f);
            AxisX = xmath::fvec3(X.m_X, X.m_Y, X.m_Z);
            AxisY = xmath::fvec3(Y.m_X, Y.m_Y, Y.m_Z);
            if (Tx.m_SizeMode == text_size_mode::SCREEN) { AxisX.NormalizeSafe(); AxisY.NormalizeSafe(); }
        }

        // What a view leaves out of the world: what the game says is not drawn (no_render, in every view) and, in the editor's scene view, what the editor hid (editor_no_render). An entity with
        // an exclusive disable tag (the game's or the editor's) is not in any query to begin with.
        static void LeaveOut(xecs::query::instance& Query, view View) noexcept
        {
            Query.m_NoneOf.AddFromComponents<xlioncore::no_render_tag>();
            if (View == view::SCENE) Query.m_NoneOf.AddFromComponents<xecs::editor::no_render_tag>();
        }

        // The Texts of the world: their layouts are kept up to date (the drawing of them comes next to this). The resource system of the core starts here, the first time there is a device.
        void CollectText(view View, const roles* pRoles = nullptr) noexcept
        {
            ++m_Frame;
            if (auto* pDevice = GetDeviceInternal())
                xlioncore::resources::Initialize(*pDevice, getGameMgr().m_SceneMgr.m_ProjectPath.c_str());

            xecs::query::instance Query;
            Query.m_Must.AddFromComponents<xlioncore::transform, text>();
            LeaveOut(Query, View);
            auto S = Search(Query);
            int nSeen = 0, nWithLayout = 0;
            Foreach(S, [&](const xecs::component::entity& Ent, const xlioncore::transform& Local, const text& Tx, const xecs::component::parent* pParent, const xlioncore::render_transform* pRender) noexcept
            {
                if (pRoles && (pRoles->IsHidden(Ent.m_Value) || pRoles->IsContext(Ent.m_Value))) return;       // the Texts of the context are not drawn (the fade quad is for solids), nor those of the hidden instance
                const auto T = xlioncore::WorldOf(Local, pParent, pRender, xlioncore::FixedInterpolateOf(getGameMgr()));   // a child's Transform is relative to its parent: what is drawn is at its world pose
                ++nSeen;
                const auto* pLayout = LayoutOf(Ent.m_Value, Tx);
                if (!pLayout) return;
                ++nWithLayout;

                // The same guard as the Primitive: one entity with a degenerate scale must not take the frame down.
                if (!T.m_Scale.isFinite() || T.m_Scale.m_X == 0.0f || T.m_Scale.m_Y == 0.0f || T.m_Scale.m_Z == 0.0f) return;

                xmath::fvec3 AxisX, AxisY;
                TextAxes(T, Tx, AxisX, AxisY);
                g_TextRenderer.Submit(m_Texts[Ent.m_Value].m_Font, *pLayout, Tx, T.m_Position, AxisX, AxisY);
            });

            g_TextRenderer.NoteCollected(nSeen, nWithLayout);
            std::erase_if(m_Texts, [&](const auto& Pair) noexcept { return m_Frame - Pair.second.m_Frame > 3; });       // (not "this one": two views collect in turn, and each sees entities the other does not)
        }

        // pRoles (the editor's session, null: every entity is the document): the hidden entities are not submitted; the context ones are, and the renderer draws them first, faded (xlionrender::roles).
        void Collect(view View = view::SCENE, const roles* pRoles = nullptr) noexcept
        {
            xlioncore::PropagateHierarchy(*this);       // the world pose of every child, from the Transforms as they are now (this is after everything that moves things)
            CollectText(View, pRoles);
            xecs::query::instance Query;
            Query.m_Must.AddFromComponents<xlioncore::transform, primitive>();
            LeaveOut(Query, View);
            auto S = Search(Query);

            Foreach(S, [&](const xecs::component::entity& Ent, const xlioncore::transform& Local, const primitive& Prim, const xecs::component::parent* pParent, const xlioncore::render_transform* pRender) noexcept
            {
                if (pRoles && pRoles->IsHidden(Ent.m_Value)) return;
                const auto T = xlioncore::WorldOf(Local, pParent, pRender, xlioncore::FixedInterpolateOf(getGameMgr()));   // a child's Transform is relative to its parent: what is drawn is at its world pose
                // setupSRT asserts on a zero/non-finite scale (a degenerate world matrix has no
                // inverse, breaking anything downstream that needs one) - correct as a contract
                // check on ITS OWN inputs, but Transform.Scale here comes from arbitrary upstream
                // data (a saved level, a script, a gizmo drag, hand-typed Inspector zeros), any one
                // of which corrupting a single entity must not crash the whole render loop for
                // every OTHER entity too. Direct user report: a zeroed Scale on one entity in a
                // saved level asserted (Debug) on load. Skip just that entity's draw instead.
                if (!T.m_Scale.isFinite() || T.m_Scale.m_X == 0.0f || T.m_Scale.m_Y == 0.0f || T.m_Scale.m_Z == 0.0f)
                    return;

                // Unit mesh half-extents are 0.5 (ShapeLocalHalfExtents). Size is Transform.Scale
                // only: setupSRT(Scale, ...) => world half-extents = 0.5 * Scale. Outline Draw uses
                // this same L2W (Item.m_L2W shared with the solid draw).
                xmath::fmat4 L2W;
                L2W.setupSRT(T.m_Scale, T.m_Rotation, T.m_Position);
                SubmitInternal(Prim.m_Shape, L2W, T.m_Scale, Prim.m_Color, Ent.m_Value);
            });
        }

        // The world's Meshes, lights and Environment as this frame's scene, rendered into the world's view (xlionrender_scene_renderer.h); the host
        // blits it (BlitScene). Called while the frame is built, before the UI is rendered: it records render passes.
        void RenderScene(xgpu::window& Window, xgpu::tools::view& Camera, int W, int H, view View, scene_renderer::overlay_fn pOverlay, void* pUser, const roles* pRoles) noexcept
        {
            auto* pDevice = GetDeviceInternal();
            if (!pDevice) return;
            xlioncore::resources::Initialize(*pDevice, getGameMgr().m_SceneMgr.m_ProjectPath.c_str());
            if (!m_pScene) m_pScene = std::make_unique<scene_renderer>();
            auto& R = *m_pScene;
            if (!R.Init(*pDevice)) return;

            xlioncore::PropagateHierarchy(*this);
            R.Begin();
            R.m_Scene.m_SelectedUser = GetSelectedInternal();
            const auto Interpolate = xlioncore::FixedInterpolateOf(getGameMgr());
            const auto L2WOf = [&](const xlioncore::transform& Local, const xecs::component::parent* pParent, const xlioncore::render_transform* pRender, xmath::fmat4& L2W) noexcept
            {
                const auto T = xlioncore::WorldOf(Local, pParent, pRender, Interpolate);
                if (!T.m_Scale.isFinite() || T.m_Scale.m_X == 0.0f || T.m_Scale.m_Y == 0.0f || T.m_Scale.m_Z == 0.0f) return false;    // a degenerate scale: not drawn (as the Primitive)
                L2W.setupSRT(T.m_Scale, T.m_Rotation, T.m_Position);
                return true;
            };
            const auto Hidden = [&](std::uint64_t Entity) noexcept { return pRoles && pRoles->IsHidden(Entity); };
            const auto Each = [&]<typename T_COMPONENT>(auto&& Fn)
            {
                xecs::query::instance Query;
                Query.m_Must.AddFromComponents<xlioncore::transform, T_COMPONENT>();
                LeaveOut(Query, View);
                auto S = Search(Query);
                Foreach(S, [&](const xecs::component::entity& Ent, const xlioncore::transform& Local, const T_COMPONENT& C, const xecs::component::parent* pParent, const xlioncore::render_transform* pRender) noexcept
                {
                    xmath::fmat4 L2W;
                    if (Hidden(Ent.m_Value) || !L2WOf(Local, pParent, pRender, L2W)) return;
                    Fn(Ent.m_Value, C, L2W);
                });
            };
            {   // first: the meshes need to know whether GI is on (their surfels are baked when it is)
                xecs::query::instance Query;
                Query.m_Must.AddFromComponents<environment>();
                auto S = Search(Query);
                bool bFirst = true;
                Foreach(S, [&](const environment& E) noexcept { if (bFirst) R.SetEnvironment(E); bFirst = false; });
                auto& G = R.m_Scene.m_GI;                                   // the editor's viewing choices on top (view_options)
                if (m_ViewOptions.m_GIDebug >= 0) G.m_Debug = static_cast<gi_debug>(m_ViewOptions.m_GIDebug);
                if (m_ViewOptions.m_bNoGI)        G.m_bEnable   = false;
                if (m_ViewOptions.m_bNoPerPixel)  G.m_bPerPixel = false;
                if (m_ViewOptions.m_bNoMoments)   G.m_bMoments  = false;
                if (m_ViewOptions.m_GITraceAllowanceMs >= 0) G.m_TraceAllowanceMs = m_ViewOptions.m_GITraceAllowanceMs;
            }
            R.m_Renderer.m_bBoxesOnly    = m_ViewOptions.m_bBoxesOnly;         // huge meshes: their boxes, or the clusters a frame budget allows
            R.m_Renderer.m_FrameBudgetMs = m_ViewOptions.m_FrameBudgetMs;
            R.m_Renderer.m_bGPUCull      = !m_ViewOptions.m_bNoGPUCull;        // GPU cluster culling: on, its occlusion, frozen
            R.m_Renderer.m_bGPUOcclusion = !m_ViewOptions.m_bNoGPUOcclusion;
            R.m_Renderer.m_bFreezeCull   = m_ViewOptions.m_bFreezeCull;
            {   // the Meshes, with their MeshMaterial when they have one (old files: OldMaterials lists them for the editor to convert)
                xecs::query::instance Query;
                Query.m_Must.AddFromComponents<xlioncore::transform, mesh>();
                LeaveOut(Query, View);
                auto S = Search(Query);
                m_OldMaterials.clear();
                Foreach(S, [&](const xecs::component::entity& Ent, const xlioncore::transform& Local, const mesh& C, const mesh_material* pMaterial, const xecs::component::parent* pParent, const xlioncore::render_transform* pRender) noexcept
                {
                    if (HasOldMaterials(C, pMaterial)) m_OldMaterials.push_back(Ent.m_Value);
                    xmath::fmat4 L2W;
                    if (Hidden(Ent.m_Value) || !L2WOf(Local, pParent, pRender, L2W)) return;
                    R.AddMesh(C, pMaterial, L2W, Ent.m_Value);
                });
            }
            Each.template operator()<light_sun>  ([&](std::uint64_t,   const light_sun& C,   const xmath::fmat4& L2W) { R.AddSun(C, L2W); });
            Each.template operator()<light_point>([&](std::uint64_t,   const light_point& C, const xmath::fmat4& L2W) { R.AddPoint(C, L2W * xmath::fvec3(0, 0, 0)); });
            Each.template operator()<light_spot> ([&](std::uint64_t,   const light_spot& C,  const xmath::fmat4& L2W) { R.AddSpot(C, L2W); });
            R.Render(Window, Camera, W, H, pOverlay, pUser);
        }

        void BlitScene(xgpu::cmd_buffer& CmdBuffer) noexcept { if (m_pScene) m_pScene->Blit(CmdBuffer); }

        // Old files' material data (a MeshMaterial, the Mesh's deprecated Materials list): the Meshes the last RenderScene found with it, and one
        // of them converted into its Mesh's own slots (UpgradeOldMaterials: false while its materials load, or when it has none)
        std::vector<std::uint64_t> m_OldMaterials;
        bool UpgradeOldMaterialsOf(std::uint64_t Entity) noexcept
        {
            bool bDone = false;
            xecs::query::instance Query;
            Query.m_Must.AddFromComponents<mesh>();
            auto S = Search(Query);
            Foreach(S, [&](const xecs::component::entity& Ent, mesh& M, mesh_material* pOld) noexcept
            {
                if (!bDone && Ent.m_Value == Entity) bDone = UpgradeOldMaterials(M, pOld);
            });
            return bDone;
        }

        // What the last RenderScene had (the editor's DescribeScene, tests)
        std::string DescribeScene() const noexcept
        {
            if (!m_pScene) return "DescribeScene: not rendered yet";
            const auto& S  = m_pScene->m_Stats;
            const auto& GI = m_pScene->m_Renderer.getGIStats();
            const auto  C  = m_pScene->m_Renderer.getCullStats();     // (the clusters kept: read back a few frames late)
            const auto  L  = m_pScene->m_Renderer.getLightStats();
            return std::format("DescribeScene: ok\nMeshes={}\nDrawn={}\nWaiting={}\nConstantSlots={}\nMaterials={}\nSun={}\nLights={}\nDrawsRendered={}\nGIMeshes={}\nGIUpdated={}\nGIRays={}\nGIMs={:.2f}\nGILoadMs={:.0f}\nGILoadParts={:.0f}/{:.0f}/{:.0f}\nGIBad={}\nGICaptured={}\nGIEmissive={}\nGIAlbedo={:.3f},{:.3f},{:.3f}\nGILinks={}\nGIMaxEmitted={:.4f}\nGIMaxEmitter={:.3f},{:.3f},{:.3f}\nGICapturing={}\nGIUnsupported={}\nGITraceMs={:.2f}\nGIUploadMs={:.2f}\nBoxesOnly={}\nFrameBudgetMs={:.0f}\nClustersInView={}\nClustersDrawn={}\nBoxes={}\nIndexAllowance={}\nBudgetFrameMs={:.1f}\nGPUCullCandidates={}\nGPUCullRuns={}\nGPUCullDrawn={}\nGPUCullPrepass={}\nGPUCullPass2={}\nGPUCullFrozen={}\nGPUCullTriangles={}\nGPUCullOcclusionSkipped={}\nGPUCullOccludedShare={:.2f}\nGPUCullMs={:.3f}\nGPUCullOccludeMs={:.3f}"
                              , S.m_Meshes, S.m_Drawn, S.m_Waiting, S.m_ConstantSlots, S.m_Materials, S.m_bSun, S.m_Lights, m_pScene->m_Renderer.getDrawsRendered()
                              , S.m_GIMeshes, GI.m_Updated, GI.m_Rays, GI.m_Ms, m_pScene->m_TotalGILoadMs, m_pScene->m_GILoadPartMs[0], m_pScene->m_GILoadPartMs[1], m_pScene->m_GILoadPartMs[2], GI.m_nBad, GI.m_nColored, GI.m_nEmissive, GI.m_MeanAlbedo.m_X, GI.m_MeanAlbedo.m_Y, GI.m_MeanAlbedo.m_Z, GI.m_nLinks, GI.m_MaxEmitted, GI.m_MaxEmitter.m_X, GI.m_MaxEmitter.m_Y, GI.m_MaxEmitter.m_Z, int(m_pScene->m_Renderer.isGICapturingAny()), GI.m_nUnsupported, GI.m_TraceMs, GI.m_UploadMs
                              , int(m_pScene->m_Renderer.m_bBoxesOnly), m_pScene->m_Renderer.m_FrameBudgetMs, m_pScene->m_Renderer.getBudgetStats().m_nClusters, m_pScene->m_Renderer.getBudgetStats().m_nDrawn
                              , m_pScene->m_Renderer.getNotDrawn().size(), m_pScene->m_Renderer.getBudgetStats().m_Allowance, m_pScene->m_Renderer.getBudgetStats().m_FrameMs
                              , C.m_nCandidates, C.m_nRuns, C.m_nDrawn, C.m_nPrepassDrawn, C.m_nDrawnPass2, int(C.m_bFrozen), C.m_nTriangles, int(C.m_bOcclusionSkipped), C.m_OccludedShare, C.m_CullGpuMs, C.m_OccludeGpuMs)
                 + std::format("\nLightsLocal={}\nLightsDropped={}\nLightsShadowed={}\nLightsShadowFallback={}\nLightsShadowStale={}\nLightCellsOverflow={}"
                              , L.m_nLights, L.m_nDropped, L.m_nShadowed, L.m_nShadowFallback, L.m_nShadowStale, L.m_nOverflowCells);
        }

        // The GI's main-thread time in the last RenderScene, all of it (the update and the upload: what the host's shared allowance is charged)
        float GITraceMs() const noexcept { return m_pScene ? float(m_pScene->m_Renderer.getGIStats().m_UpdateGIMs) : 0.0f; }

        // The GI's surfels of the last RenderScene to a CSV file (gi::world::Dump): the surfels written, -1 when there is no GI
        int DumpGI(const char* pPath) const noexcept { return m_pScene ? m_pScene->m_Renderer.DumpGI(pPath) : -1; }

        // Ray-vs-OBB: Transform Position/Rotation + half-extents 0.5*Scale (matches Collect L2W).
        // Closest hit wins. MaxT caps the ray (e.g. ground-plane hit) so empty floor clears
        // selection instead of selecting through distant AABBs.
        // The box of an entity (what Pick tests): its center, rotation and half extents in the world. False: none (a light, a text, not there)
        bool EntityBox(std::uint64_t Entity, xmath::fvec3& Center, xmath::fquat& Rotation, xmath::fvec3& Half) noexcept
        {
            bool bFound = false;
            xecs::query::instance MeshQuery;
            MeshQuery.m_Must.AddFromComponents<xlioncore::transform, mesh>();
            auto MS = Search(MeshQuery);
            Foreach(MS, [&](const xecs::component::entity& Ent, const xlioncore::transform& Local, const mesh& M, const xecs::component::parent* pParent, const xlioncore::render_transform* pRender) noexcept
            {
                if (bFound || Ent.m_Value != Entity || M.m_Geom.empty()) return;
                auto* pGeom = static_cast<xgeom_static::xgpu::geom*>(xlioncore::resources::GetResource(geom_static_type_guid_v.m_Value, M.m_Geom.m_Instance.m_Value));
                if (!pGeom) return;
                const auto T = xlioncore::WorldOf(Local, pParent, pRender, xlioncore::FixedInterpolateOf(getGameMgr()));
                if (!T.m_Scale.isFinite()) return;
                const xmath::fvec3 LocalCenter = (pGeom->m_BBox.m_Min + pGeom->m_BBox.m_Max) * 0.5f;
                Half     = ((pGeom->m_BBox.m_Max - pGeom->m_BBox.m_Min) * 0.5f * T.m_Scale).AbsCopy();
                Center   = T.m_Position + T.m_Rotation.RotateVector(LocalCenter * T.m_Scale);
                Rotation = T.m_Rotation;
                bFound   = true;
            });
            if (bFound) return true;
            xecs::query::instance Query;
            Query.m_Must.AddFromComponents<xlioncore::transform, primitive>();
            auto S = Search(Query);
            Foreach(S, [&](const xecs::component::entity& Ent, const xlioncore::transform& Local, const primitive&, const xecs::component::parent* pParent, const xlioncore::render_transform* pRender) noexcept
            {
                if (bFound || Ent.m_Value != Entity) return;
                const auto T = xlioncore::WorldOf(Local, pParent, pRender, xlioncore::FixedInterpolateOf(getGameMgr()));
                if (!T.m_Scale.isFinite()) return;
                Center = T.m_Position; Rotation = T.m_Rotation; Half = (T.m_Scale * 0.5f).AbsCopy(); bFound = true;
            });
            return bFound;
        }

        std::uint64_t Pick(const xmath::fvec3& Origin, const xmath::fvec3& Dir, float MaxT = std::numeric_limits<float>::max(), view View = view::SCENE, const roles* pRoles = nullptr) noexcept
        {
            xecs::query::instance Query;
            Query.m_Must.AddFromComponents<xlioncore::transform, primitive>();
            LeaveOut(Query, View);
            auto S = Search(Query);

            xeditor_tools::picking::closest_hit<std::uint64_t> Hit;
            // The context and the hidden are never picked: nothing of them has an id to find (editing in context, prefabs_plan.md phase 7).
            const auto NotPickable = [&](std::uint64_t Entity) noexcept { return pRoles && (pRoles->IsContext(Entity) || pRoles->IsHidden(Entity)); };
            Foreach(S, [&](const xecs::component::entity& Ent, const xlioncore::transform& Local, const primitive&, const xecs::component::parent* pParent, const xlioncore::render_transform* pRender) noexcept
            {
                if (NotPickable(Ent.m_Value)) return;
                const auto T = xlioncore::WorldOf(Local, pParent, pRender, xlioncore::FixedInterpolateOf(getGameMgr()));   // a child's Transform is relative to its parent: what is drawn is at its world pose
                // Same degenerate-scale guard as Collect() above - a zeroed OBB shouldn't ever be
                // hit, but there's no reason to feed RayOBBIntersect garbage either.
                if (!T.m_Scale.isFinite() || T.m_Scale.m_X == 0.0f || T.m_Scale.m_Y == 0.0f || T.m_Scale.m_Z == 0.0f)
                    return;

                float THit;
                const xmath::fvec3 LocalHalfExtents = T.m_Scale * 0.5f;
                if (xeditor_tools::picking::RayOBBIntersect(Origin, Dir, T.m_Position, T.m_Rotation, LocalHalfExtents, THit)
                    && THit < MaxT)
                    Hit.Consider(Ent.m_Value, THit);
            });

            // The Texts: the box of the text (by its metrics) as two triangles in the plane it is drawn in. A text that faces the camera is in the plane across the ray (horizontal reading
            // direction: the roll of the camera is not known here). Screen size texts are not picked: their size in the world depends on the camera, which a ray does not carry.
            xecs::query::instance TextQuery;
            TextQuery.m_Must.AddFromComponents<xlioncore::transform, text>();
            LeaveOut(TextQuery, View);
            auto TS = Search(TextQuery);
            Foreach(TS, [&](const xecs::component::entity& Ent, const xlioncore::transform& Local, const text& Tx, const xecs::component::parent* pParent, const xlioncore::render_transform* pRender) noexcept
            {
                if (NotPickable(Ent.m_Value)) return;
                const auto T = xlioncore::WorldOf(Local, pParent, pRender, xlioncore::FixedInterpolateOf(getGameMgr()));   // a child's Transform is relative to its parent: what is drawn is at its world pose
                if (Tx.m_SizeMode == text_size_mode::SCREEN || Tx.m_Opacity <= 0.0f) return;
                if (!T.m_Scale.isFinite() || T.m_Scale.m_X == 0.0f || T.m_Scale.m_Y == 0.0f || T.m_Scale.m_Z == 0.0f) return;
                const auto* pLayout = LayoutOf(Ent.m_Value, Tx);
                if (!pLayout || pLayout->m_Quads.empty()) return;

                xmath::fvec3 AxisX, AxisY;
                TextAxes(T, Tx, AxisX, AxisY);
                if (Tx.m_Orientation != text_orientation::ENTITY)
                {
                    const float ScaleX = AxisX.Length(), ScaleY = AxisY.Length();
                    AxisX = Dir.Cross(xmath::fvec3(0.0f, 1.0f, 0.0f));
                    if (AxisX.Length() < 1.0e-4f) return;                   // looking straight along the vertical: no reading direction to build
                    AxisX.NormalizeSafe();
                    AxisY = AxisX.Cross(Dir);
                    AxisY.NormalizeSafe();
                    AxisX *= ScaleX; AxisY *= ScaleY;
                }
                const auto Corner = [&](float X, float Y) noexcept { return T.m_Position + AxisX * (X * Tx.m_Size) + AxisY * (Y * Tx.m_Size); };
                const auto P0 = Corner(pLayout->m_Min.m_X, pLayout->m_Min.m_Y), P1 = Corner(pLayout->m_Max.m_X, pLayout->m_Min.m_Y);
                const auto P2 = Corner(pLayout->m_Max.m_X, pLayout->m_Max.m_Y), P3 = Corner(pLayout->m_Min.m_X, pLayout->m_Max.m_Y);
                float THit;
                if ((xeditor_tools::picking::RayTriangleIntersect(Origin, Dir, P0, P1, P2, THit) || xeditor_tools::picking::RayTriangleIntersect(Origin, Dir, P0, P2, P3, THit)) && THit < MaxT)
                    Hit.Consider(Ent.m_Value, THit);
            });
            // The Meshes: the box of their geometry (its bounds, scaled, rotated and placed by the entity). A geometry not loaded yet is not picked.
            // ponytail: the bounds, not the triangles - a click in the empty part of a big mesh's box picks it; ray vs the clusters when it shows
            xecs::query::instance MeshQuery;
            MeshQuery.m_Must.AddFromComponents<xlioncore::transform, mesh>();
            LeaveOut(MeshQuery, View);
            auto MS = Search(MeshQuery);
            Foreach(MS, [&](const xecs::component::entity& Ent, const xlioncore::transform& Local, const mesh& M, const xecs::component::parent* pParent, const xlioncore::render_transform* pRender) noexcept
            {
                if (NotPickable(Ent.m_Value) || M.m_Geom.empty()) return;
                auto* pGeom = static_cast<xgeom_static::xgpu::geom*>(xlioncore::resources::GetResource(geom_static_type_guid_v.m_Value, M.m_Geom.m_Instance.m_Value));
                if (!pGeom) return;
                const auto T = xlioncore::WorldOf(Local, pParent, pRender, xlioncore::FixedInterpolateOf(getGameMgr()));
                if (!T.m_Scale.isFinite() || T.m_Scale.m_X == 0.0f || T.m_Scale.m_Y == 0.0f || T.m_Scale.m_Z == 0.0f) return;
                const xmath::fvec3 LocalCenter = (pGeom->m_BBox.m_Min + pGeom->m_BBox.m_Max) * 0.5f;
                const xmath::fvec3 Half        = (pGeom->m_BBox.m_Max - pGeom->m_BBox.m_Min) * 0.5f * T.m_Scale;
                const xmath::fvec3 Center      = T.m_Position + T.m_Rotation.RotateVector(LocalCenter * T.m_Scale);
                float THit;
                if (xeditor_tools::picking::RayOBBIntersect(Origin, Dir, Center, T.m_Rotation, Half.AbsCopy(), THit) && THit < MaxT)
                    Hit.Consider(Ent.m_Value, THit);
            });

            return Hit.isValid() ? Hit.m_Id : xecs::component::entity::invalid_entity_v;
        }
    };
}

#endif // XLIONRENDER_SYSTEM_H