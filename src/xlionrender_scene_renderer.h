#ifndef XLIONRENDER_SCENE_RENDERER_H
#define XLIONRENDER_SCENE_RENDERER_H
#pragma once

// One world's forward renderer (xlionrender::instance): the Level's Meshes, lights and Environment (xlionrender_scene_components.h) become a
// scene snapshot each frame, drawn into this world's own render_view; the host blits that into its viewport. Owned by the world's render system
// (xlionrender_system.h), so every open Level has its own.
//
// Resources come from the resource system of the core this DLL is bound to (xlioncore::resources::GetResource): the GeomStatic's GPU buffers
// become a mesh of the renderer (referenced, not copied), a MaterialInstance its material (the Material's HDR shader + the instance's textures).
// A resource not loaded yet is asked for again next frame; until then a mesh is not drawn and a material slot draws with the constant material.
// ponytail: renderer meshes/materials are kept until the world closes (a recompiled resource is a new pointer: a new entry); evict when it shows.
#include "xlionrender_instance.h"
#include "xlionrender_scene_components.h"
#include "dependencies/xLIONCore/src/resources/xlioncore_resources_api.h"

#include "plugins/xtexture.plugin/source/xtexture_xgpu_rsc_loader.h"
#include "plugins/xmaterial.plugin/source/xmaterial_xgpu_rsc_loader.h"
#include "plugins/xmaterial_instance.plugin/source/xmaterial_instance_xgpu_rsc_loader.h"
#include "plugins/xmaterial.plugin/source/xmaterial_runtime.h"
#include "plugins/xmaterial_instance.plugin/source/xmaterial_instance_runtime.h"
#include "plugins/xgeom_static.plugin/source/xgeom_static_xgpu_rsc_loader.h"
#include "plugins/xgeom_static.plugin/source/xgeom_static_xgpu_runtime.h"

#include <unordered_map>
#include <unordered_set>

namespace xlionrender
{
    struct scene_renderer
    {
        instance                                        m_Renderer;
        render_view                                     m_View;
        scene                                           m_Scene;
        bool                                            m_bReady    = false;
        bool                                            m_bFailed   = false;
        std::unordered_map<const void*, mesh_id>        m_Meshes;       // per loaded geometry
        struct made_material { material_id m_Id; std::vector<const void*> m_From; };     // m_From: the instance, its material, its textures it was made from
        std::unordered_map<std::uint64_t, made_material> m_Materials;  // per material instance (instance guid)
        std::uint32_t                                   m_UpdateCount = 0;              // xlioncore::resources::UpdateCount when the materials were last checked
        instance_ids                                    m_InstanceIds;                  // per drawn entity: its draw's instance id

        struct stats
        {
            int m_Meshes = 0, m_Drawn = 0, m_Waiting = 0, m_Lights = 0, m_bSun = 0, m_Materials = 0, m_ConstantSlots = 0, m_GIMeshes = 0;
        };
        double                                          m_GILoadPartMs[3] = { 0, 0, 0 };   // the first uses: decode (main), surfels (worker), renderer (main) (Phase 7)
        struct gi_load { std::atomic<bool> m_bDone{ false }; bool m_bOk = false; double m_WorkerMs = 0; gi::surfel_mesh m_Surfels; std::vector<std::uint32_t> m_Packed; };
        std::unordered_map<const void*, std::shared_ptr<gi_load>>  m_GILoads;      // Phase 7: the surfels being made on a worker, per geometry
        double                                          m_TotalGILoadMs = 0;           // every geometry's surfels made ready so far (DescribeScene)
        std::unordered_map<const void*, int>            m_GIBaked;
        std::unordered_set<const void*>                 m_OutlineMade;  // per geometry: its outline mesh given to the renderer (made the first time it is the selected one)      // per geometry: its surfels given to the renderer (1), or it has none (0)
        stats                                           m_Stats;

        bool Init(xgpu::device& Device) noexcept
        {
            if (m_bReady || m_bFailed) return m_bReady;
            if (m_Renderer.Init(Device)) { m_bFailed = true; return false; }
            m_bReady = true;
            return true;
        }

        // The renderer's mesh of a geometry (null: the geometry is not loaded yet)
        const mesh_id* MeshOf(const geom_static_ref& Ref) noexcept
        {
            if (Ref.empty()) return nullptr;
            auto* pGeom = static_cast<xgeom_static::xgpu::geom*>(xlioncore::resources::GetResource(geom_static_type_guid_v.m_Value, Ref.m_Instance.m_Value));
            if (!pGeom) return nullptr;
            auto [It, bNew] = m_Meshes.try_emplace(pGeom);
            if (bNew && m_Renderer.CreateClusterGeom(It->second, MakeClusterGeom(*pGeom))) { m_Meshes.erase(It); return nullptr; }
            // GI on: the surfels the GeomStatic compiler baked, with the rays' structures built from the geometry's CPU copy, the first time.
            // Phase 7: the geometry's arrays are copied here; it is decoded, and its surfels, the rays' BVH, the read grid and the GPU
            // buffer's data are made on a worker (0.6 s for the Police building, all of it on the main thread before); when they are
            // ready the buffer is made here and they go to the renderer. Until then the
            // mesh has no surfels: it draws with the declared fallback (the scene's ambient)
            if (m_Scene.m_GI.m_bEnable && !m_GIBaked.contains(pGeom))
            {
                const auto Ms = [](std::chrono::steady_clock::time_point A) { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - A).count(); };
                auto& P = m_GILoads[pGeom];
                const auto T0 = std::chrono::steady_clock::now();
                if (!P)
                {
                    P = std::make_shared<gi_load>();
                    auto Copy = std::make_shared<gi::geom_gi_copy<xgeom_static::xgpu::geom>>(*pGeom);
                    m_GILoadPartMs[0] += Ms(T0);                               // (where a first use goes) the geometry's arrays copied, here
                    gi::Scheduler().SubmitLambda(xscheduler::str_v<"GI mesh">, [P, Copy]() noexcept
                    {
                        const auto T1 = std::chrono::steady_clock::now();
                        gi::FromXGeomStatic(P->m_Surfels, *Copy);              // decoded on the worker too
                        P->m_bOk = gi::SurfelsFromXGeomStatic(P->m_Surfels, *Copy) && P->m_Surfels.m_Levels[0].size() < 0xFFFF;
                        if (P->m_bOk && P->m_Surfels.m_ReadTable.empty()) P->m_Surfels.BuildReadGrid();
                        if (P->m_bOk) P->m_Packed = PackMeshGI(P->m_Surfels, {}, {});             // the GPU buffer's data too
                        P->m_WorkerMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - T1).count();
                        P->m_bDone.store(true, std::memory_order_release);
                    });
                }
                else if (P->m_bDone.load(std::memory_order_acquire))
                {
                    m_GILoadPartMs[1] += P->m_WorkerMs;                        // the surfels and the rays' structures, on the worker
                    const bool bGI = P->m_bOk && !m_Renderer.SetMeshGI(It->second, std::move(P->m_Surfels), std::move(P->m_Packed));
                    m_GILoadPartMs[2] += Ms(T0);                               // handed to the renderer (its GPU buffers), here
                    m_GIBaked[pGeom] = bGI;
                    m_GILoads.erase(pGeom);
                }
                const double LoadMs = Ms(T0);                                  // the main thread's part
                m_TotalGILoadMs += LoadMs;
                m_Renderer.NoteGILoad(LoadMs);
            }
            if (auto G = m_GIBaked.find(pGeom); G != m_GIBaked.end()) m_Stats.m_GIMeshes += G->second;
            return &It->second;
        }

        // The renderer's material of a material instance (invalid: not loaded yet, or it does not resolve - the slot draws with the constant material)
        material_id MaterialOf(std::uint64_t MIInstance) noexcept
        {
            if (MIInstance == 0) return {};
            if (auto It = m_Materials.find(MIInstance); It != m_Materials.end()) return It->second.m_Id;
            std::vector<const void*> From;
            if (!MaterialSources(MIInstance, From)) return {};
            auto* pMI  = static_cast<xmaterial_instance::rt*>(const_cast<void*>(From[0]));
            auto* pMat = static_cast<xmaterial::rt*>(const_cast<void*>(From[1]));
            std::vector<xgpu::texture*> Textures;
            for (auto& E : pMI->getTextures())
            {
                if (&E == pMI->getTextures().data()) { Textures.push_back(nullptr); continue; }    // slot 0: the renderer's shadow atlas
                Textures.push_back(static_cast<xgpu::texture*>(const_cast<void*>(From[1 + Textures.size()])));
            }
            material_id Id;
            // ponytail: the material's default block until material instances carry their own (contract task 8)
            if (Textures.empty() || m_Renderer.CreateMaterial(Id, pMat->getShaderHDR(), Textures, pMat->m_Flags.m_bAlpha, { pMat->m_pDefaultBlock, pMat->m_BlockBytes / 4u })) return {};
            m_Materials[MIInstance] = { Id, std::move(From) };
            return Id;
        }

        // What a material instance's renderer material is made from, as loaded now: the instance, its material, its textures (but slot 0's).
        // False while one of them cannot be had.
        static bool MaterialSources(std::uint64_t MIInstance, std::vector<const void*>& From) noexcept
        {
            using namespace xlioncore::resources;
            From.clear();
            auto* pMI  = static_cast<xmaterial_instance::rt*>(GetResource(material_instance_type_guid_v.m_Value, MIInstance));
            if (!pMI) return false;
            auto* pMat = GetResource(material_type_guid_v.m_Value, pMI->m_MaterialRef.m_Instance.m_Value);
            if (!pMat) return false;
            From = { pMI, pMat };
            for (auto& E : pMI->getTextures())
            {
                if (&E == pMI->getTextures().data()) continue;
                auto* pTexture = GetResource(texture_type_guid_v.m_Value, E.m_TexureRef.m_Instance.m_Value);
                if (!pTexture) return false;
                From.push_back(pTexture);
            }
            return true;
        }

        // A Mesh entity into the scene: its geometry's mesh and per slot the component's material instance, else the geometry's own
        void AddMesh(const mesh& Mesh, const xmath::fmat4& L2W, std::uint64_t Entity) noexcept
        {
            ++m_Stats.m_Meshes;
            const mesh_id* pMesh = MeshOf(Mesh.m_Geom);
            if (!pMesh) { ++m_Stats.m_Waiting; return; }
            auto* pGeom = static_cast<xgeom_static::xgpu::geom*>(xlioncore::resources::GetResource(geom_static_type_guid_v.m_Value, Mesh.m_Geom.m_Instance.m_Value));
#if 0       // The first attempt (the OBB outline shader over a proxy mesh): replaced by the mask and the composite (instance::renderOutline)
            if( Entity == m_Scene.m_SelectedUser && m_OutlineMade.insert(pGeom).second )       // the selection's outline: the geometry's triangles in position and normal (the editor's outline shader draws it)
            {
                gi::surfel_mesh Decoded;
                gi::FromXGeomStatic(Decoded, *pGeom);
                std::vector<float> PosNrm; PosNrm.reserve(Decoded.m_Pos.size() * 6);
                for( std::size_t v = 0; v < Decoded.m_Pos.size(); ++v ) { const auto& P = Decoded.m_Pos[v]; const auto& N = Decoded.m_Nrm[v]; PosNrm.insert(PosNrm.end(), { P.m_X, P.m_Y, P.m_Z, N.m_X, N.m_Y, N.m_Z }); }
                std::vector<std::uint32_t> Idx; Idx.reserve(Decoded.m_Tris.size() * 3);
                for( const auto& T : Decoded.m_Tris ) Idx.insert(Idx.end(), { std::uint32_t(T[0]), std::uint32_t(T[1]), std::uint32_t(T[2]) });
                const xmath::fvec3 Center = (pGeom->m_BBox.m_Min + pGeom->m_BBox.m_Max) * 0.5f, Half = (pGeom->m_BBox.m_Max - pGeom->m_BBox.m_Min) * 0.5f;
                (void)m_Renderer.SetOutlineProxy(*pMesh, PosNrm, Idx, Center, Half);
            }
#endif
            draw D{ .m_Mesh = *pMesh, .m_L2W = L2W, .m_UserID = Entity
                  , .m_Instance = std::uint64_t(m_InstanceIds.Of(Entity)) | (std::uint64_t(Mesh.m_GIRole) << draw_gi_role_shift_v) };
            const auto Defaults = pGeom->getDefaultMaterialInstances();
            D.m_Materials.resize(Defaults.size());
            for (std::size_t i = 0; i < Defaults.size(); ++i)
            {
                const std::uint64_t Override = i < Mesh.m_Materials.size() ? Mesh.m_Materials[i].m_Instance.m_Value : 0;
                D.m_Materials[i] = MaterialOf(Override ? Override : Defaults[i].m_Instance.m_Value);
                m_Stats.m_ConstantSlots += !D.m_Materials[i].isValid();
            }
            ++m_Stats.m_Drawn;
            m_Scene.m_Draws.push_back(std::move(D));
        }

        // A light shines along its entity's -Y
        static xmath::fvec3 DownOf(const xmath::fmat4& L2W) noexcept { auto D = L2W.TransformDirection({ 0, -1, 0 }); D.NormalizeSafe(); return D; }

        void AddSun(const light_sun& L, const xmath::fmat4& L2W) noexcept
        {
            if (m_Stats.m_bSun++) return;               // one sun: the first
            m_Scene.m_Sun = { .m_Direction = DownOf(L2W), .m_Color = L.m_Color, .m_Intensity = L.m_Intensity, .m_bCastShadows = L.m_bCastShadows, .m_ShadowTileSize = std::max(L.m_ShadowTileSize, 0.01f) };
        }

        void AddPoint(const light_point& L, const xmath::fvec3& Pos) noexcept
        {
            ++m_Stats.m_Lights;
            m_Scene.m_LocalLights.push_back({ .m_Type = local_light::type::POINT, .m_Position = Pos, .m_Radius = L.m_Radius, .m_Color = L.m_Color, .m_Intensity = L.m_Intensity, .m_bCastShadows = L.m_bCastShadows });
        }

        void AddSpot(const light_spot& L, const xmath::fmat4& L2W) noexcept
        {
            ++m_Stats.m_Lights;
            m_Scene.m_LocalLights.push_back({ .m_Type = local_light::type::SPOT, .m_Position = L2W * xmath::fvec3(0, 0, 0), .m_Direction = DownOf(L2W), .m_Radius = L.m_Radius
                                            , .m_InnerAngle = L.m_InnerAngle, .m_OuterAngle = std::max(L.m_OuterAngle, L.m_InnerAngle), .m_Color = L.m_Color, .m_Intensity = L.m_Intensity, .m_bCastShadows = L.m_bCastShadows });
        }

        void SetEnvironment(const environment& E) noexcept
        {
            m_Scene.m_Sky     = { .m_Zenith = E.m_SkyZenith, .m_Horizon = E.m_SkyHorizon, .m_Ground = E.m_SkyGround, .m_Intensity = E.m_SkyIntensity };
            m_Scene.m_Ambient = E.m_Ambient;
            m_Scene.m_GI      = { .m_bEnable = E.m_bGI, .m_RayBudget = E.m_GIRayBudget, .m_bPerPixel = E.m_bGIPerPixel, .m_Debug = E.m_GIDebug, .m_RaysPerSurfel = E.m_GIRaysPerSurfel, .m_FilterPasses = E.m_GIFilterPasses, .m_bMoments = E.m_bGIMoments, .m_EmissiveScale = E.m_GIEmissiveScale };
        }

        // A new frame's snapshot: no sun unless a LightSun says so, a dim flat ambient unless an Environment says otherwise
        void Begin() noexcept
        {
            m_Scene.m_Draws.clear();
            m_Scene.m_LocalLights.clear();
            m_Scene.m_Sun     = { .m_Intensity = 0, .m_bCastShadows = false };
            m_Scene.m_Sky     = {};
            m_Scene.m_Ambient = { 0.15f, 0.15f, 0.17f };
            m_Scene.m_GI      = {};
            m_Stats           = {};
            m_InstanceIds.Begin();

            // Something was compiled again (xlioncore::resources::MarkUpdated): a material made from what changed is made again when next asked for
            if (const auto n = xlioncore::resources::UpdateCount(); n != m_UpdateCount)
            {
                m_UpdateCount = n;
                std::vector<const void*> Now;
                for (auto It = m_Materials.begin(); It != m_Materials.end();)
                {
                    if (MaterialSources(It->first, Now) && Now == It->second.m_From) { ++It; continue; }
                    m_Renderer.DestroyMaterial(It->second.m_Id);
                    It = m_Materials.erase(It);
                }
            }
        }

        using overlay_fn = void (*)(xgpu::cmd_buffer& CmdBuffer, void* pUser) noexcept;

        // The snapshot into this world's view (W x H), then the host's overlay over the tone-mapped image with the scene's depth. Records render
        // passes: call while the frame is built, before the UI is rendered; then Blit inside the UI.
        void Render(xgpu::window& Window, xgpu::tools::view& Camera, int W, int H, overlay_fn pOverlay, void* pUser) noexcept
        {
            if (!m_bReady || W <= 1 || H <= 1) return;
            if (m_View.m_Width != W || m_View.m_Height != H)
                if (m_Renderer.CreateView(m_View, { .m_Width = W, .m_Height = H })) return;
            m_Stats.m_Materials = int(m_Materials.size());
            m_Renderer.Render(Window, m_Scene, m_View, Camera, [&](xgpu::cmd_buffer& CmdBuffer) { if (pOverlay) pOverlay(CmdBuffer, pUser); });
        }

        void Blit(xgpu::cmd_buffer& CmdBuffer) noexcept
        {
            if (m_bReady && m_View.m_Width > 0) m_Renderer.Blit(CmdBuffer, m_View);
        }

        ~scene_renderer() noexcept
        {
            if (!m_bReady) return;
            m_Renderer.DestroyView(m_View);
            for (auto& [Key, M] : m_Materials) m_Renderer.DestroyMaterial(M.m_Id);
            for (auto& [Key, Id] : m_Meshes)    m_Renderer.DestroyMesh(Id);
        }
    };
}

#endif
