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
        std::unordered_map<std::uint64_t, material_id>  m_Materials;    // per material instance (instance guid)

        struct stats
        {
            int m_Meshes = 0, m_Drawn = 0, m_Waiting = 0, m_Lights = 0, m_bSun = 0, m_Materials = 0, m_ConstantSlots = 0, m_GIMeshes = 0;
        };
        double                                          m_TotalGILoadMs = 0;           // every geometry's surfels made ready so far (DescribeScene)
        std::unordered_map<const void*, int>            m_GIBaked;      // per geometry: its surfels given to the renderer (1), or it has none (0)
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
            // GI on: the surfels the GeomStatic compiler baked, with the rays' structures built from the geometry's CPU copy, the first time
            if (m_Scene.m_GI.m_bEnable && !m_GIBaked.contains(pGeom))
            {
                const auto T0 = std::chrono::steady_clock::now();
                gi::surfel_mesh Surfels;
                gi::FromXGeomStatic(Surfels, *pGeom);
                const bool bGI = gi::SurfelsFromXGeomStatic(Surfels, *pGeom) && Surfels.m_Levels[0].size() < 0xFFFF && !m_Renderer.SetMeshGI(It->second, std::move(Surfels));
                m_GIBaked[pGeom] = bGI;
                m_TotalGILoadMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - T0).count();
            }
            if (auto G = m_GIBaked.find(pGeom); G != m_GIBaked.end()) m_Stats.m_GIMeshes += G->second;
            return &It->second;
        }

        // The renderer's material of a material instance (invalid: not loaded yet, or it does not resolve - the slot draws with the constant material)
        material_id MaterialOf(std::uint64_t MIInstance) noexcept
        {
            if (MIInstance == 0) return {};
            if (auto It = m_Materials.find(MIInstance); It != m_Materials.end()) return It->second;
            using namespace xlioncore::resources;
            auto* pMI  = static_cast<xmaterial_instance::rt*>(GetResource(material_instance_type_guid_v.m_Value, MIInstance));
            if (!pMI) return {};
            auto* pMat = static_cast<xmaterial::rt*>(GetResource(material_type_guid_v.m_Value, pMI->m_MaterialRef.m_Instance.m_Value));
            if (!pMat) return {};
            std::vector<xgpu::texture*> Textures;
            for (auto& E : pMI->getTextures())
            {
                if (&E == pMI->getTextures().data()) { Textures.push_back(nullptr); continue; }    // slot 0: the renderer's shadow atlas
                auto* pTexture = static_cast<xgpu::texture*>(GetResource(texture_type_guid_v.m_Value, E.m_TexureRef.m_Instance.m_Value));
                if (!pTexture) return {};
                Textures.push_back(pTexture);
            }
            material_id Id;
            if (Textures.empty() || m_Renderer.CreateMaterial(Id, pMat->getShaderHDR(), Textures)) return {};
            m_Materials[MIInstance] = Id;
            return Id;
        }

        // A Mesh entity into the scene: its geometry's mesh and per slot the component's material instance, else the geometry's own
        void AddMesh(const mesh& Mesh, const xmath::fmat4& L2W, std::uint64_t Entity) noexcept
        {
            ++m_Stats.m_Meshes;
            const mesh_id* pMesh = MeshOf(Mesh.m_Geom);
            if (!pMesh) { ++m_Stats.m_Waiting; return; }
            auto* pGeom = static_cast<xgeom_static::xgpu::geom*>(xlioncore::resources::GetResource(geom_static_type_guid_v.m_Value, Mesh.m_Geom.m_Instance.m_Value));
            draw D{ .m_Mesh = *pMesh, .m_L2W = L2W, .m_UserID = Entity, .m_Instance = std::uint64_t(std::uint32_t(Entity ^ (Entity >> 32)) | 1u) };
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
            m_Scene.m_GI      = { .m_bEnable = E.m_bGI, .m_RayBudget = E.m_GIRayBudget, .m_bPerPixel = E.m_bGIPerPixel, .m_Debug = E.m_GIDebug };
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
            for (auto& [Key, Id] : m_Materials) m_Renderer.DestroyMaterial(Id);
            for (auto& [Key, Id] : m_Meshes)    m_Renderer.DestroyMesh(Id);
        }
    };
}

#endif
