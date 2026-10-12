#ifndef XLIONRENDER_INSTANCE_H
#define XLIONRENDER_INSTANCE_H
#pragma once

// One renderer. A process may hold several (each editor its own, the level DLL its own); nothing is global.
// Owns the GPU side: pipelines, meshes, per-frame uniforms and GPU timings. Draws a scene (snapshot) into a
// render_view as seen from an xgpu::tools::view camera.

#include "scene/xlionrender_scene.h"
#include "view/xlionrender_render_view.h"
#include "meshes/xlionrender_static_geom.h"
#include "dependencies/xGPU/source/Tools/xgpu_view.h"
#include <span>
#include <vector>
#include <functional>

namespace xlionrender
{
    class instance
    {
    public:

        // GPU time of the last finished frame, in milliseconds (measured with xgpu::timestamps, never stalls)
        struct gpu_times
        {
            double  m_SceneMs       = 0;
            double  m_ToneMapMs     = 0;
        };

                                instance        ( void )                                                        noexcept = default;
                               ~instance        ( void )                                                        noexcept;
                                instance        ( const instance& )                                             = delete;
        instance&               operator=       ( const instance& )                                             = delete;

        [[nodiscard]] xgpu::device::error* Init         ( xgpu::device& Device )                                noexcept;

        [[nodiscard]] xgpu::device::error* CreateMesh   ( mesh_id& Id, std::span<const vertex> Vertices, std::span<const std::uint32_t> Indices ) noexcept;
        // References Geom's GPU buffers (not owned, not copied): they must outlive the mesh id
        [[nodiscard]] xgpu::device::error* CreateStaticGeom( mesh_id& Id, static_geom Geom )                    noexcept;
                      void                 DestroyMesh  ( mesh_id Id )                                          noexcept;

        // A material drawn on static geometry: its HDR fragment shader (xmaterial::rt::getShaderHDR()) and its sampler
        // bindings in shader order - slot 0 is the shadow map (nullptr: white, i.e. unshadowed), then the textures
        // (the same list the static geom editor binds). Neither is copied or owned: they must outlive the id.
        [[nodiscard]] xgpu::device::error* CreateMaterial( material_id& Id, const xgpu::shader& FragmentHDR, std::span<xgpu::texture* const> Textures ) noexcept;
                      void                 DestroyMaterial( material_id Id )                                    noexcept;

        [[nodiscard]] xgpu::device::error* CreateView   ( render_view& View, const render_view::setup& Setup )  noexcept;
                      void                 DestroyView  ( render_view& View )                                   noexcept;

        // Optional editor pass (grid, wire frame, debug lines): recorded after tone mapping into View.m_Output with
        // the scene's depth, so it draws display-space colors and is occluded by the scene. Camera is the same view.
        using overlay_fn = std::function<void(xgpu::cmd_buffer&)>;

        // Records the forward plan (plans/forward) into Window's frame: scene -> View.m_HDR, then tone map ->
        // View.m_Output (alpha 0 where nothing was drawn), then Overlay. Call between Window.BeginRendering() and
        // Window.PageFlip(), outside any other render pass.
                      void                 Render       ( xgpu::window& Window, const scene& Scene, render_view& View, xgpu::tools::view& Camera, const overlay_fn& Overlay = {} ) noexcept;

        // Draws View.m_Output into the current render pass and viewport as is (no color conversion, alpha blended), e.g. from an
        // ImGui render callback. Not ImGui::Image: the xGPU ImGui shader gamma encodes textures again.
                      void                 Blit         ( xgpu::cmd_buffer& CmdBuffer, render_view& View )      noexcept;

        // true when a newer frame's timings arrived since the last call
                      bool                 getGPUTimes  ( gpu_times& Times )                                    noexcept;

    private:

        struct mesh_slot
        {
            xgpu::buffer            m_VertexBuffer;
            xgpu::buffer            m_IndexBuffer;
            int                     m_nIndices      = 0;
            static_geom             m_StaticGeom;                       // used when m_bStatic
            std::uint32_t           m_Generation    = 0;
            bool                    m_bAlive        = false;
            bool                    m_bStatic       = false;
        };

        struct material_slot
        {
            xgpu::pipeline          m_Pipeline;
            xgpu::pipeline_instance m_Instance;
            std::uint32_t           m_Generation    = 0;
            bool                    m_bAlive        = false;
        };

        const mesh_slot*            findMesh        ( mesh_id Id ) const                                        noexcept;
        material_slot*              findMaterial    ( material_id Id )                                          noexcept;

        xgpu::device                m_Device;
        xgpu::vertex_descriptor     m_MeshVD;
        xgpu::vertex_descriptor     m_FullscreenVD;
        xgpu::pipeline              m_ForwardPipeline;
        xgpu::pipeline_instance     m_ForwardInstance;
        xgpu::vertex_descriptor     m_StaticGeomVD;
        xgpu::pipeline              m_StaticGeomPipeline;
        xgpu::pipeline_instance     m_StaticGeomInstance;
        xgpu::buffer                m_StaticGeomUBO;
        xgpu::pipeline              m_ToneMapPipeline;
        xgpu::pipeline              m_BlitPipeline;
        xgpu::texture               m_WhiteTexture;         // stands in for the shadow map until shadows exist
        xgpu::buffer                m_LightUBO;
        xgpu::buffer                m_MaterialUBO;
        xgpu::buffer                m_FullscreenVB;
        xgpu::buffer                m_FullscreenIB;
        xgpu::timestamps            m_Timestamps;
        std::vector<mesh_slot>      m_Meshes;
        std::vector<material_slot>  m_Materials;
        std::vector<std::uint64_t>  m_TimestampNs;
        bool                        m_bReady        = false;
    };
}

#endif
