#ifndef XLIONRENDER_PLAN_FORWARD_H
#define XLIONRENDER_PLAN_FORWARD_H
#pragma once

// Plan A (forward, the reference path). CPU mirrors of the GPU data its shaders read; keep them field-for-field
// in step with shaders/xlionrender_forward_*.glsl and shaders/mb_standard_pbr.frag.

#include <bit>
#include "dependencies/xmath/source/xmath.h"
#include "scene/xlionrender_scene.h"
#include "features/lights/xlionrender_light_cells.h"
#include "features/shadows/xlionrender_shadow_vsm.h"
#include "xlionrender_stage_graph.h"
#include <array>
#include <cstddef>

namespace xlionrender::details
{
    // push constants of xlionrender_forward_vert.glsl (same L2W/W2C convention as the mb_* vertex shaders)
    struct push_forward
    {
        xmath::fmat4    m_L2W;
        xmath::fmat4    m_W2C;
    };

    // "MeshUniforms" (set 2, binding 0) of both cluster vertex paths, and the rigid path's push constant
    // Per view or shadow tile (the per-draw data is the instance buffer's; each UBO entry is padded by xGPU)
    struct ubo_geom_view
    {
        xmath::fmat4                    m_w2C;          // world (camera-centered for views, absolute for shadow tiles) -> clip
        xmath::fvec4                    m_WorldOffset;  // xyz: the instance buffer's origin in that world
        // last frame's (the screen tail's motion: instance::renderTail): its camera-centred world -> clip, the instance buffer's origin
        // in that world (w: 1 there is a last frame; 0: no motion is written)
        xmath::fmat4                    m_PrevW2C       = xmath::fmat4::fromIdentity();
        xmath::fvec4                    m_PrevWorldOffset{ 0, 0, 0, 0 };
    };

    // The instance buffer's entry (std430 InstanceData)
    struct instance_data
    {
        std::array<xmath::fvec4, 3>     m_L2wRows;      // local -> the instance buffer's origin: the affine matrix's rows
        std::array<xmath::fvec4, 3>     m_Material;     // the constant material: base rgb + metallic, emissive rgb + roughness, AO, F0
        std::array<xmath::fvec4, 3>     m_PrevL2wRows;  // last frame's transform (instance_slots: the same as this frame's once it stopped)

        void set(const xmath::fmat4& M, const xmath::fmat4& Prev, const xmath::fvec3& Origin, const material& Mat) noexcept
        {
            for (int r = 0; r < 3; ++r) m_L2wRows[r]     = { M(r, 0), M(r, 1), M(r, 2), M(r, 3) - Origin[r] };
            for (int r = 0; r < 3; ++r) m_PrevL2wRows[r] = { Prev(r, 0), Prev(r, 1), Prev(r, 2), Prev(r, 3) - Origin[r] };
            m_Material[0] = xmath::fvec4(Mat.m_BaseColor, Mat.m_Metallic);
            m_Material[1] = xmath::fvec4(Mat.m_Emissive,  Mat.m_Roughness);
            m_Material[2] = { Mat.m_AO, Mat.m_F0, 0, 0 };
        }
    };
    static_assert(sizeof(instance_data) == 144);

    // A float into an IEEE half (round to nearest; overflow to inf, tiny to 0): the GI depth moments' upload (GLSL unpackHalf2x16)
    inline std::uint16_t Half(float f) noexcept
    {
        const std::uint32_t x = std::bit_cast<std::uint32_t>(f), Sign = (x >> 16) & 0x8000u;
        const int           E = int((x >> 23) & 0xFF) - 127 + 15;
        if (E <= 0)  return std::uint16_t(Sign);
        if (E >= 31) return std::uint16_t(Sign | 0x7C00u);
        const std::uint32_t M = x & 0x7FFFFFu, H = Sign | (std::uint32_t(E) << 10) | (M >> 13);
        return std::uint16_t(H + ((M >> 12) & 1u));                       // carries into the exponent when it rounds up: still correct
    }

    // An IEEE half into a float (read backs of half float targets)
    inline float HalfToFloat(std::uint16_t h) noexcept
    {
        const std::uint32_t e = (h >> 10) & 0x1F, m = h & 0x3FF, s = std::uint32_t(h >> 15) << 31;
        if (e == 0)  return std::bit_cast<float>(s) + (m ? float(m) * (1.0f / 16777216.0f) * (s ? -1.0f : 1.0f) : 0.0f);
        if (e == 31) return std::bit_cast<float>(s | 0x7F800000u | (m << 13));
        return std::bit_cast<float>(s | ((e + 112) << 23) | (m << 13));
    }

    struct push_static_geom
    {
        std::uint32_t   m_ClusterIndex;
        std::uint32_t   m_InstanceIndex;
        std::uint32_t   m_GIOffset      = 0;            // surfel GI (xlionrender_static_geom_gi_vert.glsl): the first surfel in GILight
        std::uint32_t   m_GILevel       = 0xFFFFFFFFu;  // its level (0xFFFFFFFF: none)
        std::uint32_t   m_MaterialBase  = 0;            // the material's first vec4 in MaterialParams (the fragment's "MaterialBase", offset 16)
        std::int32_t    m_GIVertexDelta = 0;            // GI sections: the cluster's vertex in its section's GIMesh is its own plus this
    };
    static_assert(offsetof(push_static_geom, m_MaterialBase) == 16 && offsetof(push_static_geom, m_GIVertexDelta) == 20);
    static_assert(sizeof(push_static_geom) == 24);         // (xgeom_static_mb_clusters.vert DrawRecord, std430)

    // xgpu::cmd_buffer::DrawIndexedIndirect's command (VkDrawIndexedIndirectCommand)
    struct draw_indexed_indirect
    {
        std::uint32_t   m_nIndices;
        std::uint32_t   m_nInstances;
        std::uint32_t   m_iFirstIndex;
        std::int32_t    m_VertexOffset;
        std::uint32_t   m_iFirstInstance;       // (gl_InstanceIndex: the draw's record)
    };
    static_assert(sizeof(draw_indexed_indirect) == 20);

    // xlionrender_skin_geom_vert.glsl (xgeom_skin_mb_clusters.vert) push constant
    struct push_skin_geom
    {
        std::uint32_t   m_ClusterIndex;
        std::uint32_t   m_MaxInfluences;
        std::uint32_t   m_InstanceIndex;
        std::uint32_t   m_Pad           = 0;
        std::uint32_t   m_MaterialBase  = 0;            // as push_static_geom's: one material fragment serves both paths
        std::uint32_t   m_PrevBones     = 0;            // cluster_geom::m_PrevBones: last frame's pose (the motion), 0 none
    };
    static_assert(offsetof(push_skin_geom, m_MaterialBase) == 16 && sizeof(push_skin_geom) <= sizeof(push_static_geom));

    // mb_standard_pbr.frag "lighting_uniforms" (set 2, binding 1)
    struct ubo_lighting
    {
        xmath::fvec4    m_LightColor;           // rgb * intensity
        xmath::fvec4    m_AmbientLightColor;
        xmath::fvec4    m_wSpaceLightPos;       // w == 0: xyz = direction towards the light; w > 0: position + radius
        xmath::fvec4    m_wSpaceEyePos;
        xmath::fvec4    m_LightParams;          // x = area radius, y = temperature K (0 = none), z = intensity mult
        // the directional light's shadow clipmap (shaders/xlionrender_shadow_clipmap.glsl)
        xmath::fmat4                            m_W2Light;          // world -> light space (rotation)
        xmath::fvec4                            m_CameraPos;        // the camera's world position (positions are camera-centered)
        std::array<xmath::fvec4, 4>             m_ShadowLevels;     // per level: xy first tile of the window, z tile size, w depth center
        xmath::fvec4                            m_ShadowParams;     // the local lights' pages: x the pool's atlas x (0: none), y the table's first uint, z slots a row, w the resolution bias
        // point and spot lights: the lights and their world cells are in storage buffers (shaders/xlionrender_local_lights.glsl)
        xmath::fvec4                            m_LocalCount;       // x: lights, y: mode (instance::light_mode), z: seed, w: the tail's resampling candidates (instance::m_LightTailCandidates, > 1: RIS; negative: with their visibility)
        xmath::fvec4                            m_LocalBase;        // x: first light in the light buffer, y: first uint in the cell buffer, z: level 0 cell size, w: tail samples
        std::array<xmath::fvec4, light_cells::levels_v> m_CellOrigin;       // per cell level: the window's first cell index
        xmath::fvec4                            m_EnvParams;        // x: 1 the environment probe lights (scene::m_Sky), y: its intensity, z: 1 the reflection debug view (scene::m_bDebugReflection)
        xmath::fvec4                            m_ProbeCount;       // x: captured probes (innermost first), y: the first's record in the light buffer,
                                                                    // z: their cell headers' first uint in the cell buffer (-1: no cells, every probe)
        // the sun's virtual shadow pages (shaders/xlionrender_shadow_vsm.glsl; appended, so materials compiled before read the rest as it was)
        xmath::fvec4                            m_VSMParams;        // x: 1 the virtual pages (0: the clipmap), y: level 0 texel, z: levels, w: the page table's first uint in the cell buffer
        xmath::fvec4                            m_VSMParams2;       // x: window (pages), y: page texels, z: pool slots per row, w: 1 every tap through the table (tests)
        std::array<xmath::fvec4, shadows::vsm::max_levels_v> m_VSMLevels;   // per level: xy the window's first page, z page size, w depth centre
        xmath::fvec4                            m_VSMParams3;       // x: the resolution bias (levels added to the footprint's)
        // The previous epoch (P1e, shadows::vsm::HasPrevious): its world -> light, its levels, x: there is one, y: the current epoch's weight
        // (the cross-fade), z: its level 0 texel, w: its table's first uint
        xmath::fmat4                            m_VSMPrevW2Light;
        std::array<xmath::fvec4, shadows::vsm::max_levels_v> m_VSMPrevLevels;
        xmath::fvec4                            m_VSMPrevParams;
        // the screen tail (lighting_p3_design.md "The decision"; appended): x 1 the material reads the denoised tail (mode 3), y, z the half-size
        // targets' width and height, w its first texel's record in the tail buffer (binding 14); the tail pass's clip -> camera-centred world
        xmath::fvec4                            m_TailParams;
        xmath::fmat4                            m_TailC2W;
        xmath::fmat4                            m_TailW2C;          // the tail pass's camera-centred world -> clip (its screen trace)
        xmath::fvec4                            m_TailTrace;        // the screen trace: x its length (world units; 0: none), y steps, z the depth's thickness (fraction of the view depth)
        xmath::fvec4                            m_TailBvh;          // P4, the traced visibility through the GI's BVH: x 1 on, y its TLAS root, z its primitives' offset (ints as float bits)
    };

    // The sun's virtual pages: the request pass's frame data (shaders/xlionrender_vsm_request_comp.glsl frame_t, std430)
    struct vsm_frame
    {
        xmath::fmat4                            m_C2W;              // the main view's clip -> camera-centred world
        xmath::fvec4                            m_Eye;              // the camera's world position
        xmath::fvec4                            m_LX, m_LY, m_LZ;   // world -> light rows
        xmath::fvec4                            m_P0;               // x: level 0 texel, y: levels, z: window, w: the resolution bias
        std::array<xmath::fvec4, shadows::vsm::max_levels_v> m_Levels;     // per level: xy the window's first page, z page size, w depth centre
    };
    static_assert(sizeof(vsm_frame) == 64 + 5 * 16 + shadows::vsm::max_levels_v * 16);

    // The local lights' request pass's frame (shaders/xlionrender_local_request_comp.glsl frame_t)
    struct local_request_frame
    {
        struct light { xmath::fvec4 m_PosRadius, m_DirTan, m_Info; };
        xmath::fmat4                            m_C2W;              // the main view's clip -> camera-centred world
        xmath::fvec4                            m_Eye;
        xmath::fvec4                            m_P0;               // x: lights, y: the resolution bias, z: the table's first uint
        std::array<light, max_local_lights_v>   m_Lights;
    };

    // A point or spot light in the light buffer (std430 LocalLightData)
    struct local_light_data
    {
        xmath::fvec4                            m_PosRadius;        // camera-centered position, radius
        xmath::fvec4                            m_Color;            // rgb * intensity, w: cos inner angle
        xmath::fvec4                            m_DirCosOuter;      // spot: direction, w cos outer angle; point: w = -2
        xmath::fvec4                            m_Shadow;           // x: its first face's entry in the local pages' table, z: 1 shadowed, w: spot tan half fov
    };

    // A reflection probe in the light buffer (shaders/xlionrender_environment.glsl ProbeRecord: the same 64 bytes as a light)
    struct probe_data
    {
        xmath::fvec4                            m_PosBlend;         // camera-centered center, w: blend distance
        xmath::fvec4                            m_BoxMin, m_BoxMax; // camera-centered
        xmath::fvec4                            m_Slot;             // xy: its slot's origin in the probe atlas
    };
    static_assert(sizeof(probe_data) == sizeof(local_light_data));

    // xlionrender_forward_standard_frag.glsl "material_uniforms" (set 2, binding 2)
    struct ubo_material
    {
        xmath::fvec4    m_BaseColor;
        xmath::fvec4    m_Params;               // x = metallic, y = roughness, z = AO, w = F0
        xmath::fvec4    m_Emissive;
    };

    // xlionrender_env_sky_frag.glsl, xlionrender_env_prefilter_frag.glsl, xlionrender_env_dfg_frag.glsl
    struct push_env_sky
    {
        xmath::fvec4    m_SunDir, m_SunColor, m_Zenith, m_Horizon, m_Ground;       // m_Ground.w: the source's size
    };
    struct push_env_prefilter
    {
        xmath::fvec4    m_Rect;             // the level's origin and size in the atlas
        xmath::fvec4    m_Params;           // roughness, the source's size, samples
    };
    struct push_env_downsample
    {
        xmath::fvec4    m_Params;           // this level's size, the parent's size
    };
    struct push_env_cube_to_oct
    {
        xmath::fvec4    m_Params;           // the source's size
    };
    struct push_env_dfg
    {
        xmath::fvec4    m_Params;           // the table's size, samples
    };

    // xlionrender_outline_composite_frag.glsl
    struct push_outline_composite
    {
        xmath::fvec4    m_Color;        // rgb the outline's color, a its opacity
        xmath::fvec4    m_Params;       // x radius (px), y the hidden part's opacity, z the hidden fill's opacity
    };

    // xlionrender_outline_obb_vert.glsl (the editor's selection outline: the first attempt, not used)
    struct push_outline_obb
    {
        xmath::fmat4    m_L2C;
        xmath::fvec4    m_ViewportAndRadius;    // xy viewport px, z radius px, w max fractional expand
        xmath::fvec4    m_Color;
        xmath::fvec4    m_Bounds;               // xyz local box center, w cage growth
        xmath::fvec4    m_Axis;                 // xyz local box half extents, w near-parallel rejection angle (degrees)
    };

    // xlionrender_sky_background_frag.glsl
    struct push_sky_background
    {
        xmath::fmat4    m_InvW2C;           // clip -> the camera-centered world
        xmath::fvec4    m_Params;           // x: the intensity, yz: the size of the view
    };

    // xlionrender_gi_gather_common.glsl gather_push
    struct push_gi_gather
    {
        std::uint32_t   m_TableBase;        // this frame's region of the table (uints)
        std::uint32_t   m_Buffer;           // the pass's GI buffer (its depth: (m_Buffer + 1) / 65536)
        std::uint32_t   m_Div;              // the read's resolution divisor
        std::uint32_t   m_AO = 0;           // 1: the ambient occlusion is on (render_view::m_AO; the pixels' AO entries)
    };

    // The light tiles as a stage description (render graph R1, xlionrender_stage_graph.h): each 8 x 8 tile's local lights from the depth
    // (xlionrender_light_tiles_comp.glsl; push: the clip-to-world matrix, the view's size, the frame's local lights)
    inline const stage::desc& LightTilesStage( void ) noexcept
    {
        static const stage::desc D{ { { .m_Name = "light tiles", .m_Kernel = "light_tiles"
            , .m_Buffers  = { { "Lights: light ring", rg::lifetime::RETAINED }, { "Lights: the screen tiles' lists", rg::lifetime::FRAME, rg::usage::STORAGE, true, rg::policy::FULL_OVERWRITE } }
            , .m_Textures = { { "Render target: depth", rg::lifetime::FRAME, rg::usage::SAMPLED } }
            , .m_Push     = { { "frame.c2w" }, { "view.width" }, { "view.height" }, { "lights.local_base" }, { "lights.local_count" } } } } };
        return D;
    }

    // The ambient occlusion as a stage (xlionrender_ao_comp.glsl, on the GI read's grid): the raw AO from the G-buffer (and, XLION_SSGI,
    // last frame's lit image), then its 4 x 4 filter into the AO and an "AO only" entry into each pixel of the GI screen buffer (its
    // header: last frame's world to clip). The modes are frame values (XLION_SSGI: 4 and 5, else 0 and 1); last frame's AO is bound,
    // not read (no history: each frame good on its own)
    inline const stage::desc& AOStage( void ) noexcept
    {
        const auto Pass = []( const char* pName, bool bFilter )
        {
            const auto Tex = [&]( const char* p ) { return stage::binding{ p, rg::lifetime::FRAME, rg::usage::SAMPLED }; };
            return stage::pass{ .m_Name = pName, .m_Kernel = "ao"
                , .m_Buffers  = { { "GI: screen buffer*", rg::lifetime::FRAME, rg::usage::STORAGE, bFilter, bFilter ? rg::policy::PARTIAL : rg::policy::NONE, bFilter ? "" : "never" }
                                , { "GI: gather table" }
                                , { "GI: ambient occlusion (last frame)", rg::lifetime::HISTORY, rg::usage::STORAGE, false, rg::policy::NONE, "never" }
                                , { "GI: ambient occlusion", rg::lifetime::FRAME, rg::usage::STORAGE, bFilter, bFilter ? rg::policy::FULL_OVERWRITE : rg::policy::NONE, bFilter ? "" : "never" }
                                , { "GI: ambient occlusion (raw)", rg::lifetime::FRAME, rg::usage::STORAGE, !bFilter, bFilter ? rg::policy::NONE : rg::policy::FULL_OVERWRITE } }
                , .m_Textures = { Tex("Render target: depth"), Tex("Render target: prepass normals"), Tex("Render target: prepass ids")
                                , { "Render target: last frame's HDR", rg::lifetime::HISTORY, rg::usage::SAMPLED, false, rg::policy::NONE, bFilter ? "never" : "ao.ssgi" } }
                , .m_Push     = { { "gi.screen_width" }, { "gi.screen_height" }, { "gi.divisor" }, { bFilter ? "ao.mode_filter" : "ao.mode_raw" }, { "ao.radius" }, { "ao.thickness" }, { "ao.pixels_per_unit" }
                                , { "gi.table_base" }, { "", 0u }, { "", 0x3EA8F5C3u } }      // (0x3EA8F5C3: 0.33f, the history's blend - unused)
                , .m_Size     = "gi.screen_size" };
        };
        static const stage::desc D{ { Pass("AO (raw)", false), Pass("AO (filtered)", true) } };
        return D;
    }

    // The GI read as a stage (render passes: xlionrender_gi_gather): at the GI read's size, the classification (each pixel's GI buffer as
    // its depth), then a full-screen draw a GI buffer tested EQUAL - the gather buffers' list - into the GI screen buffer the scene pass
    // upsamples. The pipeline instances hold their textures (declared here, not bound). Optional buffers (the voxel clipmap, the path
    // traced reference, the published voxel light, the heat) bound as a dummy when off, declared when on
    inline const stage::desc& GIReadStage( void ) noexcept
    {
        const auto Pass = []( const char* pName, bool bGather )
        {
            const auto Rd = [&]( const char* p, const char* pIf = "" ) { return stage::binding{ p, rg::lifetime::RETAINED, rg::usage::STORAGE, false, rg::policy::NONE, pIf }; };
            const auto Tx = [&]( const char* p ) { return stage::binding{ p, rg::lifetime::FRAME, rg::usage::SAMPLED }; };
            return stage::pass{ .m_Name = pName, .m_Kernel = bGather ? "gi_gather" : "gi_classify"
                , .m_Buffers  = { Rd("Frame: instance data"), bGather ? Rd("GI: gather buffers") : Rd("GI: dummy", "never"), Rd("GI: light ring"), Rd("GI: moment ring")
                                , { "GI: screen buffer*", rg::lifetime::FRAME, rg::usage::STORAGE, true, rg::policy::PARTIAL }
                                , Rd("GI: screen buffer previous"), { "GI: gather table" }, { "GI: ambient occlusion", rg::lifetime::FRAME, rg::usage::STORAGE, false, rg::policy::NONE, "gi.ao" }
                                , Rd("GI: voxel clipmap", "gi.clip"), Rd("GI: reference samples", "gi.reference"), { "GI: reference sums", rg::lifetime::RETAINED, rg::usage::STORAGE, true, rg::policy::PARTIAL, "gi.reference" }
                                , Rd("GI: the published voxel light", "gi.voxpub"), { "Heat: the gather's clock a pixel", rg::lifetime::FRAME, rg::usage::STORAGE, true, rg::policy::PARTIAL, "gi.heat" }
                                , { "GI: screen history", rg::lifetime::RETAINED, rg::usage::STORAGE, true, rg::policy::PARTIAL } }
                , .m_Textures = { Tx("Probes: environment DFG"), Tx("Probes: environment atlas"), Tx("Probes: probe atlas"), Tx("Render target: contact shadow"), Tx("Render target: depth")
                                , Tx("Render target: prepass ids"), Tx("Render target: prepass normals"), Tx("Render target: sun shadow mask"), Tx("Shadows: white dummy") }
                , .m_Images   = { { "GI: screen colour", rg::lifetime::FRAME, rg::usage::COLOR_ATTACHMENT, true, bGather ? rg::policy::PRESERVE : rg::policy::DISCARD }
                                , { "GI: screen depth", rg::lifetime::FRAME, rg::usage::DEPTH_ATTACHMENT, true, bGather ? rg::policy::PRESERVE : rg::policy::DISCARD } }
                , .m_Push     = { { "gi.table_base" }, bGather ? stage::word{ "repeat.index" } : stage::word{ "", 0u }, { "gi.divisor" }, { "gi.ao_mode" } }
                , .m_RenderPass = "GI: screen pass", .m_Repeat = bGather ? "gi.gather_buffers" : "", .m_RepeatSlot = bGather ? 1 : -1 };
        };
        static const stage::desc D{ { Pass("GI read: classify", false), Pass("GI read: gather", true) } };
        return D;
    }

    // The GI's update as a stage: one recorded step (its dispatches: the scatter, the migration, the relight, the links, the probes - the
    // GI's code, gi::gpu_trace), declared: the frame's host-visible GI data read, the world's GI data kept and updated
    inline const stage::desc& GIUpdateStage( void ) noexcept
    {
        static const stage::desc D{ { { .m_Name = "GI update"
            , .m_Buffers = { { "GI: host-visible data" }, { "GI: *", rg::lifetime::RETAINED, rg::usage::STORAGE, true, rg::policy::PARTIAL } }
            , .m_Recorder = "gi.update" } } };
        return D;
    }

    // The GI's voxel rays as a stage: one recorded step (gi::gpu_trace VoxStep: rays from the GI read's pixels into the voxel store), declared:
    // the rays' BVH, the world's GI data, last frame's lit image read; the voxel store kept and updated
    inline const stage::desc& VoxRaysStage( void ) noexcept
    {
        static const stage::desc D{ { { .m_Name = "GI voxel rays"
            , .m_Buffers = { { "Rays: *", rg::lifetime::RETAINED }, { "GI: world data", rg::lifetime::RETAINED }, { "GI: device-local data", rg::lifetime::RETAINED }
                           , { "GI: voxel store", rg::lifetime::RETAINED, rg::usage::STORAGE, true, rg::policy::PARTIAL } }
            , .m_Textures = { { "Render target: last frame's HDR", rg::lifetime::RETAINED } }
            , .m_Recorder = "gi.voxel_rays" } } };
        return D;
    }

    // The shadow pages as a stage: two recorded passes in the shadow atlas's render pass - the sun's virtual pages, the local lights'
    // pages (their page systems choose the pages and the casters: code, xlionrender_plan_forward.cpp renderShadows). Declared: what they
    // read (the instances, the geometry, the shadow data) and the atlas they draw into (the requested pages only: partial)
    inline const stage::desc& ShadowPagesStage( void ) noexcept
    {
        const auto Pass = []( const char* pName, const char* pRecorder )
        {
            return stage::pass{ .m_Name = pName
                , .m_Buffers  = { { "Frame: instance data", rg::lifetime::RETAINED }, { "Geometry: *", rg::lifetime::RETAINED }, { "Shadows: *", rg::lifetime::RETAINED } }
                , .m_Images   = { { "Shadows: shadow atlas", rg::lifetime::RETAINED, rg::usage::DEPTH_ATTACHMENT, true, rg::policy::PARTIAL } }
                , .m_RenderPass = "Shadows: shadow atlas", .m_Recorder = pRecorder };
        };
        static const stage::desc D{ { Pass("shadow pages: sun", "shadows.sun_pages"), Pass("shadow pages: local lights", "shadows.local_pages") } };
        return D;
    }

    // The lighting tile pass as a stage (xlionrender_tile_pixel.glsl, a material graph's tile variant): one full-screen draw a material
    // graph (the list of draw items the code fills: each graph's pipeline instance - its textures: the G-buffer, the tile attributes - and
    // its materials' texture table) onto the HDR the prepass started; the push: clip to world and the graph's index
    inline const stage::desc& TileStage( void ) noexcept
    {
        const auto Rd = []( const char* p, const char* pIf = "" ) { return stage::binding{ p, rg::lifetime::RETAINED, rg::usage::STORAGE, false, rg::policy::NONE, pIf }; };
        const auto Tx = []( const char* p ) { return stage::binding{ p, rg::lifetime::FRAME, rg::usage::SAMPLED }; };
        static const stage::desc D{ {
              // (A5.2, XLION_TILE_CLASSIFY: each pixel's graph as the depth the graphs' draws test EQUAL - no item when off)
              { .m_Name = "tile pass: classification", .m_Kernel = "tile_classify"
            , .m_Buffers  = { { "Materials: material parameters", rg::lifetime::RETAINED } }
            , .m_Textures = { { "Render target: prepass ids", rg::lifetime::FRAME, rg::usage::SAMPLED } }
            , .m_Images   = { { "Render target: tile graphs", rg::lifetime::FRAME, rg::usage::DEPTH_ATTACHMENT, true, rg::policy::DISCARD } }
            , .m_Push     = { { "", 0u }, { "", 0u }, { "", 0u }, { "", 0u } }
            , .m_RenderPass = "Render target: HDR colour (the tile pass)", .m_Items = "tile.classify" }
            , { .m_Name = "tile pass (lighting)", .m_Kernel = "tile"
            , .m_Buffers  = { Rd("GI: dummy", "never"), Rd("Frame: instance data"), Rd("Lights: light ring"), Rd("Lights: cell ring"), Rd("GI: screen buffer*"), Rd("GI: light ring"), Rd("GI: moment ring")
                            , Rd("Materials: material parameters"), Rd("Materials: entity parameters"), Rd("Render target: tail records")
                            , { "Tile pass: the light funnel's counters", rg::lifetime::RETAINED, rg::usage::STORAGE, true, rg::policy::PARTIAL, "tile.funnel" }
                            , { "Lights: the screen tiles' lists", rg::lifetime::FRAME, rg::usage::STORAGE, false, rg::policy::NONE, "lights.tiles" } }
            , .m_Textures = { Tx("Render target: depth"), Tx("Render target: prepass normals"), Tx("Render target: prepass ids"), Tx("Render target: tile attributes")
                            , Tx("Probes: environment DFG"), Tx("Probes: environment atlas"), Tx("Probes: probe atlas"), Tx("Render target: contact shadow"), Tx("Render target: sun shadow mask")
                            , Tx("Shadows: shadow atlas"), Tx("Shadows: white dummy") }
            , .m_Images   = { { "Render target: HDR colour", rg::lifetime::FRAME, rg::usage::COLOR_ATTACHMENT, true, rg::policy::PRESERVE }
                            , { "Render target: tile graphs", rg::lifetime::FRAME, rg::usage::DEPTH_ATTACHMENT, false, rg::policy::NONE, "tile.classify" } }
            , .m_Push     = { { "frame.c2w" }, { "repeat.index" }, { "", 0u }, { "", 0u }, { "", 0u } }
            , .m_RenderPass = "Render target: HDR colour (the tile pass)", .m_Items = "tile.graphs", .m_UBOs = { "Geometry: view block", "Lights: lighting block" } } } };
        return D;
    }

    // The sun's shadow mask as a stage (xlionrender_sun_mask_frag.glsl): a full-screen draw over the G-buffer prepass into the mask's
    // target - its pipeline instance holds the shadow atlas, the prepass and last frame's mask (declared, not bound) - then the copy the
    // materials read (a recorded step: next frame's pass reads it as last frame's). The lighting block at index 1 (none at 0)
    inline const stage::desc& SunMaskStage( void ) noexcept
    {
        const auto Rd = []( const char* p ) { return stage::binding{ p, rg::lifetime::RETAINED, rg::usage::SAMPLED }; };
        static const stage::desc D{ {
              { .m_Name = "sun shadow mask", .m_Kernel = "sun_mask"
            , .m_Buffers  = { Rd("Lights: light ring"), Rd("Lights: cell ring"), Rd("Shadows: VSM page depth bounds") }
            , .m_Textures = { Rd("Probes: environment DFG"), Rd("Probes: environment atlas"), Rd("Probes: probe atlas"), Rd("Render target: contact shadow"), Rd("Render target: depth")
                            , Rd("Render target: prepass ids"), Rd("Render target: prepass normals"), Rd("Render target: sun shadow mask"), Rd("Shadows: shadow atlas"), Rd("Shadows: white dummy") }
            , .m_Images   = { { "Render target: sun shadow mask (pass)", rg::lifetime::FRAME, rg::usage::COLOR_ATTACHMENT, true, rg::policy::FULL_OVERWRITE } }
            , .m_Push     = { { "sun_mask.reproject" }, { "sun_mask.delta" }, { "sun_mask.params" } }
            , .m_RenderPass = "Render target: sun shadow mask (pass)", .m_UBOs = { "", "Lights: lighting block" } }
            , { .m_Name = "sun shadow mask copy"
            , .m_Textures = { { "Render target: sun shadow mask (pass)", rg::lifetime::FRAME, rg::usage::TRANSFER_SRC } }
            , .m_Images   = { { "Render target: sun shadow mask", rg::lifetime::HISTORY, rg::usage::TRANSFER_DST, true, rg::policy::FULL_OVERWRITE } }
            , .m_Recorder = "sun_mask.copy" } } };
        return D;
    }

    // The shadow request passes as stages (xlionrender_vsm_request_comp.glsl, xlionrender_local_request_comp.glsl): over the main view's
    // depth, the pages each pixel needs into the requests (cleared before, read back after by the code: the page systems are CPU side)
    inline stage::pass RequestPass( const char* pName, const char* pKernel, const char* pFrames, const char* pReq, std::vector<stage::word> Push )
    {
        return stage::pass{ .m_Name = pName, .m_Kernel = pKernel
            , .m_Buffers  = { { pFrames }, { pReq, rg::lifetime::FRAME, rg::usage::STORAGE, true, rg::policy::PARTIAL }, { "Lights: cell ring" } }
            , .m_Textures = { { "Render target: depth", rg::lifetime::FRAME, rg::usage::STORAGE } }      // (declared as the code always did: a storage read)
            , .m_Push     = std::move(Push) };
    }
    inline const stage::desc& VSMRequestStage( void ) noexcept
    {
        static const stage::desc D{ { RequestPass("Shadows: VSM requests made", "vsm_request", "Shadows: VSM frames", "Shadows: VSM requests"
            , { { "shadow.slot" }, { "", 0u }, { "vsm.table_base" }, { "view.width" }, { "view.height" }, { "vsm.bit_uints" }, { "vsm.soft_levels" }, { "vsm.level_band" } }) } };
        return D;
    }
    inline const stage::desc& LocalRequestStage( void ) noexcept
    {
        static const stage::desc D{ { RequestPass("Shadows: local light requests made", "local_request", "Shadows: local light frames", "Shadows: local light requests"
            , { { "shadow.slot" }, { "", 0u }, { "view.width" }, { "view.height" }, { "local.bit_uints" }, { "", 0u }, { "", 0u }, { "", 0u } }) } };
        return D;
    }

    // The reflections' screen trace as a stage (xlionrender_refl_ssr_comp.glsl): last frame's lit image at the ray's hit in this frame's
    // thin G-buffer, into the tail's records (last frame's image declared on the frames there is one: else the view's own HDR, cleared)
    inline const stage::desc& ReflScreenStage( void ) noexcept
    {
        static const stage::desc D{ { { .m_Name = "reflections: screen trace", .m_Kernel = "refl_ssr"
            , .m_Buffers  = { { "Frame: reflection frames" }, { "Render target: tail records", rg::lifetime::FRAME, rg::usage::STORAGE, true, rg::policy::FULL_OVERWRITE } }
            , .m_Textures = { { "Render target: tail colour", rg::lifetime::FRAME, rg::usage::SAMPLED }, { "Render target: tail depth", rg::lifetime::FRAME, rg::usage::SAMPLED }
                            , { "Render target: last frame's HDR", rg::lifetime::HISTORY, rg::usage::SAMPLED, false, rg::policy::NONE, "view.hdr_prev" } }
            , .m_Push     = { { "refl.slot" }, { "", 0u }, { "refl.ssr_frame" }, { "refl.ssr_steps" }, { "refl.ssr_max_dist" }, { "refl.ssr_thickness" }, { "refl.ssr_max_rough" }, { "", 0u } }
            , .m_Size     = "refl.size" } } };
        return D;
    }

    // The reflections' reconstruction as a stage (xlionrender_refl_denoise_comp.glsl): temporal by the reflection's virtual motion, then
    // spatial, on the tail's records; the history read by the temporal pass on the frames it is valid, written by the spatial one
    inline const stage::desc& ReflReconStage( void ) noexcept
    {
        const auto Pass = []( const char* pName, std::uint32_t Pass )
        {
            const bool b1 = Pass == 1;
            return stage::pass{ .m_Name = pName, .m_Kernel = "refl_denoise"
                , .m_Buffers  = { { "Frame: reflection frames" }
                                , { "Render target: tail records", rg::lifetime::FRAME, rg::usage::STORAGE, true, rg::policy::PRESERVE }
                                , { "Reflections: reconstruction temporal", rg::lifetime::FRAME, rg::usage::STORAGE, !b1, b1 ? rg::policy::NONE : rg::policy::FULL_OVERWRITE }
                                , { "Reflections: history", rg::lifetime::HISTORY, rg::usage::STORAGE, b1, b1 ? rg::policy::FULL_OVERWRITE : rg::policy::NONE, b1 ? "" : "refl.history_valid" } }
                , .m_Textures = { { "Render target: tail colour", rg::lifetime::FRAME, rg::usage::SAMPLED }, { "Render target: tail depth", rg::lifetime::FRAME, rg::usage::SAMPLED }
                                , { "Render target: tail motion", rg::lifetime::FRAME, rg::usage::SAMPLED } }
                , .m_Push     = { { "", Pass }, { "refl.slot" }, { "refl.disk_fresh" }, { "", 0u } }
                , .m_Size     = "refl.size" };
        };
        static const stage::desc D{ { Pass("reflections: temporal", 0), Pass("reflections: spatial", 1) } };
        return D;
    }

    // xlionrender_ao_comp.glsl push_t
    struct push_ao
    {
        std::uint32_t   m_W, m_H, m_Div, m_Mode;      // (Mode 0: the raw AO, 1: its 4 x 4 filter and the history)
        float           m_Radius, m_Thickness, m_PixelsPerUnit;
        std::uint32_t   m_TableBase, m_bPrev;
        float           m_Blend;
    };

    // xlionrender_tonemap_frag.glsl
    struct push_tonemap
    {
        xmath::fvec4    m_Params;               // x = exposure, y = the bloom's share, z = 1: auto exposure
        xmath::fvec4    m_Background;           // render_view::m_Background
    };
}

#endif
