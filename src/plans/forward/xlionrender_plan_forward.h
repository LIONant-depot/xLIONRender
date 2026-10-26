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
    };

    // The instance buffer's entry (std430 InstanceData)
    struct instance_data
    {
        std::array<xmath::fvec4, 3>     m_L2wRows;      // local -> the instance buffer's origin: the affine matrix's rows
        std::array<xmath::fvec4, 3>     m_Material;     // the constant material: base rgb + metallic, emissive rgb + roughness, AO, F0

        void set(const xmath::fmat4& M, const xmath::fvec3& Origin, const material& Mat) noexcept
        {
            for (int r = 0; r < 3; ++r) m_L2wRows[r] = { M(r, 0), M(r, 1), M(r, 2), M(r, 3) - Origin[r] };
            m_Material[0] = xmath::fvec4(Mat.m_BaseColor, Mat.m_Metallic);
            m_Material[1] = xmath::fvec4(Mat.m_Emissive,  Mat.m_Roughness);
            m_Material[2] = { Mat.m_AO, Mat.m_F0, 0, 0 };
        }
    };
    static_assert(sizeof(instance_data) == 96);

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
    };
    static_assert(offsetof(push_skin_geom, m_MaterialBase) == 16);

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
        xmath::fvec4                            m_ShadowParams;     // x: 1 shadows on, y: level 0 depth half range
        // point and spot lights: the lights and their world cells are in storage buffers (shaders/xlionrender_local_lights.glsl)
        xmath::fvec4                            m_LocalCount;       // x: lights, y: mode (instance::light_mode), z: seed, w: head size
        xmath::fvec4                            m_LocalBase;        // x: first light in the light buffer, y: first uint in the cell buffer, z: level 0 cell size, w: tail samples
        std::array<xmath::fvec4, light_cells::levels_v> m_CellOrigin;       // per cell level: the window's first cell index
        xmath::fvec4                            m_EnvParams;        // x: 1 the environment probe lights (scene::m_Sky), y: its intensity
        xmath::fvec4                            m_ProbeCount;       // x: captured probes (innermost first), y: the first's record in the light buffer,
                                                                    // z: their cell headers' first uint in the cell buffer (-1: no cells, every probe)
        // the sun's virtual shadow pages (shaders/xlionrender_shadow_vsm.glsl; appended, so materials compiled before read the rest as it was)
        xmath::fvec4                            m_VSMParams;        // x: 1 the virtual pages (0: the clipmap), y: level 0 texel, z: levels, w: the page table's first uint in the cell buffer
        xmath::fvec4                            m_VSMParams2;       // x: window (pages), y: page texels, z: pool slots per row, w: 1 every tap through the table (tests)
        std::array<xmath::fvec4, shadows::vsm::max_levels_v> m_VSMLevels;   // per level: xy the window's first page, z page size, w depth centre
    };

    // The sun's virtual pages: the request pass's frame data (shaders/xlionrender_vsm_request_comp.glsl frame_t, std430)
    struct vsm_frame
    {
        xmath::fmat4                            m_C2W;              // the main view's clip -> camera-centred world
        xmath::fvec4                            m_Eye;              // the camera's world position
        xmath::fvec4                            m_LX, m_LY, m_LZ;   // world -> light rows
        xmath::fvec4                            m_P0;               // x: level 0 texel, y: levels, z: window
        std::array<xmath::fvec4, shadows::vsm::max_levels_v> m_Levels;     // per level: xy the window's first page, z page size, w depth centre
    };
    static_assert(sizeof(vsm_frame) == 64 + 5 * 16 + shadows::vsm::max_levels_v * 16);

    // A point or spot light in the light buffer (std430 LocalLightData)
    struct local_light_data
    {
        xmath::fvec4                            m_PosRadius;        // camera-centered position, radius
        xmath::fvec4                            m_Color;            // rgb * intensity, w: cos inner angle
        xmath::fvec4                            m_DirCosOuter;      // spot: direction, w cos outer angle; point: w = -2
        xmath::fvec4                            m_Shadow;           // xy: its shadow map's origin in the atlas, z: its size (0 none), w: spot tan half fov
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

    // xlionrender_shadow_remap_frag.glsl
    struct push_shadow_remap
    {
        std::array<std::int32_t, 4>     m_Tile;             // x, y in the atlas, size, face size in the scratch
        std::array<float, 4>            m_Params;           // near, far, spot tan half fov, 1 point / 0 spot
        std::array<std::int32_t, 12>    m_Faces;            // the faces' origins in the scratch (x, y) x 6
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

    // xlionrender_tonemap_frag.glsl
    struct push_tonemap
    {
        xmath::fvec4    m_Params;               // x = exposure
        xmath::fvec4    m_Background;           // render_view::m_Background
    };
}

#endif
