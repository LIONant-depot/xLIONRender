#ifndef XLIONRENDER_PLAN_FORWARD_H
#define XLIONRENDER_PLAN_FORWARD_H
#pragma once

// Plan A (forward, the reference path). CPU mirrors of the GPU data its shaders read; keep them field-for-field
// in step with shaders/xlionrender_forward_*.glsl and shaders/mb_standard_pbr.frag.

#include "dependencies/xmath/source/xmath.h"
#include "scene/xlionrender_scene.h"
#include "features/lights/xlionrender_light_cells.h"
#include <array>

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

    struct push_static_geom
    {
        std::uint32_t   m_ClusterIndex;
        std::uint32_t   m_InstanceIndex;
    };

    // xlionrender_skin_geom_vert.glsl (xgeom_skin_mb_clusters.vert) push constant
    struct push_skin_geom
    {
        std::uint32_t   m_ClusterIndex;
        std::uint32_t   m_MaxInfluences;
        std::uint32_t   m_InstanceIndex;
    };

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
    };

    // A point or spot light in the light buffer (std430 LocalLightData)
    struct local_light_data
    {
        xmath::fvec4                            m_PosRadius;        // camera-centered position, radius
        xmath::fvec4                            m_Color;            // rgb * intensity, w: cos inner angle
        xmath::fvec4                            m_DirCosOuter;      // spot: direction, w cos outer angle; point: w = -2
        xmath::fvec4                            m_Shadow;           // xy: its shadow map's origin in the atlas, z: its size (0 none), w: spot tan half fov
    };

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
    struct push_env_dfg
    {
        xmath::fvec4    m_Params;           // the table's size, samples
    };

    // xlionrender_tonemap_frag.glsl
    struct push_tonemap
    {
        xmath::fvec4    m_Params;               // x = exposure
        xmath::fvec4    m_Background;           // render_view::m_Background
    };
}

#endif
