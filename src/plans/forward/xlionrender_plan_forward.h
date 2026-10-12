#ifndef XLIONRENDER_PLAN_FORWARD_H
#define XLIONRENDER_PLAN_FORWARD_H
#pragma once

// Plan A (forward, the reference path). CPU mirrors of the GPU data its shaders read; keep them field-for-field
// in step with shaders/xlionrender_forward_*.glsl and shaders/mb_standard_pbr.frag.

#include "dependencies/xmath/source/xmath.h"

namespace xlionrender::details
{
    // push constants of xlionrender_forward_vert.glsl (same L2W/W2C convention as the mb_* vertex shaders)
    struct push_forward
    {
        xmath::fmat4    m_L2W;
        xmath::fmat4    m_W2C;
    };

    // "MeshUniforms" (set 2, binding 0) of both cluster vertex paths, and the rigid path's push constant
    struct alignas(256) ubo_static_geom_mesh
    {
        xmath::fmat4    m_L2w;                  // local -> camera-centered world
        xmath::fmat4    m_w2C;                  // camera-centered world -> clip
        xmath::fmat4    m_L2CShadowT;           // local -> shadow texture space
    };

    struct push_static_geom
    {
        std::uint32_t   m_ClusterIndex;
    };

    // xlionrender_skin_geom_vert.glsl (xgeom_skin_mb_clusters.vert) push constant
    struct push_skin_geom
    {
        std::uint32_t   m_ClusterIndex;
        std::uint32_t   m_MaxInfluences;
    };

    // mb_standard_pbr.frag "lighting_uniforms" (set 2, binding 1)
    struct alignas(256) ubo_lighting
    {
        xmath::fvec4    m_LightColor;           // rgb * intensity
        xmath::fvec4    m_AmbientLightColor;
        xmath::fvec4    m_wSpaceLightPos;       // w == 0: xyz = direction towards the light; w > 0: position + radius
        xmath::fvec4    m_wSpaceEyePos;
        xmath::fvec4    m_LightParams;          // x = area radius, y = temperature K (0 = none), z = intensity mult
    };

    // xlionrender_forward_standard_frag.glsl "material_uniforms" (set 2, binding 2)
    struct alignas(256) ubo_material
    {
        xmath::fvec4    m_BaseColor;
        xmath::fvec4    m_Params;               // x = metallic, y = roughness, z = AO, w = F0
        xmath::fvec4    m_Emissive;
    };

    // xlionrender_tonemap_frag.glsl
    struct push_tonemap
    {
        xmath::fvec4    m_Params;               // x = exposure
        xmath::fvec4    m_Background;           // render_view::m_Background
    };
}

#endif
