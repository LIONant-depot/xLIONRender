#ifndef XLIONRENDER_SCENE_H
#define XLIONRENDER_SCENE_H
#pragma once

// The snapshot a tool or the game hands to the renderer: plain data, no GPU objects, no ECS.
// Meshes are GPU resources owned by an xlionrender::instance and referenced here by id.

#include "dependencies/xmath/source/xmath.h"
#include <cstdint>
#include <vector>
#include <optional>

namespace xgpu { struct texture; }

namespace xlionrender
{
    // Vertex layout of meshes created through instance::CreateMesh. Matches the mb_* varying interface
    // (position, UV, color, normal, tangent with the binormal sign in w).
    struct vertex
    {
        float           m_Pos[3];
        float           m_UV[2];
        std::uint32_t   m_Color;
        float           m_Normal[3];
        float           m_Tangent[4];
    };

    // Generation-checked ids: a destroyed slot can be reused without old ids reaching the new object
    struct mesh_id
    {
        std::uint32_t   m_Index         = ~0u;
        std::uint32_t   m_Generation    = 0;
        bool isValid() const noexcept { return m_Index != ~0u; }
    };

    // A compiled material (its HDR shader + textures), see instance::CreateMaterial
    struct material_id
    {
        std::uint32_t   m_Index         = ~0u;
        std::uint32_t   m_Generation    = 0;
        bool isValid() const noexcept { return m_Index != ~0u; }
        bool operator==(const material_id&) const noexcept = default;
    };

    // Standard metal/roughness material (glTF-style), scene-linear values
    struct material
    {
        xmath::fvec3    m_BaseColor     { 0.8f, 0.8f, 0.8f };
        float           m_Metallic      = 0.0f;
        float           m_Roughness     = 0.5f;
        float           m_AO            = 1.0f;
        float           m_F0            = 0.04f;        // dielectric reflectance
        xmath::fvec3    m_Emissive      { 0, 0, 0 };
    };

    // A draw's entity parameters (plugins/xmaterial.plugin/documentation/material_contract.md section 4): the EntityParams entry of its
    // instance slot (std430, 64 B). A field replaces the material's value where its m_Mask bit is set and the material opts the
    // parameter into that field; mask 0 (the default): every material reads its own values
    struct entity_params
    {
        enum : std::uint32_t { TINT = 1u << 0, PARAM0 = 1u << 1, PARAM1 = 1u << 2 };
        std::uint32_t   m_Mask      = 0;
        std::uint32_t   m_SlotMask  = 0;                // reserved (per-slot targeting), 0 in v1
        std::uint32_t   m_Pad[2]    = {};
        xmath::fvec4    m_Tint      { 0, 0, 0, 0 };     // linear rgba
        xmath::fvec4    m_Param0    { 0, 0, 0, 0 };
        xmath::fvec4    m_Param1    { 0, 0, 0, 0 };
    };
    static_assert(sizeof(entity_params) == 64);

    struct draw
    {
        mesh_id                     m_Mesh;
        xmath::fmat4                m_L2W           = xmath::fmat4::fromIdentity();
        material                    m_Material;         // used where m_Materials has no valid entry for a material slot
        std::vector<material_id>    m_Materials;        // per material slot of the mesh (xgeom_static submesh m_iMaterial)
        std::uint64_t               m_UserID        = 0;    // e.g. the entity; what editor pick/ID passes write
        // Low 32 bits: the caller's instance id - its data stays in the renderer's instance buffer across frames and is
        // written only when it changes (0: none, written every frame). The caller owns the ids (unique among its draws).
        // High 32 bits: flags. Bits 32-33: its gi_role (draw_gi_role_shift_v).
        std::uint64_t               m_Instance      = 0;
        entity_params               m_EntityParams;     // read by the materials through its instance slot
    };

    // What an instance does in the GI's transport (documentation/gi_dynamic_master_plan.md, Phase 5), overriding its geometry's GI mode:
    // RECEIVER is lit by the GI but neither occludes its rays nor bounces or emits into it (a prop, a character); FULL does all of it
    enum class gi_role : std::uint8_t { ASSET = 0, RECEIVER = 1, FULL = 2 };
    constexpr int draw_gi_role_shift_v = 32;

    // The directional light (the sun): shades everything, shadowed by the shadow clipmap
    struct sun_light
    {
        xmath::fvec3    m_Direction     { -0.5f, -1.0f, -0.3f };    // the direction the light travels
        float           m_Temperature   = 0;                        // Kelvin tint, 0 = none
        xmath::fvec3    m_Color         { 1, 1, 1 };
        float           m_Intensity     = 4.0f;                     // ponytail: unitless until the physical-unit contract (research doc §6) is approved
        bool            m_bCastShadows  = true;
        float           m_ShadowTileSize = 1.0f;                    // world size of a clipmap level-0 tile (512 texels); the levels double it
    };

    // A point or spot light (up to max_local_lights_v per scene, the nearest first is up to the caller); shadows from atlas tiles:
    // 6 for a point light (a cube), 1 for a spot
    struct local_light
    {
        enum class type : std::uint8_t { POINT, SPOT };

        type            m_Type          = type::POINT;
        xmath::fvec3    m_Position      { 0, 0, 0 };
        xmath::fvec3    m_Direction     { 0, -1, 0 };               // SPOT: where it points
        float           m_Radius        = 5.0f;                     // falloff radius (zero light at and beyond it)
        float           m_InnerAngle    = 0.4f;                     // SPOT: radians from the axis: full light inside, fading to the outer
        float           m_OuterAngle    = 0.6f;                     // SPOT: radians (at most ~1 rad: a 115 degree shadow frustum)
        xmath::fvec3    m_Color         { 1, 1, 1 };
        float           m_Intensity     = 4.0f;
        bool            m_bCastShadows  = true;
    };
    constexpr int max_local_lights_v = 32;

    // The sky: a procedural gradient that lights the scene through the environment probe (image-based: diffuse + specular).
    // m_Intensity 0: off - the flat m_Ambient lights instead. The sun is the direct light (it is not in the probe: no double count).
    struct sky
    {
        xmath::fvec3    m_Zenith        { 0.20f, 0.40f, 0.85f };
        xmath::fvec3    m_Horizon       { 0.70f, 0.78f, 0.90f };
        xmath::fvec3    m_Ground        { 0.22f, 0.20f, 0.18f };
        float           m_Intensity     = 0.0f;
        bool            m_bVisible      = false;        // the cubemap is also drawn as the background: seen from inside an infinitely far sphere, behind everything (m_Intensity times its colors)
        const xgpu::texture* m_pCubemap = nullptr;      // a cubemap texture lights instead of the gradient (its colors times m_Intensity; HDR or not)
    };

    // A reflection probe: the renderer captures the scene around m_Position (6 faces, prefiltered like the sky) and surfaces in its
    // box reflect it, box-projected (parallax); the box's faces fade it over m_Blend. Re-captured when a caster in its box changes,
    // a local light reaching its box changes, or the sun / sky / ambient change.
    struct reflection_probe
    {
        xmath::fvec3    m_Position  { 0, 0, 0 };
        xmath::fvec3    m_BoxMin    { -5, -5, -5 };         // world: influence and parallax box
        xmath::fvec3    m_BoxMax    {  5,  5,  5 };
        float           m_Blend     = 1.0f;
    };

    // Diffuse GI (features/gi/xlionrender_surfel_gi.h): draws of meshes given surfels (instance::SetMeshGI) with a persistent
    // m_Instance id; solved on the CPU (m_RayBudget rays per render, the changed first), read per vertex (surfel weights)
    // The GI's debug views (shaders/xlionrender_surfel_gi.glsl): every pixel the color of the surfel that dominates its read (black: none
    // reaches it - a gap in the placement), or the GI light alone (no textures, no direct light)
    enum class gi_debug : std::uint8_t { NONE = 0, SURFELS = 1, LIGHT = 2, SURFACE_COLOR = 4, COVERAGE = 5 };      // the shaders' codes (3: the GI's material capture; xlionrender_surfel_gi.glsl GIDebugCode)

    struct gi_settings
    {
        bool            m_bEnable       = false;
        int             m_RayBudget     = 16384;
        bool            m_bPerPixel     = true;     // near (the fine surfels) the pixels read the surfels themselves (the mesh's read grid), far the vertices
        gi_debug        m_Debug         = gi_debug::NONE;
        int             m_RaysPerSurfel = 8;        // rays a surfel casts per update (stratified over its hemisphere): more is less noise, fewer surfels per frame
        int             m_FilterPasses  = 4;        // the shown GI light's spatial filter: each pass blends every surfel with its neighbors (0: none)
        bool            m_bMoments      = true;     // the reads' depth-moment test (a surfel hidden from the point is faded); off: only the plane test
        float           m_TraceAllowanceMs = 1.0f;  // the GI trace's main-thread time per frame (documentation/gi_dynamic_master_plan.md, Phase 1); 0: no limit
        bool            m_bGPUTrace     = true;     // the GI's rays on the GPU (Part II, G3: gi::gpu_trace); off, or no compute: on the CPU within m_TraceAllowanceMs
        bool            m_bEmitterSampling = true;  // the emitters' light by light sampling in each pass (G5), not the links the CPU builds (a minute for Police)
        float           m_GPUBudgetMs   = 2.0f;     // the GPU time a frame's GI slice may take (its rays: the measured rays per ms times this); 0: no limit, every batch (m_RayBudget) in one slice
        float           m_PixelsPerSurfel = 256;    // the screen area (pixels) an instance needs per fine surfel: below it the GI holds its parent surfels, whatever its shape
        float           m_Saturation    = 1;        // the shown GI light's saturation (1: as solved; luminance kept): the filter and the coarse surfels average its colors toward grey
        float           m_EmissiveScale = 100;      // self-illuminated materials are lights with no light entity, their emission a texture color (0..1): the GI takes it this many times brighter (their look on screen stays)
    };

    struct scene
    {
        std::vector<draw>           m_Draws;
        sun_light                   m_Sun;
        std::vector<local_light>    m_LocalLights;
        xmath::fvec3                m_Ambient   { 0, 0, 0 };
        sky                         m_Sky;
        std::uint64_t               m_SelectedUser = ~0ull;     // the draw (its m_UserID) the editor has selected: outlined with the editor's outline shader (xlionrender_outline_obb_*)
        gi_settings                 m_GI;
        std::vector<reflection_probe> m_Probes;           // any number: the nearest 32 (instance::probe_slots_v) are resident
    };
}

#endif
