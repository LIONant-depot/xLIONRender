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
#include "xlionrender_primitive.h"
#include "xlionrender_text.h"
#include "xlionrender_text_layout.h"
#include "xlionrender_text_renderer.h"
#include "xlionrender_internal.h"
#include "dependencies/xLIONCore/src/resources/xlioncore_resources_api.h"
#include "dependencies/xGPU/source/xgpu.h"
#include "dependencies/xLIONCore/src/transform/xlioncore_transform.h"
#include "dependencies/xLIONCore/src/transform/xlioncore_hierarchy.h"
#include "dependencies/xLIONCore/src/physics/xlioncore_physics.h"
#include "dependencies/xeditor_tools/src/xeditor_tools_picking.h"
#include <cstring>
#include <format>
#include <functional>
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

        // The Texts of the world: their layouts are kept up to date (the drawing of them comes next to this). The resource system of the core starts here, the first time there is a device.
        void CollectText(void) noexcept
        {
            ++m_Frame;
            if (auto* pDevice = GetDeviceInternal())
                xlioncore::resources::Initialize(*pDevice, getGameMgr().m_SceneMgr.m_ProjectPath.c_str());

            xecs::query::instance Query;
            Query.m_Must.AddFromComponents<xlioncore::transform, text>();
            auto S = Search(Query);
            int nSeen = 0, nWithLayout = 0;
            Foreach(S, [&](const xecs::component::entity& Ent, const xlioncore::transform& Local, const text& Tx, const xecs::component::parent* pParent) noexcept
            {
                const auto T = xlioncore::WorldOf(Local, pParent);   // a child's Transform is relative to its parent: what is drawn is at its world pose
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
            std::erase_if(m_Texts, [&](const auto& Pair) noexcept { return Pair.second.m_Frame != m_Frame; });
        }

        void Collect(void) noexcept
        {
            xlioncore::PropagateHierarchy(*this);       // the world pose of every child, from the Transforms as they are now (this is after everything that moves things)
            CollectText();
            xecs::query::instance Query;
            Query.m_Must.AddFromComponents<xlioncore::transform, primitive>();
            auto S = Search(Query);

            Foreach(S, [&](const xecs::component::entity& Ent, const xlioncore::transform& Local, const primitive& Prim, const xecs::component::parent* pParent) noexcept
            {
                const auto T = xlioncore::WorldOf(Local, pParent);   // a child's Transform is relative to its parent: what is drawn is at its world pose
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

        // Ray-vs-OBB: Transform Position/Rotation + half-extents 0.5*Scale (matches Collect L2W).
        // Closest hit wins. MaxT caps the ray (e.g. ground-plane hit) so empty floor clears
        // selection instead of selecting through distant AABBs.
        std::uint64_t Pick(const xmath::fvec3& Origin, const xmath::fvec3& Dir, float MaxT = std::numeric_limits<float>::max()) noexcept
        {
            xecs::query::instance Query;
            Query.m_Must.AddFromComponents<xlioncore::transform, primitive>();
            auto S = Search(Query);

            xeditor_tools::picking::closest_hit<std::uint64_t> Hit;
            Foreach(S, [&](const xecs::component::entity& Ent, const xlioncore::transform& Local, const primitive&, const xecs::component::parent* pParent) noexcept
            {
                const auto T = xlioncore::WorldOf(Local, pParent);   // a child's Transform is relative to its parent: what is drawn is at its world pose
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
            auto TS = Search(TextQuery);
            Foreach(TS, [&](const xecs::component::entity& Ent, const xlioncore::transform& Local, const text& Tx, const xecs::component::parent* pParent) noexcept
            {
                const auto T = xlioncore::WorldOf(Local, pParent);   // a child's Transform is relative to its parent: what is drawn is at its world pose
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
            return Hit.isValid() ? Hit.m_Id : xecs::component::entity::invalid_entity_v;
        }
    };
}

#endif // XLIONRENDER_SYSTEM_H