#ifndef XLIONRENDER_SYSTEM_H
#define XLIONRENDER_SYSTEM_H
#pragma once

// The render system - compiled entirely inside LIONRender.dll (see xlionrender_plugin_entry.cpp, the
// only place RegisterComponents<primitive>()/RegisterSystems<system>() are called). Same reasoning as
// xlioncore::physics::system: xecs::component::type::info_v<T> is per-binary, so registration and
// every query/entity-iteration call for a component have to be compiled into the same binary - this
// one, not xLION.exe. Queries xlioncore::transform for pose (shared with physics); pick also needs
// rigid_body for HalfExtents. OnUpdate only ever calls SubmitInternal (xlionrender_internal.h) - it
// does not touch xGPU/cmd_buffer at all; the actual draw calls happen later, from the host's own
// turn, via the exported xlionrender::Draw (xlionrender_api.h) - see xlionrender_renderer.h's comment.
//
// Collect() (not OnUpdate) is what actually runs the query - called from Draw every frame regardless
// of Play state (see xlionrender_api.cpp). Editor-created entities only become visible to Search/Foreach
// after xecs::archetype::mgr::UpdateStructuralChanges() flushes their pending pool append - the host
// calls that unconditionally each frame too (xlevel_session.h), not just while Playing.
#include "xlionrender_primitive.h"
#include "xlionrender_internal.h"
#include "dependencies/xLIONCore/src/transform/xlioncore_transform.h"
#include "dependencies/xLIONCore/src/physics/xlioncore_physics.h"
#include "dependencies/xeditor_tools/src/xeditor_tools_picking.h"

namespace xlionrender
{
    struct system : xecs::system::instance
    {
        constexpr static auto typedef_v = xecs::system::type::update{ .m_pName = "Render" };
        using query = std::tuple<xecs::query::must<xlioncore::transform, primitive>>;

        system(xecs::game_mgr::instance& GameMgr) noexcept : xecs::system::instance(GameMgr) {}

        void OnCreate(void)  noexcept { SetActiveSystemInternal(this); }
        void OnDestroy(void) noexcept { SetActiveSystemInternal(nullptr); }
        void OnUpdate(void)  noexcept {} // required by xECS (else it Foreach's operator()); work is in Collect

        void Collect(void) noexcept
        {
            xecs::query::instance Query;
            Query.m_Must.AddFromComponents<xlioncore::transform, primitive>();
            auto S = Search(Query);

            Foreach(S, [&](const xecs::component::entity& Ent, const xlioncore::transform& T, const primitive& Prim) noexcept
            {
                xmath::fmat4 L2W;
                L2W.setupSRT(T.m_Scale, T.m_Rotation, T.m_Position);
                SubmitInternal(Prim.m_Shape, L2W, Prim.m_Color, Ent.m_Value);
            });
        }

        // Ray-vs-AABB: Transform position +/- (rigid_body HalfExtents * Transform Scale). Same generous-solid-volume
        // philosophy as xskeleton.plugin's PickWedge, using the shared primitives in
        // xeditor_tools_picking.h. Closest hit wins.
        std::uint64_t Pick(const xmath::fvec3& Origin, const xmath::fvec3& Dir) noexcept
        {
            xecs::query::instance Query;
            Query.m_Must.AddFromComponents<xlioncore::transform, xlioncore::physics::rigid_body, primitive>();
            auto S = Search(Query);

            xeditor_tools::picking::closest_hit<std::uint64_t> Hit;
            Foreach(S, [&](const xecs::component::entity& Ent, const xlioncore::transform& T, const xlioncore::physics::rigid_body& RB, const primitive&) noexcept
            {
                float THit;
                if (xeditor_tools::picking::RayAABBIntersect(Origin, Dir, T.m_Position, RB.m_HalfExtents * T.m_Scale, THit))
                    Hit.Consider(Ent.m_Value, THit);
            });
            return Hit.isValid() ? Hit.m_Id : xecs::component::entity::invalid_entity_v;
        }
    };
}

#endif // XLIONRENDER_SYSTEM_H
