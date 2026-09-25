#ifndef XLIONRENDER_SYSTEM_H
#define XLIONRENDER_SYSTEM_H
#pragma once

// The render system - compiled entirely inside LIONRender.dll (see xlionrender_plugin_entry.cpp, the
// only place RegisterComponents<primitive>()/RegisterSystems<system>() are called). Same reasoning as
// xlioncore::physics::system: xecs::component::type::info_v<T> is per-binary, so registration and
// every query/entity-iteration call for a component have to be compiled into the same binary - this
// one, not xLION.exe. Queries xlioncore::physics::rigid_body for position (a real, header-level
// dependency between the two DLLs is fine - what mattered was never linking against another DLL's
// compiled ECS glue, and this doesn't). OnUpdate only ever calls SubmitInternal (xlionrender_internal.h)
// - it does not touch xGPU/cmd_buffer at all; the actual draw calls happen later, from the host's own
// turn, via the exported xlionrender::Draw (xlionrender_api.h) - see xlionrender_renderer.h's comment.
//
// Collect() (not OnUpdate) is what actually runs the query - called from Draw every frame regardless
// of Play state (see xlionrender_api.cpp). Editor-created entities only become visible to Search/Foreach
// after xecs::archetype::mgr::UpdateStructuralChanges() flushes their pending pool append - the host
// calls that unconditionally each frame too (xlevel_session.h), not just while Playing.
#include "xlionrender_primitive.h"
#include "xlionrender_internal.h"
#include "dependencies/xLIONCore/src/physics/xlioncore_physics.h"
#include "dependencies/xeditor_tools/src/xeditor_tools_picking.h"

namespace xlionrender
{
    struct system : xecs::system::instance
    {
        constexpr static auto typedef_v = xecs::system::type::update{ .m_pName = "Render" };
        using query = std::tuple<xecs::query::must<xlioncore::physics::rigid_body, primitive>>;

        system(xecs::game_mgr::instance& GameMgr) noexcept : xecs::system::instance(GameMgr) {}

        void OnCreate(void)  noexcept { SetActiveSystemInternal(this); }
        void OnDestroy(void) noexcept { SetActiveSystemInternal(nullptr); }
        void OnUpdate(void)  noexcept {} // required by xECS (else it Foreach's operator()); work is in Collect

        void Collect(void) noexcept
        {
            xecs::query::instance Query;
            Query.m_Must.AddFromComponents<xlioncore::physics::rigid_body, primitive>();
            auto S = Search(Query);

            Foreach(S, [&](const xecs::component::entity& Ent, const xlioncore::physics::rigid_body& RB, const primitive& Prim) noexcept
            {
                // Rotation isn't tracked back from box3d yet (xlioncore::physics::system only syncs
                // position) - translation-only for now, a natural follow-up once rigid_body's rotation
                // is kept in sync too.
                const auto L2W = xmath::fmat4::fromTranslation(RB.m_Position);
                SubmitInternal(Prim.m_Shape, L2W, Prim.m_Color, Ent.m_Value);
            });
        }

        // Ray-vs-AABB against every entity's own rigid_body volume (Position +/- HalfExtents) - the
        // same generous-solid-volume philosophy as xskeleton.plugin's PickWedge, using the shared
        // primitives in xeditor_tools_picking.h instead of duplicating them. Closest hit wins.
        std::uint64_t Pick(const xmath::fvec3& Origin, const xmath::fvec3& Dir) noexcept
        {
            xecs::query::instance Query;
            Query.m_Must.AddFromComponents<xlioncore::physics::rigid_body, primitive>();
            auto S = Search(Query);

            xeditor_tools::picking::closest_hit<std::uint64_t> Hit;
            Foreach(S, [&](const xecs::component::entity& Ent, const xlioncore::physics::rigid_body& RB, const primitive&) noexcept
            {
                float T;
                if (xeditor_tools::picking::RayAABBIntersect(Origin, Dir, RB.m_Position, RB.m_HalfExtents, T))
                    Hit.Consider(Ent.m_Value, T);
            });
            return Hit.isValid() ? Hit.m_Id : xecs::component::entity::invalid_entity_v;
        }
    };
}

#endif // XLIONRENDER_SYSTEM_H
