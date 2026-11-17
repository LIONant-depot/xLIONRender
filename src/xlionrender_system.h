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
#include "xlionrender_view.h"
#include "xlionrender_primitive.h"
#include "xlionrender_text.h"
#include "xlionrender_text_layout.h"
#include "xlionrender_text_renderer.h"
#include "xlionrender_internal.h"
#include "xlionrender_scene_renderer.h"
#include "dependencies/xLIONCore/src/resources/xlioncore_resources_api.h"
#include "dependencies/xGPU/source/xGPU.h"
// LIONCore.dll owns (registers) these components; this DLL only needs their types/reflection, so
// they are read import-only (see xscript_registration.h). Without this, every compiler that runs
// the dynamic initializers of the headers' inline g_AutoReg_* variables (clang/gcc always do; MSVC
// happens to drop the unreferenced ones) gives LIONRender its own registration of transform,
// static_tag, the physics components, ... - a second bit per GUID, and whichever copy sorts last
// wins findComponentTypeInfo, leaving LIONCore's own info_v copies (which its systems query)
// pointing at a bit no entity has (Linux: the ground's static_tag was invisible to physics, so it
// became a kinematic body). The few LIONCore types this DLL queries are synced by GUID in
// XecsPlugin_RegisterSystems (xlionrender_plugin_entry.cpp).
#define XSCRIPT_IMPORT_ONLY
#include "dependencies/xLIONCore/src/transform/xlioncore_transform.h"
#include "dependencies/xLIONCore/src/transform/xlioncore_hierarchy.h"
#include "dependencies/xLIONCore/src/tags/xlioncore_tags.h"
#include "dependencies/xLIONCore/src/physics/xlioncore_physics.h"
#include "dependencies/xLIONCore/src/game/xlioncore_game.h"
#undef  XSCRIPT_IMPORT_ONLY
#include "plugins/xscript_module.plugin/source/Runtime/xscript_registration.h"   // registering macro back for anything after
#include "dependencies/xeditor_tools/src/xeditor_tools_picking.h"
#include "dependencies/xLIONCore/src/physics/xlioncore_ragdoll_api.h"
#include "dependencies/xLIONCore/src/physics/xlioncore_physics_api.h"     // CastRayClosest (the feet's ground, K3)
#include "dependencies/xstrtool/source/xstrtool.h"           // (a clip requested by name: its hash, as the compiler makes it)
#include "plugins/xragdoll.plugin/source/xragdoll_body_graph.h"
#include <cstring>
#include <format>
#include <functional>
#include <memory>
#include <mutex>
#include <chrono>
#include <limits>
#include <string>
#include <unordered_map>

namespace xlionrender
{
    // The Animation system: while the game runs, each playing Animation's time moves on by the frame's game time x its speed - wrapped into its
    // clip when it loops, stopped at an end when it does not (Playing off). The clip's length from its package as loaded (the Animation's, else
    // its Skinned Mesh's geometry's own); not loaded yet: the time just moves. The pose is the renderer's (scene_renderer::AddSkinnedMesh)
    struct animation_system : xecs::system::instance
    {
        constexpr static auto typedef_v = xecs::system::type::update{ .m_pName = "Animation" };
        animation_system(xecs::game_mgr::instance& GameMgr) noexcept : xecs::system::instance(GameMgr) {}

        // K0c: the Animation system owns the pose. After every character's update (time, root motion, matching, warp, ground casts), each
        // skinned character with an Animation is posed once, on game time, in the frame it is drawn at (its world transform as the renderer
        // reads it: hierarchy propagated, fixed-step interpolation); the renderer draws that pose (scene_renderer::m_pUpdated) and poses only
        // what this did not (an active ragdoll, a world that does not update). Before: posed per view, per-entity state on the wall clock.
        anim_poser m_Poser;
        // K4's budgeted search queue: at most SearchBudget motion matching searches an update (XLION_ANIM_SEARCH_BUDGET, default 64); a character
        // due for a search past it waits for the next update (its clip plays on; one that is ending still searches). A crowd's searches spread
        // over frames instead of piling on one. ponytail: the same entity order every update - past ~6x the budget in characters searching at 10 Hz the
        // last ones would starve; a round-robin start when crowds get there
        int m_Searches = 0, m_SearchWaits = 0;
        static int SearchBudget(void) noexcept { static const int B = [] { const char* e = std::getenv("XLION_ANIM_SEARCH_BUDGET"); return e && *e ? std::max(1, std::atoi(e)) : 64; }(); return B; }
        void OnPreUpdate(void) noexcept { m_Searches = 0; }
        void OnPostUpdate(void) noexcept
        {
            if (scene_renderer::AnimMode() != scene_renderer::anim_mode::NEW) return;
            const auto* pGame = xlioncore::game::From(getGameMgr());
            if (pGame == nullptr) return;
            XPROFILE_SCOPE("Animation: pose (update)");
            using namespace xlioncore::resources;
            xlioncore::PropagateHierarchy(*this);
            ++m_Poser.m_Update; m_Poser.m_GameDt = std::max(0.0f, pGame->m_Time.m_DeltaTime);
            const auto Interpolate = xlioncore::FixedInterpolateOf(getGameMgr());
            xecs::query::instance Query;
            Query.m_Must.AddFromComponents<xlioncore::transform, skinned_mesh, animation>();
            Foreach(Search(Query), [&](const xecs::component::entity& Ent, const xlioncore::transform& Local, const skinned_mesh& Skin, const animation& A
                                      , const xecs::component::parent* pParent, const xlioncore::render_transform* pRender, const ragdoll* pRagdoll) noexcept
            {
                if (pRagdoll && pRagdoll->m_bActive) return;           // (the renderer poses it from its bodies)
                auto* pGeom = Skin.m_Geom.empty() ? nullptr : static_cast<xgeom_skin::xgpu::geom*>(GetResource(geom_skin_type_guid_v.m_Value, Skin.m_Geom.m_Instance.m_Value));
                auto* pSkel = pGeom ? static_cast<xskeleton::skeleton*>(GetResource(skeleton_type_guid_v.m_Value, pGeom->getSkeletonRef().m_Instance.m_Value)) : nullptr;
                if (pGeom == nullptr || pSkel == nullptr || pSkel->getBones().empty()) return;
                const auto T = xlioncore::WorldOf(Local, pParent, pRender, Interpolate);
                if (!T.m_Scale.isFinite() || T.m_Scale.m_X == 0.0f || T.m_Scale.m_Y == 0.0f || T.m_Scale.m_Z == 0.0f) return;
                xmath::fmat4 L2W; L2W.setupSRT(T.m_Scale, T.m_Rotation, T.m_Position);
                const std::uint64_t Pkg = A.m_Package.empty() ? pGeom->getAnimationRef().m_Instance.m_Value : A.m_Package.m_Instance.m_Value;
                auto* pPkg = Pkg ? static_cast<xanim_package::anim_package*>(GetResource(anim_package_type_guid_v.m_Value, Pkg)) : nullptr;
                const bool bClip = pPkg && A.m_Clip >= 0 && A.m_Clip < int(pPkg->getClips().size()) && pPkg->getBoneNameHashes().size() == pSkel->getBones().size();
                auto& P = m_Poser.m_Posed[Ent.m_Value];
                m_Poser.PoseEntity(*pSkel, pPkg, bClip, &A, L2W, Ent.m_Value, P.m_Worlds);
                P.m_Local = *m_Poser.m_pAnimUsed; P.m_Update = m_Poser.m_Update; P.m_pSkel = pSkel;
            });
            m_Poser.SweepPosed();
            // K3 sockets: each child with a Socket rides its parent character's bone, as posed just now (its Transform is relative to the parent:
            // the bone's place and turn in the character's frame, then the Offset and Turn in the bone's frame; its Scale stays its own)
            xecs::query::instance QS;
            QS.m_Must.AddFromComponents<xlioncore::transform, socket, xecs::component::parent>();
            Foreach(Search(QS), [&](xlioncore::transform& T, socket& S, const xecs::component::parent& Par) noexcept
            {
                const auto* pP = m_Poser.Posed(Par.m_Value.m_Value);
                if (pP == nullptr || pP->m_pSkel == nullptr) { S.m_iBone = -1; return; }
                if (S.m_pSkel != pP->m_pSkel || S.m_iBone < 0) { S.m_pSkel = pP->m_pSkel; S.m_iBone = S.m_Bone.empty() ? -1 : pP->m_pSkel->findBoneIndex(xstrtool::CRC32(S.m_Bone)); }
                if (S.m_iBone < 0 || S.m_iBone >= int(pP->m_Worlds.size())) return;
                const auto& M = pP->m_Worlds[std::size_t(S.m_iBone)];
                const auto R = M.ExtractRotation().NormalizeSafeCopy();
                auto Axis = [](float x, float y, float z, float Deg) { const float h = 0.5f * Deg * 0.017453293f, s = std::sin(h); return xmath::fquat(x * s, y * s, z * s, std::cos(h)); };
                const auto Turn = Axis(0, 0, 1, S.m_Turn.m_Z) * Axis(0, 1, 0, S.m_Turn.m_Y) * Axis(1, 0, 0, S.m_Turn.m_X);      // (about X, then Y, then Z)
                T.m_Position = M.ExtractPosition() + R * S.m_Offset;
                T.m_Rotation = (R * Turn).NormalizeSafeCopy();
            });
        }

        std::unordered_map<const xskeleton::skeleton*, xanimation::skeleton_data> m_Skeletons;        // (a mirrored clip's root motion: the skeleton's mirror table)
        struct match_db { std::vector<xanimation::matching_clip> m_Clips; xanimation::matching_db m_Db; };
        std::unordered_map<const xanim_package::anim_package*, match_db> m_Match;                      // K4: one database per package (ponytail: a reloaded package at the same address keeps its old database until the editor restarts)

        void operator()( const xecs::component::entity& Ent, animation& A, const skinned_mesh* pSkin, xlioncore::transform* pT, const xecs::component::parent* pParent ) noexcept
        {
            A.m_nFired = 0; A.m_bFinished = false;              // this frame's events and result only
            A.m_WarpStride = 1; A.m_WarpAngle = 0;              // (and its warping)
            const auto* pGame = xlioncore::game::From(getGameMgr());
            if (pGame == nullptr) return;
            using namespace xlioncore::resources;
            std::uint64_t Pkg = A.m_Package.m_Instance.m_Value;
            if (Pkg == 0 && pSkin && !pSkin->m_Geom.empty())
                if (auto* pGeom = static_cast<xgeom_skin::xgpu::geom*>(GetResource(geom_skin_type_guid_v.m_Value, pSkin->m_Geom.m_Instance.m_Value))) Pkg = pGeom->getAnimationRef().m_Instance.m_Value;
            auto* pPkg = Pkg ? static_cast<xanim_package::anim_package*>(GetResource(anim_package_type_guid_v.m_Value, Pkg)) : nullptr;
            // a request by name (K1 play): a new name switches the clip (from its start; the renderer eases it in); a one-shot remembers the way back
            if (A.m_ClipName.empty()) A.m_RequestHash = 0;     // (cleared: the same name asked again is a new request)
            if (pPkg && !A.m_ClipName.empty())
                if (const auto H = xstrtool::CRC32(A.m_ClipName); H != A.m_RequestHash)
                {
                    A.m_RequestHash = H;
                    if (const int i = pPkg->findClipIndex(H); i >= 0)
                    {
                        if (A.m_bOneShot) { A.m_ReturnClip = A.m_Clip; A.m_ReturnTime = A.m_Time; }
                        A.m_Clip = i; A.m_Time = 0; A.m_bPlaying = true;
                    }
                    else std::printf("[Animation] no clip named '%s' in the package: the request is ignored\n", A.m_ClipName.c_str());
                }
            if (!A.m_bPlaying)                                  // (paused: the feet still stand on the ground under them)
            {
                if (A.m_bFootIK && pT && pSkin && pPkg && A.m_Clip >= 0 && A.m_Clip < int(pPkg->getClips().size()))
                    if (const auto& C = pPkg->getClips()[std::size_t(A.m_Clip)]; C.m_FPS > 0 && C.m_nFrames > 0) GroundFeet(A, *pSkin, *pT, *pPkg, float(C.m_nFrames) / float(C.m_FPS));
                return;
            }
            const float Before = A.m_Time;
            A.m_Time += pGame->m_Time.m_DeltaTime * A.m_Speed;
            if (pPkg == nullptr || A.m_Clip < 0 || A.m_Clip >= int(pPkg->getClips().size())) return;
            const auto& Clip = pPkg->getClips()[std::size_t(A.m_Clip)];
            const float Length = Clip.m_FPS > 0 ? float(Clip.m_nFrames) / float(Clip.m_FPS) : 0.0f;
            if (Length <= 0) return;
            const xanimation::skeleton_data* pSD = nullptr;        // mirrored: the skeleton's mirror table (events' bones, root motion)
            if (A.m_bMirror && pSkin && !pSkin->m_Geom.empty())
                if (auto* pGeom = static_cast<xgeom_skin::xgpu::geom*>(GetResource(geom_skin_type_guid_v.m_Value, pSkin->m_Geom.m_Instance.m_Value)))
                    if (auto* pSkel = static_cast<xskeleton::skeleton*>(GetResource(skeleton_type_guid_v.m_Value, pGeom->getSkeletonRef().m_Instance.m_Value)); pSkel && pSkel->m_nMirror)
                    {
                        auto& SD = m_Skeletons[pSkel]; if (SD.m_Parent.size() != pSkel->getBones().size()) SD = xanimation::MakeSkeleton(*pSkel);
                        if (!SD.m_Mirror.empty()) pSD = &SD;
                    }
            if (A.m_bPlayer)                                    // K4: the player's stick is the move (world, camera-relative); it faces where it goes
            {
                const auto& In = pGame->m_Input;
                const float Sp = (In.m_Buttons & xlioncore::game::input::RUN) ? A.m_PlayerRunSpeed : A.m_PlayerSpeed;
                A.m_bMoveInWorld = true; A.m_bFaceMove = true;
                A.m_Move = In.m_bActive ? xmath::fvec3(In.m_MoveWorld[0] * Sp, 0, In.m_MoveWorld[1] * Sp) : xmath::fvec3(0, 0, 0);
            }
            xmath::fvec3 FollowFace(0, 0, 0);                  // G0 FollowParent: the parent's velocity (world) is the move, and where it goes is the facing asked
            if (A.m_bFollowParent && pParent && pGame->m_Time.m_DeltaTime > 0)
            {
                const xmath::fvec3 W(pParent->m_WorldPosition[0], pParent->m_WorldPosition[1], pParent->m_WorldPosition[2]);
                xmath::fvec3 V = A.m_bHasLastWorld ? (W - A.m_LastWorld) * (1.0f / pGame->m_Time.m_DeltaTime) : xmath::fvec3(0, 0, 0);
                // (a physics body's own velocity when the parent has one: the world position the hierarchy last derived is the drawn,
                // fixed-step-interpolated one of the last frame, and its difference jittered the move - SoccerAnim's players churned clips)
                (void)findEntity(pParent->m_Value, [&](const xlioncore::physics::physics_dynamics& D) noexcept { V = D.m_LinearVelocity; });
                V.m_Y = 0; if (V.Length() > 20.0f) V = xmath::fvec3(0, 0, 0);         // (a teleport: no speed)
                A.m_LastWorld = W; A.m_bHasLastWorld = true;
                A.m_bMoveInWorld = true; A.m_bFaceMove = false; A.m_Move = V;
                if (V.Length() > 0.3f) FollowFace = V;
            }
            const xmath::fvec3 MoveAsked = A.m_Move;              // (MoveInWorld: turned into the entity's frame for this frame's work, given back after)
            if (A.m_bMoveInWorld && pT) A.m_Move = pT->m_Rotation.ConjugateCopy() * A.m_Move;
            struct restore_move { animation& A; xmath::fvec3 M; ~restore_move() { A.m_Move = M; } } RestoreMove{ A, MoveAsked };
            // Collide: the move the world lets through (a probe of the asked move through the mover: into a wall, only its part along it) is what
            // warping and matching fit the clip to - asked as it was, the wall walker kept walking into the wall and her feet slid 115-125 cm/s
            // (motion trace) while the capsule slid along it (as Unreal's anim graph reads the movement component's velocity after collision)
            if (A.m_bCollide && pT && (A.m_Move.m_X != 0 || A.m_Move.m_Z != 0) && pGame->m_Time.m_DeltaTime > 0)
            {
                const float Dt = pGame->m_Time.m_DeltaTime;
                auto Out = Collide(Ent.m_Value, A, *pT, pT->m_Rotation * (A.m_Move * Dt), true) * (1.0f / Dt);
                Out = pT->m_Rotation.ConjugateCopy() * Out; A.m_Move = xmath::fvec3(Out.m_X, 0, Out.m_Z);
            }
            // stride and orientation warping (K2): the game asks a velocity (Move), the clip walks at its own; the clip's time runs at the Rate,
            // its root motion turns by the Angle and stretches by the Stride, and the renderer warps the pose the same way (the feet stay planted)
            xanimation::warp_params Wp;
            if (pT && (A.m_bRootMotion || A.m_bFollowParent) && A.m_Speed > 0 && Clip.m_RootMotionMode != xanim_package::root_motion_mode::NONE && (A.m_Move.m_X != 0 || A.m_Move.m_Z != 0))
            {
                auto CV = (A.m_bMatching ? xanimation::ClipVelocityAt(*pPkg, A.m_Clip, Before) : xanimation::ClipVelocity(*pPkg, A.m_Clip)) * (A.m_Speed * pT->m_Scale.m_X);      // (matching: around the frame playing - starts and stops do not walk evenly)      // ponytail: a uniform scale (X) for the clip's world speed
                if (pSD) { const auto N = xanimation::ClipMirrorNormal(*pSD, *pPkg, A.m_Clip); CV = CV - N * (2.0f * CV.Dot(N)); }
                Wp = xanimation::WarpFor(CV, A.m_Move);
                A.m_Time = Before + pGame->m_Time.m_DeltaTime * A.m_Speed * Wp.m_Rate;
            }
            // (eased, not snapped: a matching jump changes the clip's travel and with it the angle - up to 90 deg in a frame swung the legs:
            // the AnimDemo pops, seen in the motion trace. The pose and the root motion both take the eased values)
            {
                const float K = A.m_bWarpEased ? 1.0f - std::exp(-pGame->m_Time.m_DeltaTime / 0.15f) : 1.0f;
                float d = Wp.m_Angle - A.m_WarpAngleEased; while (d > 3.14159265f) d -= 6.2831853f; while (d < -3.14159265f) d += 6.2831853f;
                A.m_WarpAngleEased += d * K; A.m_WarpStrideEased += (Wp.m_Stride - A.m_WarpStrideEased) * K; A.m_bWarpEased = true;
                Wp.m_Angle = A.m_WarpAngleEased; Wp.m_Stride = A.m_WarpStrideEased;
                A.m_WarpStride = Wp.m_Stride; A.m_WarpAngle = Wp.m_Angle; A.m_WarpDir = Wp.m_Dir;
            }
            FireEvents(*pPkg, A, Before, Length, pSD ? &pSD->m_Mirror : nullptr);
            if (A.m_bPushOnContact && pT && pSkin && A.m_nFired) PushOnContact(A, *pSkin, *pT, *pPkg, Length);
            // root motion (K2, plan 13.7): the entity goes where the clip walks - the step the clip's root took this frame, in the entity's frame
            // (its rotation and scale; a child's Transform is its parent's frame); mirrored: reflected through the skeleton's plane
            if (pT && A.m_bRootMotion && Clip.m_RootMotionMode != xanim_package::root_motion_mode::NONE)
            {
                const auto Step = xanimation::RootStep(*pPkg, A.m_Clip, Before, A.m_bLoop ? A.m_Time : std::clamp(A.m_Time, 0.0f, Length));
                auto D = Step.m_Move; float Turn = Step.m_Turn;        // (the move in the character's frame now, the turn: a turning clip turns the entity)
                if (pSD)
                {
                    const auto N = xanimation::ClipMirrorNormal(*pSD, *pPkg, A.m_Clip);      // (the plane turned with the clip's facing, as the pose)
                    D = D - N * (2.0f * D.Dot(N));
                    Turn = -Turn;                                   // (a vertical mirror turns the other way)
                }
                if (Wp.active()) D = xanimation::details::AxisAngle(xmath::fvec3(0, 1, 0), Wp.m_Angle) * D * Wp.m_Stride;
                D = D + Wp.m_Slide * (pGame->m_Time.m_DeltaTime / std::max(1e-6f, pT->m_Scale.m_X));       // (past the warp's limits: still where the game asks; Move is world units, D is scaled below)
                auto WD = pT->m_Rotation * xmath::fvec3(D.m_X * pT->m_Scale.m_X, D.m_Y * pT->m_Scale.m_Y, D.m_Z * pT->m_Scale.m_Z);
                if (A.m_bCollide) WD = Collide(Ent.m_Value, A, *pT, WD);
                pT->m_Position = pT->m_Position + WD;
                if (Turn != 0) pT->m_Rotation = (pT->m_Rotation * xmath::fquat(0.0f, std::sin(0.5f * Turn), 0.0f, std::cos(0.5f * Turn))).NormalizeSafeCopy();     // (after the move: the move is in the facing it had)
            }
            struct restore_face { animation& A; xmath::fvec3 F; bool b; ~restore_face() { if (b) A.m_Face = F; } } RestoreFace{ A, A.m_Face, FollowFace.LengthSq() > 0 };
            if (FollowFace.LengthSq() > 0) A.m_Face = FollowFace;  // (FollowParent: it turns toward where it goes, as Face; given back after the frame)
            if (pT && (A.m_Face.m_X != 0 || A.m_Face.m_Z != 0))      // Face: what the clips' turn did not do, in place (at most 180 deg/s)
            {
                const auto F = pT->m_Rotation.ConjugateCopy() * A.m_Face;
                const float Err = std::atan2(F.m_X, F.m_Z), Max = 3.14159265f * pGame->m_Time.m_DeltaTime;
                const float Turn = std::clamp(Err, -Max, Max);
                if (std::abs(Turn) > 1e-5f) pT->m_Rotation = (pT->m_Rotation * xmath::fquat(0.0f, std::sin(0.5f * Turn), 0.0f, std::cos(0.5f * Turn))).NormalizeSafeCopy();
            }
            if (A.m_bMatching && pT && pSkin) Match(A, *pSkin, *pPkg, pT);
            if (A.m_bFootIK && pT && pSkin) GroundFeet(A, *pSkin, *pT, *pPkg, Length);
            else A.m_nGround = 0;
            if (A.m_bOneShot && A.m_ReturnClip >= 0)            // a one-shot: at its end the clip before it comes back where it was (Finished this frame)
            {
                if (A.m_Time >= Length || A.m_Time < 0)
                {
                    A.m_Clip = A.m_ReturnClip; A.m_Time = A.m_ReturnTime; A.m_ReturnClip = -1;
                    A.m_bOneShot = false; A.m_ClipName.clear(); A.m_RequestHash = 0; A.m_bFinished = true;
                }
                return;
            }
            if (A.m_bLoop)
            {
                const float Prev = A.m_Time;
                A.m_Time = std::fmod(A.m_Time, Length); if (A.m_Time < 0) A.m_Time += Length;
                if (A.m_bResetPushed && A.m_nPushed && A.m_Time < Prev - 0.5f * Length)       // (the clip started again: a drill's pushed bodies go back)
                {
                    for (int k = 0; k < A.m_nPushed; ++k)
                        { xecs::component::entity E; E.m_Value = A.m_Pushed[std::size_t(k)].m_Entity; xlioncore::physics::TeleportDynamicBody(getGameMgr(), E, A.m_Pushed[std::size_t(k)].m_Position, A.m_Pushed[std::size_t(k)].m_Rotation); }
                    A.m_nPushed = 0;
                }
            }
            else if (A.m_Time >= Length || A.m_Time < 0) { A.m_Time = std::clamp(A.m_Time, 0.0f, Length); A.m_bPlaying = false; }
        }

        // Collide (plan 13.4): the root motion's world step through Box3D's character mover (LIONCore MoveCapsule), with the other colliding
        // characters as planes (their capsules where they stood last; a vertical-axis pair: the plane between them).
        // ponytail: other characters by a list under a lock, all against all; a grid when crowds collide
        struct mover_info { xmath::fvec3 m_Pos; float m_Radius; std::chrono::steady_clock::time_point m_T; };
        std::mutex                                       m_MoversLock;
        std::unordered_map<std::uint64_t, mover_info> m_Movers;      // (by entity: the component handed to operator() may not keep its address)
        xmath::fvec3 Collide(std::uint64_t Entity, const animation& A, const xlioncore::transform& T, const xmath::fvec3& Step, bool bProbe = false) noexcept      // (bProbe: only asks, nothing recorded)
        {
            const auto Now = std::chrono::steady_clock::now();
            std::vector<xlioncore::physics::mover_plane> Planes;
            {
                std::scoped_lock Lk(m_MoversLock);
                for (auto It = m_Movers.begin(); It != m_Movers.end(); )
                {
                    if (Now - It->second.m_T > std::chrono::milliseconds(500)) { It = m_Movers.erase(It); continue; }       // (gone, or stopped colliding)
                    if (It->first != Entity)
                    {
                        xmath::fvec3 N(T.m_Position.m_X - It->second.m_Pos.m_X, 0, T.m_Position.m_Z - It->second.m_Pos.m_Z);
                        const float D = N.Length(), Reach = A.m_CapsuleRadius + It->second.m_Radius;
                        if (D > 1e-4f && D < Reach + Step.Length() + 0.1f) Planes.push_back({ N * (1.0f / D), Reach - D });      // (normal . move >= overlap)
                    }
                    ++It;
                }
            }
            const auto Out = xlioncore::physics::MoveCapsule(getGameMgr(), T.m_Position, A.m_CapsuleRadius, A.m_CapsuleHeight, 0.35f, Step, Planes);
            if (bProbe) return Out;
            std::scoped_lock Lk(m_MoversLock);
            static const bool bDbg = std::getenv("XLION_ANIM_MOVER_DEBUG") != nullptr;
            if (bDbg) std::printf("[AnimMover] %llu at (%.2f %.2f) step (%.3f %.3f) -> (%.3f %.3f), %zu others, %zu planes\n", (unsigned long long)Entity, T.m_Position.m_X, T.m_Position.m_Z, Step.m_X, Step.m_Z, Out.m_X, Out.m_Z, m_Movers.size(), Planes.size());
            m_Movers[Entity] = { T.m_Position + Out, A.m_CapsuleRadius, Now };
            return Out;
        }

        // Motion matching (K4): the package's database (every clip, built the first time), a search every 0.1 s with the wish from Move (in the
        // character's frame: the skeleton's rest forward is its +Z), a jump sets Clip and Time (and counts: the renderer eases it in).
        // ponytail: the wish keeps the rest's facing (no root rotation in our root motion yet); the velocity now is the last Move asked
        void Match(animation& A, const skinned_mesh& Skin, const xanim_package::anim_package& Pkg, const xlioncore::transform* pT) noexcept
        {
            using namespace xlioncore::resources;
            auto* pGeom = Skin.m_Geom.empty() ? nullptr : static_cast<xgeom_skin::xgpu::geom*>(GetResource(geom_skin_type_guid_v.m_Value, Skin.m_Geom.m_Instance.m_Value));
            auto* pSkel = pGeom ? static_cast<xskeleton::skeleton*>(GetResource(skeleton_type_guid_v.m_Value, pGeom->getSkeletonRef().m_Instance.m_Value)) : nullptr;
            if (!pSkel || Pkg.m_nBones != pSkel->getBones().size()) return;
            auto& SD = m_Skeletons[pSkel]; if (SD.m_Parent.size() != pSkel->getBones().size()) SD = xanimation::MakeSkeleton(*pSkel);
            if (!SD.m_Warp.valid()) return;
            auto& M = m_Match[&Pkg];
            if (M.m_Clips.size() != std::size_t(Pkg.m_nClips))
            {
                M.m_Clips.clear();
                for (int c = 0; c < int(Pkg.m_nClips); ++c)
                {
                    const auto& C = Pkg.getClips()[std::size_t(c)];
                    xanimation::matching_clip Mc;
                    Mc.m_Fps = float(std::max<int>(1, C.m_FPS)); Mc.m_Length = float(C.m_nFrames) / Mc.m_Fps; Mc.m_bLoop = C.m_bLoop;
                    Mc.m_Sample = [pPkg = &Pkg, c, pS = &SD](float t, xanimation::pose& P) { xanimation::pose Scr; xanimation::SampleClip(*pS, *pPkg, c, t, false, P, Scr); };
                    if (C.m_RootMotionMode != xanim_package::root_motion_mode::NONE) Mc.m_Root = [pPkg = &Pkg, c](float t) { return xanimation::RootMotionAt(*pPkg, c, t); };      // (none: the trajectory from the hips)
                    Mc.m_Turn = [pPkg = &Pkg, c](float t) { return xanimation::RootTurnAt(*pPkg, c, t); };
                    M.m_Clips.push_back(std::move(Mc));
                }
                M.m_Db = xanimation::BuildMatchingDb(SD.view(), SD.m_Warp, SD.m_Rest, M.m_Clips);
                std::printf("[AnimMatch] database: %zu entries of %d floats from %d clips; travel directions (deg@m/s):", M.m_Db.m_Entries.size(), M.m_Db.m_Dim, int(Pkg.m_nClips));
                for (std::size_t i = 0; i < M.m_Db.m_TravelYaws.size(); ++i) std::printf(" %.0f@%.2f", M.m_Db.m_TravelYaws[i] * 57.29578f, M.m_Db.m_TravelSpeeds[i]);
                std::printf("\n");
                for (int c = 0; c < int(Pkg.m_nClips); ++c) { const auto& C = Pkg.getClips()[std::size_t(c)]; std::printf("[AnimMatch]   clip %d: %.2f m/s at %.0f deg, turning %.0f deg/s, %.2f s%s\n", c, C.m_Speed, C.m_Direction, C.m_TurnRate, float(C.m_nFrames) / float(std::max<int>(1, C.m_FPS)), C.m_bLoop ? " loop" : ""); }
            }
            if (M.m_Db.m_Entries.empty()) return;
            const auto* pGame = xlioncore::game::From(getGameMgr());
            const float Dt = pGame ? pGame->m_Time.m_DeltaTime * std::max(0.0f, A.m_Speed) : 0.0f;
            const float Yaw0 = xanimation::details::Yaw(M.m_Db.m_RestForward);
            // the wish is the speed asked, straight ahead: the direction is orientation warping's (up to 90 deg) - our root motion carries no
            // turn yet, so a turning clip cannot turn the entity
            // FaceMove: the whole move in the character's frame and a facing asked toward it (the turn clips win; their root turn turns it)
            xmath::fvec3 Wish(0, 0, std::hypot(A.m_Move.m_X, A.m_Move.m_Z));
            float YawAsked = 0;
            if (A.m_bFaceMove && Wish.m_Z > 1e-3f)
            {
                const auto M = xanimation::details::TurnY(A.m_Move, -Yaw0); Wish = xmath::fvec3(M.m_X, 0, M.m_Z);
                YawAsked = std::atan2(M.m_X, M.m_Z);
            }
            if (pT && (A.m_Face.m_X != 0 || A.m_Face.m_Z != 0))     // Face wins: the move as it is in the character's frame, the facing asked
            {
                const auto M = xanimation::details::TurnY(A.m_Move, -Yaw0); Wish = xmath::fvec3(M.m_X, 0, M.m_Z);
                const auto F = xanimation::details::TurnY(pT->m_Rotation.ConjugateCopy() * A.m_Face, -Yaw0);
                YawAsked = std::atan2(F.m_X, F.m_Z);
            }
            Wish = xanimation::ReachableWish(M.m_Db, Wish);        // (toward a direction the clips walk; warping turns them the rest)
            xanimation::matching_state St; St.m_Clip = A.m_Clip; St.m_Time = A.m_Time; St.m_Since = A.m_MatchSince + Dt; St.m_SinceJump = A.m_MatchSinceJump + Dt; St.m_Jumps = int(A.m_MatchJumps);
            xanimation::matching_params Pm;
            if (m_Searches >= SearchBudget()) { Pm.m_Interval = 1e9f; ++m_SearchWaits; }       // (over the budget: no search unless the clip is ending)
            const bool bJumped = xanimation::UpdateMatching(M.m_Db, M.m_Clips, St, xanimation::PredictWish(Wish, Wish, YawAsked), 0, Pm);
            if (St.m_Since == 0) ++m_Searches;
            if (bJumped)
            {
                static const bool bLog = [] { const char* e = std::getenv("XLION_ANIM_EVENTS"); return e && *e == '1'; }();
                if (bLog) std::printf("[AnimMatch] jump %d: clip %d at %.2f s -> clip %d at %.2f s\n", St.m_Jumps, A.m_Clip, A.m_Time, St.m_Clip, St.m_Time);
                A.m_Clip = St.m_Clip; A.m_Time = St.m_Time; A.m_bPlaying = true;
            }
            A.m_MatchSince = St.m_Since; A.m_MatchSinceJump = St.m_SinceJump; A.m_MatchJumps = std::uint32_t(St.m_Jumps);
            static const bool bDbg = std::getenv("XLION_ANIM_MATCH_DEBUG") != nullptr;      // (each search: the wish and every clip's best cost)
            if (bDbg && St.m_Since == 0)
            {
                std::vector<float> Q(std::size_t(M.m_Db.m_Dim));
                xanimation::MakeQuery(M.m_Db, xanimation::PlayingEntry(M.m_Db, St), xanimation::PredictWish(Wish, Wish, YawAsked), Q.data());
                std::string S; char B[64];
                for (int c = 0; c + 1 < int(M.m_Db.m_ClipFirst.size()); ++c)
                {
                    float Bc = 1e30f, Bp = 0, Bt = 0;      // (the best entry's cost: pose part + trajectory part)
                    for (int i = M.m_Db.m_ClipFirst[std::size_t(c)]; i < M.m_Db.m_ClipFirst[std::size_t(c) + 1]; ++i)
                    {
                        float Cp = 0, Ct = 0;
                        for (int d = 0; d < M.m_Db.m_Dim; ++d) { const float x = M.m_Db.At(i)[d] - Q[std::size_t(d)]; (d < M.m_Db.TrajPos() ? Cp : Ct) += x * x; }
                        if (Cp + Ct < Bc) { Bc = Cp + Ct; Bp = Cp; Bt = Ct; }
                    }
                    std::snprintf(B, sizeof(B), " %d:%.1f+%.1f", c, Bp, Bt); S += B;
                }
                std::printf("[AnimMatchDbg] at (%.1f %.1f) clip %d t %.2f wish (%.2f %.2f) yaw %.0f costs (pose+trajectory)%s\n", pT ? pT->m_Position.m_X : 0.0f, pT ? pT->m_Position.m_Z : 0.0f, A.m_Clip, A.m_Time, Wish.m_X, Wish.m_Z, YawAsked * 57.29578f, S.c_str());
            }
        }

        // Contact events push (a kick): for each Contact event with a bone fired this frame, the bone's world position and velocity (the clip
        // sampled now and 1/60 s before, in the entity's frame) push the dynamic bodies within 15 cm (LIONCore PushBodies) at 1.1x its speed.
        // ponytail: the entity's own motion is not added to the bone's (a kick from standing); the entity's Transform as its world (a root entity)
        void PushOnContact(animation& A, const skinned_mesh& Skin, const xlioncore::transform& T, const xanim_package::anim_package& Pkg, float Length) noexcept
        {
            using namespace xlioncore::resources;
            auto* pGeom = Skin.m_Geom.empty() ? nullptr : static_cast<xgeom_skin::xgpu::geom*>(GetResource(geom_skin_type_guid_v.m_Value, Skin.m_Geom.m_Instance.m_Value));
            auto* pSkel = pGeom ? static_cast<xskeleton::skeleton*>(GetResource(skeleton_type_guid_v.m_Value, pGeom->getSkeletonRef().m_Instance.m_Value)) : nullptr;
            if (!pSkel || Pkg.m_nBones != pSkel->getBones().size()) return;
            auto& SD = m_Skeletons[pSkel]; if (SD.m_Parent.size() != pSkel->getBones().size()) SD = xanimation::MakeSkeleton(*pSkel);
            thread_local xanimation::pose P0, P1, Scratch; thread_local xanimation::details::rigid_worlds W0, W1;
            const float Dt = 1.0f / 60.0f;
            auto Wrap = [&](float t) { return A.m_bLoop ? t - Length * std::floor(t / Length) : std::clamp(t, 0.0f, Length); };
            static const bool bLog = [] { const char* e = std::getenv("XLION_ANIM_EVENTS"); return e && *e == '1'; }();
            for (int i = 0; i < A.m_nFired; ++i)
            {
                const auto& F = A.m_Fired[std::size_t(i)];
                if (F.m_Kind != std::uint8_t(xanim_package::event_kind::CONTACT) || F.m_Bone < 0 || F.m_Bone >= int(SD.m_Parent.size())) continue;
                if (!xanimation::SampleClip(SD, Pkg, A.m_Clip, Wrap(F.m_Time), A.m_bMirror, P1, Scratch) || !xanimation::SampleClip(SD, Pkg, A.m_Clip, Wrap(F.m_Time - Dt), A.m_bMirror, P0, Scratch)) continue;
                xanimation::details::Worlds(SD.view(), P1, W1); xanimation::details::Worlds(SD.view(), P0, W0);
                const auto& Sc = T.m_Scale;
                auto ToWorld = [&](const xmath::fvec3& M) { return T.m_Position + T.m_Rotation * xmath::fvec3(M.m_X * Sc.m_X, M.m_Y * Sc.m_Y, M.m_Z * Sc.m_Z); };
                const auto P = ToWorld(W1.m_P[std::size_t(F.m_Bone)]), V = (P - ToWorld(W0.m_P[std::size_t(F.m_Bone)])) * (1.0f / Dt);
                xlioncore::physics::pushed_body Got[4];
                const int n = xlioncore::physics::PushBodies(getGameMgr(), P, 0.15f * Sc.m_X, V * 1.1f, Got, 4);
                for (int k = 0; k < std::min(n, 4) && A.m_bResetPushed; ++k)       // (a drill remembers where they were, the first time)
                    if (std::none_of(A.m_Pushed.begin(), A.m_Pushed.begin() + A.m_nPushed, [&](const auto& B) { return B.m_Entity == Got[k].m_Entity; }) && A.m_nPushed < A.m_Pushed.size())
                        A.m_Pushed[A.m_nPushed++] = Got[k];
                if (bLog) std::printf("[AnimPush] contact at %.2f s: bone %d at (%.2f %.2f %.2f) moving %.2f units/s pushed %d bodies\n", F.m_Time, F.m_Bone, P.m_X, P.m_Y, P.m_Z, V.Length(), n);
            }
        }

        // The ground under each foot (K3): the pose this frame (the clip at its time, mirrored and warped as the renderer will draw it), a ray
        // down at each foot from half the hips' height above it (Box3D, the game's physics world), the hit's height and normal in the
        // character's frame, eased in over ~0.05 s (a foot crossing a step's edge does not pop). The renderer fits the legs to it (FitGround).
        // ponytail: the entity's Transform is taken as its world (a root entity); a child entity's parents are not applied
        void GroundFeet(animation& A, const skinned_mesh& Skin, const xlioncore::transform& T, const xanim_package::anim_package& Pkg, float Length) noexcept
        {
            using namespace xlioncore::resources;
            auto* pGeom = Skin.m_Geom.empty() ? nullptr : static_cast<xgeom_skin::xgpu::geom*>(GetResource(geom_skin_type_guid_v.m_Value, Skin.m_Geom.m_Instance.m_Value));
            auto* pSkel = pGeom ? static_cast<xskeleton::skeleton*>(GetResource(skeleton_type_guid_v.m_Value, pGeom->getSkeletonRef().m_Instance.m_Value)) : nullptr;
            if (!pSkel || Pkg.m_nBones != pSkel->getBones().size()) { A.m_nGround = 0; return; }
            auto& SD = m_Skeletons[pSkel]; if (SD.m_Parent.size() != pSkel->getBones().size()) SD = xanimation::MakeSkeleton(*pSkel);
            const auto& Rig = SD.m_Warp;
            if (!Rig.valid() || Rig.m_Legs.size() > A.m_Ground.size()) { A.m_nGround = 0; return; }
            thread_local xanimation::pose P, Scratch;
            const float t = A.m_bLoop ? A.m_Time - Length * std::floor(A.m_Time / Length) : std::clamp(A.m_Time, 0.0f, Length);
            if (!xanimation::SampleClip(SD, Pkg, A.m_Clip, t, A.m_bMirror, P, Scratch)) { A.m_nGround = 0; return; }
            xanimation::warp_params Wp; Wp.m_Stride = A.m_WarpStride; Wp.m_Angle = A.m_WarpAngle; Wp.m_Dir = A.m_WarpDir;
            xanimation::WarpPose(SD.view(), Rig, P, Wp);
            std::array<xanimation::ground_contact, 4> Now;
            xanimation::details::rigid_worlds W; xanimation::details::Worlds(SD.view(), P, W);
            xanimation::CastFeet(SD.view(), Rig, P, T.m_Position, T.m_Rotation, T.m_Scale, 0.5f * W.m_P[Rig.m_Hips].m_Y * T.m_Scale.m_Y
                , [&](const xmath::fvec3& O, const xmath::fvec3& D, xmath::fvec3& Hit, xmath::fvec3& N)
                  { const auto H = xlioncore::physics::CastRayClosest(getGameMgr(), O, D); Hit = H.m_Point; N = H.m_Normal; return H.m_bHit; }
                , std::span<xanimation::ground_contact>(Now.data(), Rig.m_Legs.size()));
            const auto* pGame = xlioncore::game::From(getGameMgr());
            const float Dt = pGame ? pGame->m_Time.m_DeltaTime : 0.0f, Ease = A.m_nGround ? 1.0f - std::exp(-Dt / 0.05f) : 1.0f;
            for (std::size_t i = 0; i < Rig.m_Legs.size(); ++i)        // eased in (a foot crossing a step's edge does not pop)
            {
                auto& G = A.m_Ground[i]; const auto& C = Now[i];
                if (!C.m_bHit) { G.m_bHit = false; continue; }
                if (!G.m_bHit) { G.m_Height = C.m_Height; G.m_Normal = C.m_Normal; }
                else { G.m_Height += (C.m_Height - G.m_Height) * Ease; G.m_Normal = (G.m_Normal + (C.m_Normal - G.m_Normal) * Ease).NormalizeSafeCopy(); }
                G.m_bHit = true;
            }
            A.m_nGround = std::uint8_t(Rig.m_Legs.size());
            static const bool bLog = [] { const char* e = std::getenv("XLION_ANIM_EVENTS"); return e && *e == '1'; }();
            if (thread_local int n = 0; bLog && (n++ % 30) == 0)
            {
                std::printf("[AnimGround] legs %u:", unsigned(A.m_nGround));
                for (std::size_t i = 0; i < A.m_nGround; ++i) std::printf(" %s %.3f (n %.2f %.2f %.2f)", A.m_Ground[i].m_bHit ? "hit" : "none", A.m_Ground[i].m_Height, A.m_Ground[i].m_Normal.m_X, A.m_Ground[i].m_Normal.m_Y, A.m_Ground[i].m_Normal.m_Z);
                std::printf("\n");
            }
        }

        // The clip's events (T5) crossed by this frame's time step (Before, after], once each (xanimation::Advance: a loop fires once per pass,
        // backwards fires nothing): listed on the Animation for this frame (m_Fired: a game system reads them after this one - a footstep
        // sound at the foot), counted (m_nEvents, m_LastEvent) and, with XLION_ANIM_EVENTS=1, logged ([AnimEvent]). Mirrored: the mirror bone.
        void FireEvents(const xanim_package::anim_package& Pkg, animation& A, float Before, float Length, const xanimation::mirror_table* pMirror = nullptr) noexcept
        {
            const auto Ev = Pkg.getClipEvents(A.m_Clip);
            if (Ev.empty()) return;
            thread_local std::vector<xanimation::event_def> Defs; thread_local std::vector<xanimation::event_fire> Fired;
            Defs.clear(); Fired.clear();
            for (std::uint32_t i = 0; i < Ev.size(); ++i) Defs.push_back({ i, Ev[i].m_Time });
            xanimation::event_cursor C; C.Seek(Before);
            xanimation::Advance(C, xanimation::event_track{ Defs, Length, A.m_bLoop }, A.m_Time, Fired);
            static const bool bLog = [] { const char* e = std::getenv("XLION_ANIM_EVENTS"); return e && *e == '1'; }();
            for (const auto& F : Fired)
            {
                const auto& E = Ev[F.m_Id];
                ++A.m_nEvents; A.m_LastEvent = E.m_NameHash;
                const std::int16_t Bone = (pMirror && E.m_iBone >= 0 && std::size_t(E.m_iBone) < pMirror->m_Bones.size()) ? std::int16_t(pMirror->m_Bones[std::size_t(E.m_iBone)].m_Src) : E.m_iBone;
                if (A.m_nFired < A.m_Fired.size()) A.m_Fired[A.m_nFired++] = { E.m_NameHash, std::uint8_t(E.m_Kind), Bone, E.m_Time, E.m_Duration };
                if (bLog) std::printf("[AnimEvent] clip %d at %.3f s: kind %d name %08X bone %d%s\n", A.m_Clip, E.m_Time, int(E.m_Kind), E.m_NameHash, Bone, pMirror ? " (mirrored: the mirror bone)" : "");
            }
        }
    };

    // The Ragdoll system (ragdoll design v2 section 8, R4 in Play): an Active Ragdoll gets its bodies at the animated pose of this frame, with
    // the motion of the last fixed step (the clip sampled one fixed step back); the bodies live in the game's physics world (LIONCore owns
    // Box3D) and are captured after every step; the renderer poses the skeleton from them (scene_renderer::AddSkinnedMesh). The entity follows
    // the root body (the hips) across the ground (x, z; its height and rotation stay). Inactive: the bodies go, the Animation poses it again.
    // Log: [Ragdoll] lines on activation and a summary 4 s after (the R4 gate's numbers: anchors, energy, time to rest)
    struct ragdoll_system : xecs::system::instance
    {
        constexpr static auto typedef_v = xecs::system::type::update{ .m_pName = "Ragdoll" };
        ragdoll_system(xecs::game_mgr::instance& GameMgr) noexcept : xecs::system::instance(GameMgr) {}

        struct graph_entry { xragdoll::runtime_graph m_G; std::vector<xanimation::rigid> m_Rest; };
        struct entity_state
        {
            xmath::fvec3        m_Offset;               // entity - hips (x, z) at the switch
            float               m_T = 0, m_KE0 = 0, m_Y0 = 0, m_Mass = 0, m_WorstEnergy = 0, m_RestAt = -1, m_Sleep = 0;
            std::vector<float>  m_Anchor;
            bool                m_bReported = false;
        };
        std::unordered_map<const void*, graph_entry>                m_Graphs;       // per Ragdoll resource as loaded
        std::unordered_map<const void*, xanimation::skeleton_data>  m_Skeletons;
        std::unordered_map<std::uint64_t, entity_state>             m_State;
        xanimation::pose                                            m_P0, m_P1;

        void operator()( const xecs::component::entity& Ent, ragdoll& R, const skinned_mesh& Skin, const animation* pAnim, xlioncore::transform& T, const xecs::component::parent* pParent ) noexcept
        {
            auto& GameMgr = getGameMgr();
            const auto* pGame = xlioncore::game::From(GameMgr);
            if (pGame == nullptr) return;
            if (!R.m_bActive)
            {
                if (R.m_pRuntime) { xlioncore::physics::DeactivateRagdoll(GameMgr, R.m_pRuntime); R.m_pRuntime = nullptr; m_State.erase(Ent.m_Value); }
                return;
            }
            if (R.m_pRuntime == nullptr && !Activate(Ent, R, Skin, pAnim, T, pParent, *pGame)) { R.m_bActive = false; return; }

            // the entity follows the hips across the ground
            auto& St = m_State[Ent.m_Value];
            const auto Hips = xlioncore::physics::RagdollRootBody(*R.m_pRuntime, 1.0f);
            if (pParent == nullptr) { T.m_Position.m_X = Hips.m_P.m_X + St.m_Offset.m_X; T.m_Position.m_Z = Hips.m_P.m_Z + St.m_Offset.m_Z; }

            // the gate's numbers over the first 4 s
            if (St.m_bReported) return;
            St.m_T += pGame->m_Time.m_DeltaTime;
            xlioncore::physics::ragdoll_stats S; xlioncore::physics::RagdollStats(*R.m_pRuntime, S);
            St.m_Anchor.push_back(S.m_MaxAnchor);
            // no energy made: KE <= (at the switch + the drop's m g h) + 10 % + 1 J (test_ragdoll_runtime.cpp's bound: the 1 J keeps a still start's
            // millijoules from reading as a ratio)
            const float Bound = 1.1f * (St.m_KE0 + St.m_Mass * 9.81f * (St.m_Y0 - S.m_ComY)) + 1.0f;
            if (S.m_KE > Bound && St.m_WorstEnergy <= 1.0f)      // the moment it first makes energy: when, how much, which body
                std::printf("[Ragdoll] entity 0x%llX makes energy at %.3f s: KE %.1f J > bound %.1f J, fastest body %d at %.2f m/s, anchors %.1f mm\n"
                    , (unsigned long long)Ent.m_Value, St.m_T, S.m_KE, Bound, S.m_iFastest, S.m_MaxSpeed, 1000 * S.m_MaxAnchor), std::fflush(stdout);
            St.m_WorstEnergy = std::max(St.m_WorstEnergy, S.m_KE / Bound);
            const bool bRest = S.m_nAwake == 0 || S.m_MaxSpeed < 2 * St.m_Sleep;
            if (bRest && St.m_RestAt < 0) St.m_RestAt = St.m_T;
            if (!bRest && S.m_MaxSpeed > 4 * St.m_Sleep) St.m_RestAt = -1;
            if (St.m_T >= 4.0f)
            {
                auto A = St.m_Anchor; std::sort(A.begin(), A.end());
                std::printf("[Ragdoll] summary entity 0x%llX: anchors p95 %.2f mm max %.2f mm (%zu frames); energy <= %.2f of the bound (KE0 + m g h + 10 %% + 1 J); at rest %s%.2f s; awake %d/%d\n"
                    , (unsigned long long)Ent.m_Value, 1000 * A[std::min(A.size() - 1, A.size() * 95 / 100)], 1000 * A.back(), A.size(), St.m_WorstEnergy
                    , St.m_RestAt < 0 ? "NOT " : "", St.m_RestAt < 0 ? St.m_T : St.m_RestAt, S.m_nAwake, S.m_nBodies);
                std::fflush(stdout);
                St.m_bReported = true;
            }
        }

        bool Activate( const xecs::component::entity& Ent, ragdoll& R, const skinned_mesh& Skin, const animation* pAnim, const xlioncore::transform& T, const xecs::component::parent* pParent, const xlioncore::game& Game ) noexcept
        {
            using namespace xlioncore::resources;
            auto Fail = [&](const char* Why) { std::printf("[Ragdoll] entity 0x%llX not activated: %s\n", (unsigned long long)Ent.m_Value, Why); std::fflush(stdout); return false; };
            if (pParent) return Fail("it has a parent (a ragdoll is a root entity)");
            auto* pRag  = R.m_Ragdoll.empty() ? nullptr : static_cast<const xragdoll::ragdoll*>(GetResource(ragdoll_type_guid_v.m_Value, R.m_Ragdoll.m_Instance.m_Value));
            auto* pGeom = Skin.m_Geom.empty() ? nullptr : static_cast<xgeom_skin::xgpu::geom*>(GetResource(geom_skin_type_guid_v.m_Value, Skin.m_Geom.m_Instance.m_Value));
            auto* pSkel = pGeom ? static_cast<const xskeleton::skeleton*>(GetResource(skeleton_type_guid_v.m_Value, pGeom->getSkeletonRef().m_Instance.m_Value)) : nullptr;
            if (pRag == nullptr || pSkel == nullptr) return Fail("its Ragdoll or its Skinned Mesh's skeleton is not loaded");
            const int nB = int(pSkel->getBones().size());
            if (int(pRag->m_nBones) != nB) return Fail("the Ragdoll was made for another skeleton (bone count)");

            auto& S = m_Skeletons[pSkel];
            if (int(S.m_Parent.size()) != nB) S = xanimation::MakeSkeleton(*pSkel);
            const xanimation::skeleton_view View = S.view();
            auto& GE = m_Graphs[pRag];
            if (GE.m_Rest.size() != std::size_t(nB))
            {
                std::vector<float> Sc(static_cast<std::size_t>(nB));
                GE.m_Rest.resize(std::size_t(nB)); xanimation::RigidBoneWorlds(View, S.m_Rest, {}, 1.0f, GE.m_Rest, Sc);
                GE.m_G = xragdoll::ToBodyGraph(*pRag, GE.m_Rest);
                std::printf("[Ragdoll] %u bodies, %u joints, %.1f kg; joint anchors at the skeleton's rest %.2f mm apart (0: the rest it was compiled with)\n"
                    , pRag->m_nBodies, pRag->m_nJoints, pRag->m_TotalMass, 1000 * GE.m_G.m_RestAnchorGap);
            }
            if (!GE.m_G.m_Graph.m_bOk) return Fail("its joints name bodies it does not have");

            // the animated pose now and one fixed step back (the motion it carries into the bodies)
            const float Dt = Game.m_Time.m_FixedDeltaTime;
            auto Sample = [&](float Dtb, xanimation::pose& P)
            {
                const std::uint64_t Pkg = pAnim == nullptr ? 0 : pAnim->m_Package.empty() ? pGeom->getAnimationRef().m_Instance.m_Value : pAnim->m_Package.m_Instance.m_Value;
                auto* pPkg = Pkg ? static_cast<xanim_package::anim_package*>(GetResource(anim_package_type_guid_v.m_Value, Pkg)) : nullptr;
                if (pPkg && pAnim->m_Clip >= 0 && pAnim->m_Clip < int(pPkg->getClips().size()) && int(pPkg->getBoneNameHashes().size()) == nB)
                {
                    const auto& C = pPkg->getClips()[std::size_t(pAnim->m_Clip)];
                    const float Len = C.m_FPS > 0 ? float(C.m_nFrames) / float(C.m_FPS) : 0.0f;
                    float t = pAnim->m_Time - (pAnim->m_bPlaying ? Dtb * pAnim->m_Speed : 0.0f);
                    if (Len > 0) t = pAnim->m_bLoop ? t - Len * std::floor(t / Len) : std::clamp(t, 0.0f, Len);
                    xanimation::pose Scratch;
                    if (xanimation::SampleClip(S, *pPkg, pAnim->m_Clip, t, pAnim->m_bMirror, P, Scratch)) return;
                }
                P = S.m_Rest;
            };
            Sample(Dt, m_P0); Sample(0, m_P1);
            const xanimation::rigid E{ T.m_Position, T.m_Rotation };
            const float Scale = T.m_Scale.m_X;
            const auto n = static_cast<std::size_t>(nB);
            std::vector<xanimation::rigid> W0(n), W1(n); std::vector<float> Sc(n);
            xanimation::RigidBoneWorlds(View, m_P0, E, Scale, W0, Sc);
            xanimation::RigidBoneWorlds(View, m_P1, E, Scale, W1, Sc);

            xlioncore::physics::ragdoll_params Params;
            Params.m_Size = 2 * (pRag->m_Capsule.m_HalfSegment + pRag->m_Capsule.m_Radius) * Scale;
            if (Params.m_Size <= 0.01f) Params.m_Size = 1.7f;
            std::string Note;
            R.m_pRuntime = xlioncore::physics::ActivateRagdoll(getGameMgr(), GE.m_G.m_Graph, GE.m_Rest, W0, W1, Dt, GE.m_G.m_Filter, Params, &Note);
            if (R.m_pRuntime == nullptr) return Fail("the game has no physics world");

            auto& St = m_State[Ent.m_Value]; St = {};
            const auto Hips = xlioncore::physics::RagdollRootBody(*R.m_pRuntime, 1.0f);
            St.m_Offset = T.m_Position - Hips.m_P;
            St.m_Sleep = 0.05f * std::sqrt(Params.m_Size / 1.7f);
            xlioncore::physics::ragdoll_stats Stats; xlioncore::physics::RagdollStats(*R.m_pRuntime, Stats);
            St.m_KE0 = Stats.m_KE; St.m_Y0 = Stats.m_ComY; St.m_Mass = Stats.m_Mass;
            float Low = 1e30f; for (const auto& W : W1) Low = std::min(Low, W.m_P.m_Y);
            std::printf("[Ragdoll] entity 0x%llX activated: %d bodies, %.1f kg, size %.2f m, KE %.2f J, COM speed max %.2f m/s, COM y %.3f, lowest bone y %.3f, anim time %.3f%s%s\n"
                , (unsigned long long)Ent.m_Value, Stats.m_nBodies, Stats.m_Mass, Params.m_Size, Stats.m_KE, Stats.m_MaxSpeed, Stats.m_ComY, Low, pAnim ? pAnim->m_Time : 0.0f, Note.empty() ? "" : "; ", Note.c_str());
            std::fflush(stdout);
            return true;
        }
    };

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
        std::unique_ptr<scene_renderer>                 m_pScene;       // the forward renderer of this world (Meshes, lights, Environment), made the first time it is asked to render
        view_options                                    m_ViewOptions;  // the editor's viewing choices (xRenderEditor::SetViewOptions): on top of the Environment, never saved

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

        // What a view leaves out of the world: what the game says is not drawn (no_render, in every view) and, in the editor's scene view, what the editor hid (editor_no_render). An entity with
        // an exclusive disable tag (the game's or the editor's) is not in any query to begin with.
        static void LeaveOut(xecs::query::instance& Query, view View) noexcept
        {
            Query.m_NoneOf.AddFromComponents<xlioncore::no_render_tag>();
            if (View == view::SCENE) Query.m_NoneOf.AddFromComponents<xecs::editor::no_render_tag>();
        }

        // The Texts of the world: their layouts are kept up to date (the drawing of them comes next to this). The resource system of the core starts here, the first time there is a device.
        void CollectText(view View, const roles* pRoles = nullptr) noexcept
        {
            ++m_Frame;
            if (auto* pDevice = GetDeviceInternal())
                xlioncore::resources::Initialize(*pDevice, getGameMgr().m_SceneMgr.m_ProjectPath.c_str());

            xecs::query::instance Query;
            Query.m_Must.AddFromComponents<xlioncore::transform, text>();
            LeaveOut(Query, View);
            auto S = Search(Query);
            int nSeen = 0, nWithLayout = 0;
            Foreach(S, [&](const xecs::component::entity& Ent, const xlioncore::transform& Local, const text& Tx, const xecs::component::parent* pParent, const xlioncore::render_transform* pRender) noexcept
            {
                if (pRoles && (pRoles->IsHidden(Ent.m_Value) || pRoles->IsContext(Ent.m_Value))) return;       // the Texts of the context are not drawn (the fade quad is for solids), nor those of the hidden instance
                const auto T = xlioncore::WorldOf(Local, pParent, pRender, xlioncore::FixedInterpolateOf(getGameMgr()));   // a child's Transform is relative to its parent: what is drawn is at its world pose
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
            std::erase_if(m_Texts, [&](const auto& Pair) noexcept { return m_Frame - Pair.second.m_Frame > 3; });       // (not "this one": two views collect in turn, and each sees entities the other does not)
        }

        // pRoles (the editor's session, null: every entity is the document): the hidden entities are not submitted; the context ones are, and the renderer draws them first, faded (xlionrender::roles).
        void Collect(view View = view::SCENE, const roles* pRoles = nullptr) noexcept
        {
            xlioncore::PropagateHierarchy(*this);       // the world pose of every child, from the Transforms as they are now (this is after everything that moves things)
            CollectText(View, pRoles);
            xecs::query::instance Query;
            Query.m_Must.AddFromComponents<xlioncore::transform, primitive>();
            LeaveOut(Query, View);
            auto S = Search(Query);

            Foreach(S, [&](const xecs::component::entity& Ent, const xlioncore::transform& Local, const primitive& Prim, const xecs::component::parent* pParent, const xlioncore::render_transform* pRender) noexcept
            {
                if (pRoles && pRoles->IsHidden(Ent.m_Value)) return;
                const auto T = xlioncore::WorldOf(Local, pParent, pRender, xlioncore::FixedInterpolateOf(getGameMgr()));   // a child's Transform is relative to its parent: what is drawn is at its world pose
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

        // The world's Meshes, lights and Environment as this frame's scene, rendered into the world's view (xlionrender_scene_renderer.h); the host
        // blits it (BlitScene). Called while the frame is built, before the UI is rendered: it records render passes.
        void RenderScene(xgpu::window& Window, xgpu::tools::view& Camera, int W, int H, view View, scene_renderer::overlay_fn pOverlay, void* pUser, const roles* pRoles) noexcept
        {
            auto* pDevice = GetDeviceInternal();
            if (!pDevice) return;
            xlioncore::resources::Initialize(*pDevice, getGameMgr().m_SceneMgr.m_ProjectPath.c_str());
            XPROFILE_MEM_OWNER("LIONRender: a world's scene renderer");        // (the profiler's memory map: its views, GI, meshes, shadows...)
            if (!m_pScene) m_pScene = std::make_unique<scene_renderer>();
            auto& R = *m_pScene;
            if (!R.Init(*pDevice)) return;
            { auto* pAnimSys = findSystem<animation_system>(); R.m_pUpdated = pAnimSys ? &pAnimSys->m_Poser : nullptr; }     // (K0c: the poses the update made)

            xlioncore::PropagateHierarchy(*this);
            R.Begin();
            R.m_Scene.m_SelectedUser = GetSelectedInternal();
            const auto Interpolate = xlioncore::FixedInterpolateOf(getGameMgr());
            const auto L2WOf = [&](const xlioncore::transform& Local, const xecs::component::parent* pParent, const xlioncore::render_transform* pRender, xmath::fmat4& L2W) noexcept
            {
                const auto T = xlioncore::WorldOf(Local, pParent, pRender, Interpolate);
                if (!T.m_Scale.isFinite() || T.m_Scale.m_X == 0.0f || T.m_Scale.m_Y == 0.0f || T.m_Scale.m_Z == 0.0f) return false;    // a degenerate scale: not drawn (as the Primitive)
                L2W.setupSRT(T.m_Scale, T.m_Rotation, T.m_Position);
                return true;
            };
            const auto Hidden = [&](std::uint64_t Entity) noexcept { return pRoles && pRoles->IsHidden(Entity); };
            const auto Each = [&]<typename T_COMPONENT>(auto&& Fn)
            {
                xecs::query::instance Query;
                Query.m_Must.AddFromComponents<xlioncore::transform, T_COMPONENT>();
                LeaveOut(Query, View);
                auto S = Search(Query);
                Foreach(S, [&](const xecs::component::entity& Ent, const xlioncore::transform& Local, const T_COMPONENT& C, const xecs::component::parent* pParent, const xlioncore::render_transform* pRender) noexcept
                {
                    xmath::fmat4 L2W;
                    if (Hidden(Ent.m_Value) || !L2WOf(Local, pParent, pRender, L2W)) return;
                    Fn(Ent.m_Value, C, L2W);
                });
            };
            {   // first: the meshes need to know whether GI is on (their surfels are baked when it is)
                xecs::query::instance Query;
                Query.m_Must.AddFromComponents<environment>();
                auto S = Search(Query);
                bool bFirst = true;
                Foreach(S, [&](const environment& E) noexcept { if (bFirst) R.SetEnvironment(E); bFirst = false; });
                {   // the SkyDome (one per level: the first): the background - after the Environment, whose cube map it defaults to
                    xecs::query::instance QDome;
                    QDome.m_Must.AddFromComponents<sky_dome>();
                    auto SD = Search(QDome);
                    bool bDome = true;
                    Foreach(SD, [&](const sky_dome& D) noexcept { if (bDome) R.SetSkyDome(D); bDome = false; });
                }
                auto& G = R.m_Scene.m_GI;                                   // the editor's viewing choices on top (view_options)
                if (m_ViewOptions.m_GIDebug >= 0) G.m_Debug = static_cast<gi_debug>(m_ViewOptions.m_GIDebug);
                if (m_ViewOptions.m_bNoGI)        G.m_bEnable   = false;
                if (m_ViewOptions.m_bNoPerPixel)  G.m_bPerPixel = false;
                if (m_ViewOptions.m_bNoMoments)   G.m_bMoments  = false;
                if (m_ViewOptions.m_bPathTraced)  G.m_bPathTraced = true;
                if (m_ViewOptions.m_GITraceAllowanceMs >= 0) G.m_TraceAllowanceMs = m_ViewOptions.m_GITraceAllowanceMs;
            }
            R.m_Renderer.m_bBoxesOnly    = m_ViewOptions.m_bBoxesOnly;         // huge meshes: their boxes, or the clusters a frame budget allows
            R.m_Renderer.m_FrameBudgetMs = m_ViewOptions.m_FrameBudgetMs;
            R.m_Renderer.m_bGPUCull      = !m_ViewOptions.m_bNoGPUCull;        // GPU cluster culling: on, its occlusion, frozen
            R.m_Renderer.m_bGPUOcclusion = !m_ViewOptions.m_bNoGPUOcclusion;
            R.m_Renderer.m_bFreezeCull   = m_ViewOptions.m_bFreezeCull;
            {   // the Meshes, with their MeshMaterial when they have one (old files: OldMaterials lists them for the editor to convert)
                xecs::query::instance Query;
                Query.m_Must.AddFromComponents<xlioncore::transform, mesh>();
                LeaveOut(Query, View);
                auto S = Search(Query);
                m_OldMaterials.clear();
                Foreach(S, [&](const xecs::component::entity& Ent, const xlioncore::transform& Local, const mesh& C, const mesh_material* pMaterial, const xecs::component::parent* pParent, const xlioncore::render_transform* pRender) noexcept
                {
                    if (HasOldMaterials(C, pMaterial)) m_OldMaterials.push_back(Ent.m_Value);
                    xmath::fmat4 L2W;
                    if (Hidden(Ent.m_Value) || !L2WOf(Local, pParent, pRender, L2W)) return;
                    R.AddMesh(C, pMaterial, L2W, Ent.m_Value);
                });
            }
            {   // the Skinned Meshes, posed by their Animation when they have one
                xecs::query::instance Query;
                Query.m_Must.AddFromComponents<xlioncore::transform, skinned_mesh>();
                LeaveOut(Query, View);
                auto S = Search(Query);
                const float Alpha = [&] { const auto* pG = xlioncore::game::From(getGameMgr()); return pG ? pG->m_Time.m_FixedInterpolate : 1.0f; }();
                Foreach(S, [&](const xecs::component::entity& Ent, const xlioncore::transform& Local, const skinned_mesh& C, const animation* pAnim, const ragdoll* pRagdoll, const xecs::component::parent* pParent, const xlioncore::render_transform* pRender) noexcept
                {
                    xmath::fmat4 L2W;
                    if (Hidden(Ent.m_Value) || !L2WOf(Local, pParent, pRender, L2W)) return;
                    R.AddSkinnedMesh(C, pAnim, L2W, Ent.m_Value, pRagdoll && pRagdoll->m_bActive ? pRagdoll->m_pRuntime : nullptr, Alpha);
                });
                static const bool bLog = [] { const char* e = std::getenv("XLION_ANIM_EVENTS"); return e && *e == '1'; }();
                if (bLog && R.m_pUpdated && (R.m_pUpdated->m_Update % 300) == 1)      // (K0c: who posed the characters)
                    std::printf("[AnimPose] update %u: %d skinned meshes drawn, %d posed by the Animation system\n", R.m_pUpdated->m_Update, R.m_Stats.m_Meshes, R.m_Stats.m_PosedByUpdate);
            }
            Each.template operator()<light_sun>  ([&](std::uint64_t,   const light_sun& C,   const xmath::fmat4& L2W) { R.AddSun(C, L2W); });
            Each.template operator()<light_point>([&](std::uint64_t Id, const light_point& C, const xmath::fmat4& L2W) { R.AddPoint(C, L2W * xmath::fvec3(0, 0, 0), Id); });
            Each.template operator()<light_spot> ([&](std::uint64_t Id, const light_spot& C,  const xmath::fmat4& L2W) { R.AddSpot(C, L2W, Id); });
            Each.template operator()<reflection_probe_component>([&](std::uint64_t, const reflection_probe_component& C, const xmath::fmat4& L2W) { R.AddProbe(C, L2W); });
            R.Render(Window, Camera, W, H, pOverlay, pUser);
        }

        void BlitScene(xgpu::cmd_buffer& CmdBuffer) noexcept { if (m_pScene) m_pScene->Blit(CmdBuffer); }
        std::uint64_t PickPixel(int X, int Y, float* pDistance = nullptr) noexcept { if (pDistance) *pDistance = 0.0f; return m_pScene ? m_pScene->PickPixel(X, Y, pDistance) : ~0ull; }      // (the G-buffer prepass's id: xRenderEditor::PickPixel)

        // Old files' material data (a MeshMaterial, the Mesh's deprecated Materials list): the Meshes the last RenderScene found with it, and one
        // of them converted into its Mesh's own slots (UpgradeOldMaterials: false while its materials load, or when it has none)
        std::vector<std::uint64_t> m_OldMaterials;
        bool UpgradeOldMaterialsOf(std::uint64_t Entity) noexcept
        {
            bool bDone = false;
            xecs::query::instance Query;
            Query.m_Must.AddFromComponents<mesh>();
            auto S = Search(Query);
            Foreach(S, [&](const xecs::component::entity& Ent, mesh& M, mesh_material* pOld) noexcept
            {
                if (!bDone && Ent.m_Value == Entity) bDone = UpgradeOldMaterials(M, pOld);
            });
            return bDone;
        }

        // What the last RenderScene had (the editor's DescribeScene, tests)
        std::string DescribeScene() const noexcept
        {
            if (!m_pScene) return "DescribeScene: not rendered yet";
            const auto& S  = m_pScene->m_Stats;
            const auto& GI = m_pScene->m_Renderer.getGIStats();
            const auto  C  = m_pScene->m_Renderer.getCullStats();     // (the clusters kept: read back a few frames late)
            const auto  L  = m_pScene->m_Renderer.getLightStats();
            const auto& V  = m_pScene->m_Renderer.getVSMReport();
            return std::format("DescribeScene: ok\nViewW={}\nViewH={}\nMeshes={}\nDrawn={}\nWaiting={}\nConstantSlots={}\nMaterials={}\nSun={}\nLights={}\nDrawsRendered={}\nGIMeshes={}\nGIUpdated={}\nGIRays={}\nGIMs={:.2f}\nGILoadMs={:.0f}\nGILoadParts={:.0f}/{:.0f}/{:.0f}\nGIBad={}\nGICaptured={}\nGIEmissive={}\nGIAlbedo={:.3f},{:.3f},{:.3f}\nGILinks={}\nGIMaxEmitted={:.4f}\nGIMaxEmitter={:.3f},{:.3f},{:.3f}\nGICapturing={}\nGIUnsupported={}\nGITraceMs={:.2f}\nGIUploadMs={:.2f}\nBoxesOnly={}\nFrameBudgetMs={:.0f}\nClustersInView={}\nClustersDrawn={}\nBoxes={}\nIndexAllowance={}\nBudgetFrameMs={:.1f}\nGPUCullCandidates={}\nGPUCullRuns={}\nGPUCullDrawn={}\nGPUCullPrepass={}\nGPUCullPass2={}\nGPUCullFrozen={}\nGPUCullTriangles={}\nGPUCullOcclusionSkipped={}\nGPUCullOccludedShare={:.2f}\nGPUCullMs={:.3f}\nGPUCullOccludeMs={:.3f}"
                              , m_pScene->m_View.m_Width, m_pScene->m_View.m_Height, S.m_Meshes, S.m_Drawn, S.m_Waiting, S.m_ConstantSlots, S.m_Materials, S.m_bSun, S.m_Lights, m_pScene->m_Renderer.getDrawsRendered()
                              , S.m_GIMeshes, GI.m_Updated, GI.m_Rays, GI.m_Ms, m_pScene->m_TotalGILoadMs, m_pScene->m_GILoadPartMs[0], m_pScene->m_GILoadPartMs[1], m_pScene->m_GILoadPartMs[2], GI.m_nBad, GI.m_nColored, GI.m_nEmissive, GI.m_MeanAlbedo.m_X, GI.m_MeanAlbedo.m_Y, GI.m_MeanAlbedo.m_Z, GI.m_nLinks, GI.m_MaxEmitted, GI.m_MaxEmitter.m_X, GI.m_MaxEmitter.m_Y, GI.m_MaxEmitter.m_Z, int(m_pScene->m_Renderer.isGICapturingAny()), GI.m_nUnsupported, GI.m_TraceMs, GI.m_UploadMs
                              , int(m_pScene->m_Renderer.m_bBoxesOnly), m_pScene->m_Renderer.m_FrameBudgetMs, m_pScene->m_Renderer.getBudgetStats().m_nClusters, m_pScene->m_Renderer.getBudgetStats().m_nDrawn
                              , m_pScene->m_Renderer.getNotDrawn().size(), m_pScene->m_Renderer.getBudgetStats().m_Allowance, m_pScene->m_Renderer.getBudgetStats().m_FrameMs
                              , C.m_nCandidates, C.m_nRuns, C.m_nDrawn, C.m_nPrepassDrawn, C.m_nDrawnPass2, int(C.m_bFrozen), C.m_nTriangles, int(C.m_bOcclusionSkipped), C.m_OccludedShare, C.m_CullGpuMs, C.m_OccludeGpuMs)
                 + std::format("\nLightsLocal={}\nLightsDropped={}\nLightsShadowed={}\nLightsShadowFallback={}\nLightsShadowStale={}\nLightCellsOverflow={}\nLightCellsMB={:.1f}"
                              , L.m_nLights, L.m_nDropped, L.m_nShadowed, L.m_nShadowFallback, L.m_nShadowStale, L.m_nOverflowCells, L.m_CellRingMB)
                 + std::format("\nSunPages={}\nSunPagesRendered={}\nSunPagesNet={}\nSunPagesRejected={}\nSunPagesResident={}\nSunPoolLoad={:.2f}\nSunDynamicBias={:.2f}\nSunPixels={}\nSunPixelsOwnLevel={}\nSunPixelsCoarser={}\nSunPixelsNone={}\nSunRequestMs={:.3f}"
                              , 1, V.m_Pages.m_Rendered, V.m_Pages.m_RenderedPinned, V.m_Pages.m_Rejected, V.m_Pages.m_ResidentFine + V.m_Pages.m_ResidentPinned, V.m_Pages.m_PoolLoad, V.m_Pages.m_DynamicBias
                              , V.m_Counters[0], V.m_Counters[1], V.m_Counters[2], V.m_Counters[3], V.m_RequestGpuMs)
                 + LocalPages();
        }

        // The local lights' shadow pages (P1d): this frame's, and a ring late the request pass's pixel-light pairs by how they were shaded
        std::string LocalPages() const noexcept
        {
            const auto& P = m_pScene->m_Renderer.getLocalReport();
            return std::format("\nLocalPagesLights={}\nLocalPagesRequested={}\nLocalPagesRendered={}\nLocalPagesPinnedRendered={}\nLocalPagesRejected={}\nLocalPagesResident={}\nLocalPagesPinned={}\nLocalPagesStale={}\nLocalPixels={}\nLocalPixelsOwnLevel={}\nLocalPixelsCoarser={}\nLocalPixelsNone={}"
                              , P.m_Pages.m_Lights, P.m_Pages.m_Requested, P.m_Pages.m_Rendered, P.m_Pages.m_RenderedPinned, P.m_Pages.m_Rejected, P.m_Pages.m_Resident, P.m_Pages.m_ResidentPinned
                              , P.m_Pages.m_Stale, P.m_Counters[0], P.m_Counters[1], P.m_Counters[2], P.m_Counters[3]);
        }

        // The GI's main-thread time in the last RenderScene, all of it (the update and the upload: what the host's shared allowance is charged)
        float GITraceMs() const noexcept { return m_pScene ? float(m_pScene->m_Renderer.getGIStats().m_UpdateGIMs) : 0.0f; }

        // The GI's surfels of the last RenderScene to a CSV file (gi::world::Dump): the surfels written, -1 when there is no GI
        int DumpGI(const char* pPath) const noexcept { return m_pScene ? m_pScene->m_Renderer.DumpGI(pPath) : -1; }

        // Ray-vs-OBB: Transform Position/Rotation + half-extents 0.5*Scale (matches Collect L2W).
        // Closest hit wins. MaxT caps the ray (e.g. ground-plane hit) so empty floor clears
        // selection instead of selecting through distant AABBs.
        // The box of an entity (what Pick tests): its center, rotation and half extents in the world. False: none (a light, a text, not there)
        bool EntityBox(std::uint64_t Entity, xmath::fvec3& Center, xmath::fquat& Rotation, xmath::fvec3& Half) noexcept
        {
            bool bFound = false;
            xecs::query::instance MeshQuery;
            MeshQuery.m_Must.AddFromComponents<xlioncore::transform, mesh>();
            auto MS = Search(MeshQuery);
            Foreach(MS, [&](const xecs::component::entity& Ent, const xlioncore::transform& Local, const mesh& M, const xecs::component::parent* pParent, const xlioncore::render_transform* pRender) noexcept
            {
                if (bFound || Ent.m_Value != Entity || M.m_Geom.empty()) return;
                auto* pGeom = static_cast<xgeom_static::xgpu::geom*>(xlioncore::resources::GetResource(geom_static_type_guid_v.m_Value, M.m_Geom.m_Instance.m_Value));
                if (!pGeom) return;
                const auto T = xlioncore::WorldOf(Local, pParent, pRender, xlioncore::FixedInterpolateOf(getGameMgr()));
                if (!T.m_Scale.isFinite()) return;
                const xmath::fvec3 LocalCenter = (pGeom->m_BBox.m_Min + pGeom->m_BBox.m_Max) * 0.5f;
                Half     = ((pGeom->m_BBox.m_Max - pGeom->m_BBox.m_Min) * 0.5f * T.m_Scale).AbsCopy();
                Center   = T.m_Position + T.m_Rotation.RotateVector(LocalCenter * T.m_Scale);
                Rotation = T.m_Rotation;
                bFound   = true;
            });
            if (bFound) return true;
            xecs::query::instance Query;
            Query.m_Must.AddFromComponents<xlioncore::transform, primitive>();
            auto S = Search(Query);
            Foreach(S, [&](const xecs::component::entity& Ent, const xlioncore::transform& Local, const primitive&, const xecs::component::parent* pParent, const xlioncore::render_transform* pRender) noexcept
            {
                if (bFound || Ent.m_Value != Entity) return;
                const auto T = xlioncore::WorldOf(Local, pParent, pRender, xlioncore::FixedInterpolateOf(getGameMgr()));
                if (!T.m_Scale.isFinite()) return;
                Center = T.m_Position; Rotation = T.m_Rotation; Half = (T.m_Scale * 0.5f).AbsCopy(); bFound = true;
            });
            return bFound;
        }

        // bMeshes false: what draws no id into the G-buffer only (primitives, texts, lamps) - the click's second pick, in front of PickPixel's surface
        std::uint64_t Pick(const xmath::fvec3& Origin, const xmath::fvec3& Dir, float MaxT = std::numeric_limits<float>::max(), view View = view::SCENE, const roles* pRoles = nullptr, bool bMeshes = true) noexcept
        {
            xecs::query::instance Query;
            Query.m_Must.AddFromComponents<xlioncore::transform, primitive>();
            LeaveOut(Query, View);
            auto S = Search(Query);

            xeditor_tools::picking::closest_hit<std::uint64_t> Hit;
            // The context and the hidden are never picked: nothing of them has an id to find (editing in context, prefabs_plan.md phase 7).
            const auto NotPickable = [&](std::uint64_t Entity) noexcept { return pRoles && (pRoles->IsContext(Entity) || pRoles->IsHidden(Entity)); };
            Foreach(S, [&](const xecs::component::entity& Ent, const xlioncore::transform& Local, const primitive&, const xecs::component::parent* pParent, const xlioncore::render_transform* pRender) noexcept
            {
                if (NotPickable(Ent.m_Value)) return;
                const auto T = xlioncore::WorldOf(Local, pParent, pRender, xlioncore::FixedInterpolateOf(getGameMgr()));   // a child's Transform is relative to its parent: what is drawn is at its world pose
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
            LeaveOut(TextQuery, View);
            auto TS = Search(TextQuery);
            Foreach(TS, [&](const xecs::component::entity& Ent, const xlioncore::transform& Local, const text& Tx, const xecs::component::parent* pParent, const xlioncore::render_transform* pRender) noexcept
            {
                if (NotPickable(Ent.m_Value)) return;
                const auto T = xlioncore::WorldOf(Local, pParent, pRender, xlioncore::FixedInterpolateOf(getGameMgr()));   // a child's Transform is relative to its parent: what is drawn is at its world pose
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
            // The Meshes: the box of their geometry (its bounds, scaled, rotated and placed by the entity). A geometry not loaded yet is not picked.
            // ponytail: the bounds, not the triangles - a click in the empty part of a big mesh's box picks it; ray vs the clusters when it shows
            xecs::query::instance MeshQuery;
            MeshQuery.m_Must.AddFromComponents<xlioncore::transform, mesh>();
            LeaveOut(MeshQuery, View);
            auto MS = Search(MeshQuery);
            if (bMeshes) Foreach(MS, [&](const xecs::component::entity& Ent, const xlioncore::transform& Local, const mesh& M, const xecs::component::parent* pParent, const xlioncore::render_transform* pRender) noexcept
            {
                if (NotPickable(Ent.m_Value) || M.m_Geom.empty()) return;
                auto* pGeom = static_cast<xgeom_static::xgpu::geom*>(xlioncore::resources::GetResource(geom_static_type_guid_v.m_Value, M.m_Geom.m_Instance.m_Value));
                if (!pGeom) return;
                const auto T = xlioncore::WorldOf(Local, pParent, pRender, xlioncore::FixedInterpolateOf(getGameMgr()));
                if (!T.m_Scale.isFinite() || T.m_Scale.m_X == 0.0f || T.m_Scale.m_Y == 0.0f || T.m_Scale.m_Z == 0.0f) return;
                const xmath::fvec3 LocalCenter = (pGeom->m_BBox.m_Min + pGeom->m_BBox.m_Max) * 0.5f;
                const xmath::fvec3 Half        = (pGeom->m_BBox.m_Max - pGeom->m_BBox.m_Min) * 0.5f * T.m_Scale;
                const xmath::fvec3 Center      = T.m_Position + T.m_Rotation.RotateVector(LocalCenter * T.m_Scale);
                float THit;
                if (xeditor_tools::picking::RayOBBIntersect(Origin, Dir, Center, T.m_Rotation, Half.AbsCopy(), THit) && THit < MaxT)
                    Hit.Consider(Ent.m_Value, THit);
                else if (THit = InsideExit(Origin, Dir, Center, T.m_Rotation, Half.AbsCopy()); THit > 0.0f && THit < MaxT)
                    Hit.Consider(Ent.m_Value, THit);        // (the eye inside its box - a street, a room: picked where its box ends, so what is in front of it wins)
            });

            // The lamps (a point, a spot): a 0.25 m sphere at the light - they draw nothing into the G-buffer, so only this picks them (Tomas,
            // 2026-11-08: the street lamps of Bistro Game could not be clicked). The sun has no place to click
            const auto LampPick = [&]<typename T_LIGHT>() noexcept
            {
                xecs::query::instance LampQuery;
                LampQuery.m_Must.AddFromComponents<xlioncore::transform, T_LIGHT>();
                LeaveOut(LampQuery, View);
                auto LS = Search(LampQuery);
                Foreach(LS, [&](const xecs::component::entity& Ent, const xlioncore::transform& Local, const T_LIGHT&, const xecs::component::parent* pParent, const xlioncore::render_transform* pRender) noexcept
                {
                    if (NotPickable(Ent.m_Value)) return;
                    const auto T = xlioncore::WorldOf(Local, pParent, pRender, xlioncore::FixedInterpolateOf(getGameMgr()));
                    float THit;
                    if (xeditor_tools::picking::RaySphereIntersect(Origin, Dir, T.m_Position, 0.25f, THit) && THit < MaxT) Hit.Consider(Ent.m_Value, THit);
                });
            };
            LampPick.template operator()<light_point>();
            LampPick.template operator()<light_spot>();

            return Hit.isValid() ? Hit.m_Id : xecs::component::entity::invalid_entity_v;
        }

        // The ray's exit distance from a box it starts inside (0: it starts outside). RayOBBIntersect refuses a ray from inside: a Level whose
        // one mesh is the whole street (Bistro) picked nothing at all, the eye always inside its bounds (Tomas, 2026-11-08)
        static float InsideExit(const xmath::fvec3& Origin, const xmath::fvec3& Dir, const xmath::fvec3& Center, const xmath::fquat& Rotation, const xmath::fvec3& Half) noexcept
        {
            const xmath::fquat InvR = Rotation.InverseCopy();
            const xmath::fvec3 O = InvR.RotateVector(Origin - Center), D = InvR.RotateVector(Dir);
            const float Oa[3]{ O.m_X, O.m_Y, O.m_Z }, Da[3]{ D.m_X, D.m_Y, D.m_Z }, Ha[3]{ Half.m_X, Half.m_Y, Half.m_Z };
            float TExit = std::numeric_limits<float>::max();
            for (int i = 0; i < 3; ++i)
            {
                if (Oa[i] < -Ha[i] || Oa[i] > Ha[i]) return 0.0f;
                if (std::fabs(Da[i]) > 1.0e-12f) TExit = std::min(TExit, ((Da[i] > 0.0f ? Ha[i] : -Ha[i]) - Oa[i]) / Da[i]);
            }
            return TExit == std::numeric_limits<float>::max() ? 0.0f : TExit;
        }
    };
}

#endif // XLIONRENDER_SYSTEM_H