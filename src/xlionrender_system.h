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
#include "dependencies/xLIONCore/src/transform/xlioncore_locomotion.h"
#include "dependencies/xLIONCore/src/tags/xlioncore_tags.h"
#include "dependencies/xLIONCore/src/physics/xlioncore_physics.h"
#include "dependencies/xLIONCore/src/game/xlioncore_game.h"
#undef  XSCRIPT_IMPORT_ONLY
#include "plugins/xscript_module.plugin/source/Runtime/xscript_registration.h"   // registering macro back for anything after
#include "dependencies/xeditor_tools/src/xeditor_tools_picking.h"
#include "dependencies/xLIONCore/src/physics/xlioncore_ragdoll_api.h"
#include "dependencies/xAnimation/source/nervous/xanimation_nervous.h"
#include "dependencies/xAnimation/source/motion/xanimation_warp_window.h"
#include "dependencies/xAnimation/source/motion/xanimation_recovery.h"
#include "dependencies/xAnimation/source/motion/xanimation_distance_match.h"
#include "dependencies/xAnimation/source/physical/xanimation_get_up.h"
#include "dependencies/xAnimation/source/motion/xanimation_reflex_pose.h"
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
            A.m_WarpStride = 1; A.m_WarpAngle = 0;              // (and its warping) A.m_RootVelocity = xmath::fvec3(0, 0, 0);
            const auto* pGame = xlioncore::game::From(getGameMgr());
            if (pGame == nullptr) return;
            using namespace xlioncore::resources;
            std::uint64_t Pkg = A.m_Package.m_Instance.m_Value;
            if (Pkg == 0 && pSkin && !pSkin->m_Geom.empty())
                if (auto* pGeom = static_cast<xgeom_skin::xgpu::geom*>(GetResource(geom_skin_type_guid_v.m_Value, pSkin->m_Geom.m_Instance.m_Value))) Pkg = pGeom->getAnimationRef().m_Instance.m_Value;
            auto* pPkg = Pkg ? static_cast<xanim_package::anim_package*>(GetResource(anim_package_type_guid_v.m_Value, Pkg)) : nullptr;
            // a request by name (K1 play): a new name switches the clip (from its start; the renderer eases it in); a one-shot remembers the way back
            if (pPkg) OverlayEvents(*pPkg, A, pGame->m_Time.m_DeltaTime);     // (the overlay's events, playing or paused below: the overlay has its own clock)
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
            xlioncore::locomotion* pLoco = nullptr;             // (G0: the game's Locomotion, on this entity or its parent: it says where to go, its feet's velocity goes back)
            if (hasComponents<xlioncore::locomotion>(Ent)) (void)findEntity(Ent, [&](xlioncore::locomotion& L) noexcept { pLoco = &L; });      // (findEntity asserts the entity has them)
            if (!pLoco && pParent && hasComponents<xlioncore::locomotion>(pParent->m_Value)) (void)findEntity(pParent->m_Value, [&](xlioncore::locomotion& L) noexcept { pLoco = &L; });
            if (pLoco)
            {
                A.m_bMoveInWorld = true; A.m_Move = xmath::fvec3(pLoco->m_Move.m_X, 0, pLoco->m_Move.m_Z); A.m_bRootToVelocity = true;
                if (pLoco->m_Action.empty()) A.m_LocoAction.clear();       // (an action asked once by the game: a one-shot by clip name; asked again after it was cleared)
                else if (pLoco->m_Action != A.m_LocoAction) { A.m_LocoAction = pLoco->m_Action; A.m_ClipName = pLoco->m_Action; A.m_bOneShot = true; A.m_RequestHash = 0; }
                if (pLoco->m_Face.m_X != 0 || pLoco->m_Face.m_Z != 0) { A.m_Face = pLoco->m_Face; A.m_bFaceMove = false; } else { A.m_Face = xmath::fvec3(0, 0, 0); A.m_bFaceMove = true; }
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
                if (hasComponents<xlioncore::physics::physics_dynamics>(pParent->m_Value)) (void)findEntity(pParent->m_Value, [&](const xlioncore::physics::physics_dynamics& D) noexcept { V = D.m_LinearVelocity; });
                V.m_Y = 0; if (V.Length() > 20.0f) V = xmath::fvec3(0, 0, 0);         // (a teleport: no speed)
                A.m_LastWorld = W; A.m_bHasLastWorld = true;
                A.m_bMoveInWorld = true; A.m_bFaceMove = false; A.m_Move = V;
                if (V.Length() > 0.3f) FollowFace = V;
                {   // (where the velocity is heading: matching searched with the velocity now only and the clip lagged the body's starts and
                    // brakes - SoccerAnim feet ~130 cm/s; Fable's order: an acceleration-aware wish first)
                    const float Dt = pGame->m_Time.m_DeltaTime, K = 1.0f - std::exp(-Dt / 0.1f);
                    const auto Ac = A.m_bHasLastWorld ? (V - A.m_LastVel) * (1.0f / Dt) : xmath::fvec3(0, 0, 0);
                    A.m_Accel = A.m_Accel + (Ac - A.m_Accel) * K; A.m_LastVel = V;
                    auto Ahead = V + A.m_Accel * 0.5f; Ahead.m_Y = 0;
                    if (const float L = Ahead.Length(); L > 6.0f) Ahead = Ahead * (6.0f / L);
                    A.m_MoveAhead = Ahead;
                }
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
            xmath::fvec3 ClipDir(0, 0, 0);                      // (the clip's travel, unit, before the turn: the eased turn is applied to it below)
            if (pT && (A.m_bRootMotion || A.m_bFollowParent) && A.m_Speed > 0 && Clip.m_RootMotionMode != xanim_package::root_motion_mode::NONE && (A.m_Move.m_X != 0 || A.m_Move.m_Z != 0))
            {
                auto CV = (A.m_bMatching ? xanimation::ClipVelocityAt(*pPkg, A.m_Clip, Before) : xanimation::ClipVelocity(*pPkg, A.m_Clip)) * (A.m_Speed * pT->m_Scale.m_X);      // (matching: around the frame playing - starts and stops do not walk evenly)      // ponytail: a uniform scale (X) for the clip's world speed
                if (pSD) { const auto N = xanimation::ClipMirrorNormal(*pSD, *pPkg, A.m_Clip); CV = CV - N * (2.0f * CV.Dot(N)); }
                Wp = xanimation::WarpFor(CV, A.m_Move);
                if (const float L = CV.Length(); L > 1e-4f) ClipDir = CV * (1.0f / L);
                A.m_Time = Before + pGame->m_Time.m_DeltaTime * A.m_Speed * Wp.m_Rate;
            }
            // (eased, not snapped: a matching jump changes the clip's travel and with it the angle - up to 90 deg in a frame swung the legs:
            // the AnimDemo pops, seen in the motion trace. The pose and the root motion both take the eased values)
            {
                const float K = A.m_bWarpEased ? 1.0f - std::exp(-pGame->m_Time.m_DeltaTime / 0.15f) : 1.0f;
                float d = Wp.m_Angle - A.m_WarpAngleEased; while (d > 3.14159265f) d -= 6.2831853f; while (d < -3.14159265f) d += 6.2831853f;
                A.m_WarpAngleEased += d * K; A.m_WarpStrideEased += (Wp.m_Stride - A.m_WarpStrideEased) * K; A.m_bWarpEased = true;
                Wp.m_Angle = A.m_WarpAngleEased; Wp.m_Stride = A.m_WarpStrideEased;
                // (the stride's direction turned by the EASED angle, as the legs and the root motion are: from the target angle, during every turn
                // the feet were stretched along one direction while they travelled along another - SoccerAnim's feet slid 250 cm/s at warp 15-75 deg)
                if (ClipDir.LengthSq() > 0) Wp.m_Dir = xanimation::details::AxisAngle(xmath::fvec3(0, 1, 0), Wp.m_Angle) * ClipDir;
                A.m_WarpStride = Wp.m_Stride; A.m_WarpAngle = Wp.m_Angle; A.m_WarpDir = Wp.m_Dir;
            }
            // U1 distance matching (motion/xanimation_distance_match.h): the time where the clip's root has as far left to go (a stop) or has gone
            // as far (a start) as the body has to or from Target, on the ground plane; the clip's root motion is the distance curve
            if (A.m_bDist && pT && Length > 0)
            {
                auto Off = A.m_DistTarget - pT->m_Position; Off.m_Y = 0;
                const float Have = Off.Length() / std::max(pT->m_Scale.m_X, 1e-6f);        // ponytail: a uniform scale (X), as the clip's world speed above
                // ponytail: the straight distance between the two times, not the path length: a start/stop is straight; a curved one needs the path summed
                const auto Flat = [&](float a, float b) { const auto M = xanimation::RootStep(*pPkg, A.m_Clip, a, b).m_Move; return std::sqrt(M.m_X * M.m_X + M.m_Z * M.m_Z); };
                A.m_Time = A.m_bDistStop ? xanimation::DistanceMatchStop(std::clamp(Before, 0.0f, Length), xanimation::DistanceMatchTime([&](float t) { return Flat(t, Length); }, Have, 0.0f, Length))
                                         : xanimation::DistanceMatchTime([&](float t) { return Flat(0.0f, t); }, Have, 0.0f, Length);
                if (A.m_DistLogAt < 0 || std::fabs(A.m_Time - A.m_DistLogAt) >= 0.1f || (A.m_Time >= Length && A.m_DistLogAt < Length))
                {
                    A.m_DistLogAt = A.m_Time;
                    std::printf("[AnimDistance] entity 0x%llX: %s, clip %.3f s of %.3f: the clip's root has %.1f cm %s, the body %.1f cm (the clip %.1f cm in all)%c", (unsigned long long)Ent.m_Value, A.m_bDistStop ? "stop" : "start"
                        , A.m_Time, Length, 100 * (A.m_bDistStop ? Flat(A.m_Time, Length) : Flat(0.0f, A.m_Time)) * pT->m_Scale.m_X, A.m_bDistStop ? "left" : "gone", 100 * Have * pT->m_Scale.m_X, 100 * Flat(0.0f, Length) * pT->m_Scale.m_X, 10);
                    std::fflush(stdout);
                }
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
                if (!A.m_bRootToVelocity) D = D + Wp.m_Slide * (pGame->m_Time.m_DeltaTime / std::max(1e-6f, pT->m_Scale.m_X));       // (past the warp's limits: still where the game asks; Move is world units, D is scaled below)      // (RootToVelocity: the feet only - the body goes where they go, never faster)
                auto WD = pT->m_Rotation * xmath::fvec3(D.m_X * pT->m_Scale.m_X, D.m_Y * pT->m_Scale.m_Y, D.m_Z * pT->m_Scale.m_Z);
                // S6 warp window: inside Warp.From..To (clip seconds) the root motion is bent so the entity stands on Warp.Target (and faces
                // Warp.Yaw) when the clip reaches To (motion/xanimation_warp_window.h); a target moved meanwhile is still met
                if (A.m_bWarp && Before < A.m_WarpTo && A.m_Time > A.m_WarpFrom && A.m_Speed > 0)
                {
                    const float End = A.m_WarpTo;
                    const auto Rest = xanimation::RootStep(*pPkg, A.m_Clip, std::min(A.m_Time, End), End);
                    const auto RestW = pT->m_Rotation * xmath::fvec3(Rest.m_Move.m_X * pT->m_Scale.m_X, Rest.m_Move.m_Y * pT->m_Scale.m_Y, Rest.m_Move.m_Z * pT->m_Scale.m_Z);
                    const auto F = pT->m_Rotation * xmath::fvec3(0, 0, 1);
                    const float Yaw = std::atan2(F.m_X, F.m_Z), Dt = pGame->m_Time.m_DeltaTime;
                    const float TimeLeft = (End - Before) / A.m_Speed;
                    auto ToTarget = A.m_WarpTarget - pT->m_Position; ToTarget.m_Y = A.m_bWarpHeight ? ToTarget.m_Y : WD.m_Y + RestW.m_Y;
                    const auto W = xanimation::WarpWindowStep(WD, Turn, RestW, Rest.m_Turn, ToTarget, A.m_WarpYaw * 0.017453293f - Yaw, Dt, TimeLeft, A.m_bWarpTurn);
                    WD = W.m_Move; Turn = W.m_Turn;
                    if (A.m_Time >= End)
                    {
                        std::printf("[AnimWarp] entity 0x%llX: window ends at clip %.2f s, %.1f cm from the target\n", (unsigned long long)Ent.m_Value, End, 100 * (pT->m_Position + WD - A.m_WarpTarget).Length());
                        std::fflush(stdout);
                    }
                }
                if (A.m_bRootToVelocity)                     // (G0 (b): the game moves its body with it; the entity itself stays where its parent puts it)
                    A.m_RootVelocity = pGame->m_Time.m_DeltaTime > 0 ? WD * (1.0f / pGame->m_Time.m_DeltaTime) : xmath::fvec3(0, 0, 0);
                    {   // (inertialized as the pose is: at a matching jump the new clip's velocity would move the body at once while the drawn
                        // pose still blends from the old clip, and the planted foot dragged - SoccerAnim: 141 cm/s near jumps, 7 away from them)
                        const float Dt = pGame->m_Time.m_DeltaTime, Half = std::max(1e-3f, A.m_BlendTime * 0.25f);
                        if (A.m_MatchJumps != A.m_RootVelJumps) { A.m_RootVelOffset = A.m_RootVelLast - A.m_RootVelocity; A.m_RootVelJumps = A.m_MatchJumps; }
                        else A.m_RootVelOffset = A.m_RootVelOffset * std::exp(-0.69314718f * Dt / Half);
                        A.m_RootVelocity = A.m_RootVelocity + A.m_RootVelOffset; A.m_RootVelLast = A.m_RootVelocity;
                    }
                if (A.m_bRootToVelocity && pLoco) pLoco->m_FeetVelocity = A.m_RootVelocity;
                else
                {
                    if (A.m_bCollide) WD = Collide(Ent.m_Value, A, *pT, WD);
                    pT->m_Position = pT->m_Position + WD;
                }
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
            if (A.m_bMatching && pT && pSkin && !A.m_bOneShot) Match(A, *pSkin, *pPkg, pT);     // (an action requested as a one-shot plays out; matching resumes after)
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
                    {   // (a clip with a "NoMatch" marker is an action, played on request - a kick, a dive - never a matching candidate: no entries)
                        static const std::uint32_t NoMatch = xstrtool::CRC32("NoMatch");
                        for (const auto& E : Pkg.getClipEvents(c)) if (E.m_Kind == xanim_package::event_kind::MARKER && E.m_NameHash == NoMatch) Mc.m_Length = 0;
                    }
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
            xmath::fvec3 WishNow = Wish;                            // (FollowParent: the velocity now, and the wish where it is heading)
            if (A.m_bFollowParent && pT)
            {
                const auto Ah = xanimation::details::TurnY(pT->m_Rotation.ConjugateCopy() * A.m_MoveAhead, -Yaw0);
                Wish = xmath::fvec3(Ah.m_X, 0, Ah.m_Z);
                WishNow = xanimation::ReachableWish(M.m_Db, WishNow);
            }
            Wish = xanimation::ReachableWish(M.m_Db, Wish);        // (toward a direction the clips walk; warping turns them the rest)
            if (!A.m_bFollowParent) WishNow = Wish;
            xanimation::matching_state St; St.m_Clip = A.m_Clip; St.m_Time = A.m_Time; St.m_Since = A.m_MatchSince + Dt; St.m_SinceJump = A.m_MatchSinceJump + Dt; St.m_Jumps = int(A.m_MatchJumps);
            xanimation::matching_params Pm;
            {   // (experiments: XLION_ANIM_CLIPCHANGE / XLION_ANIM_MINGAP override the jump cost and the shortest time between jumps)
                static const float CC = [] { const char* e = std::getenv("XLION_ANIM_CLIPCHANGE"); return e && *e ? float(std::atof(e)) : -1.0f; }();
                static const float MG = [] { const char* e = std::getenv("XLION_ANIM_MINGAP"); return e && *e ? float(std::atof(e)) : -1.0f; }();
                if (CC >= 0) Pm.m_ClipChange = CC;
                if (MG >= 0) Pm.m_MinGap = MG;
            }
            if (m_Searches >= SearchBudget()) { Pm.m_Interval = 1e9f; ++m_SearchWaits; }       // (over the budget: no search unless the clip is ending)
            const bool bJumped = xanimation::UpdateMatching(M.m_Db, M.m_Clips, St, xanimation::PredictWish(WishNow, Wish, YawAsked), 0, Pm);
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
                xanimation::MakeQuery(M.m_Db, xanimation::PlayingEntry(M.m_Db, St), xanimation::PredictWish(WishNow, Wish, YawAsked), Q.data());
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
            FireClipEvents(Pkg, A, A.m_Clip, Before, A.m_Time, Length, A.m_bLoop, pMirror);
        }
        // S1's overlay (superset: Unreal's montage slot notifies): its clip's events fire as the base clip's do, on its own clock - started by a new
        // Overlay.Play, at the overlay's speed, once per pass when it loops, until its end otherwise. ponytail: the poser keeps the overlay's blend
        // clock and this one its events' (both on game time, started by the same Play): one shared clock if they are ever seen to drift
        void OverlayEvents(const xanim_package::anim_package& Pkg, animation& A, float Dt) noexcept
        {
            if (A.m_OverlayPlay != A.m_OvEventPlay) { A.m_OvEventPlay = A.m_OverlayPlay; A.m_OvEventTime = -1e-6f; A.m_bOvEventOn = A.m_OverlayClip >= 0; A.m_OvEventClip = A.m_OverlayClip; }     // (just before 0: an event at the start fires)
            if (!A.m_bOvEventOn || A.m_OverlayClip != A.m_OvEventClip || A.m_OverlayClip < 0 || A.m_OverlayClip >= int(Pkg.getClips().size())) { A.m_bOvEventOn = false; return; }
            const auto& C = Pkg.getClips()[std::size_t(A.m_OverlayClip)];
            const float Len = C.m_FPS > 0 ? float(C.m_nFrames) / float(C.m_FPS) : 0.0f;
            if (Len <= 0) { A.m_bOvEventOn = false; return; }
            const float Before = A.m_OvEventTime;
            A.m_OvEventTime += Dt * std::max(0.0f, A.m_OverlaySpeed);
            if (!A.m_bOverlayLoop && A.m_OvEventTime >= Len) { A.m_OvEventTime = Len; A.m_bOvEventOn = false; }
            FireClipEvents(Pkg, A, A.m_OverlayClip, Before, A.m_OvEventTime, Len, A.m_bOverlayLoop, nullptr);
        }
        void FireClipEvents(const xanim_package::anim_package& Pkg, animation& A, int iClip, float Before, float After, float Length, bool bLoop, const xanimation::mirror_table* pMirror) noexcept
        {
            const auto Ev = Pkg.getClipEvents(iClip);
            if (Ev.empty()) return;
            thread_local std::vector<xanimation::event_def> Defs; thread_local std::vector<xanimation::event_fire> Fired;
            Defs.clear(); Fired.clear();
            for (std::uint32_t i = 0; i < Ev.size(); ++i) Defs.push_back({ i, Ev[i].m_Time });
            xanimation::event_cursor C; C.Seek(Before);
            xanimation::Advance(C, xanimation::event_track{ Defs, Length, bLoop }, After, Fired);
            static const bool bLog = [] { const char* e = std::getenv("XLION_ANIM_EVENTS"); return e && *e == '1'; }();
            for (const auto& F : Fired)
            {
                const auto& E = Ev[F.m_Id];
                ++A.m_nEvents; A.m_LastEvent = E.m_NameHash;
                const std::int16_t Bone = (pMirror && E.m_iBone >= 0 && std::size_t(E.m_iBone) < pMirror->m_Bones.size()) ? std::int16_t(pMirror->m_Bones[std::size_t(E.m_iBone)].m_Src) : E.m_iBone;
                if (A.m_nFired < A.m_Fired.size()) A.m_Fired[A.m_nFired++] = { E.m_NameHash, std::uint8_t(E.m_Kind), Bone, E.m_Time, E.m_Duration };
                if (bLog) std::printf("[AnimEvent] clip %d%s at %.3f s: kind %d name %08X bone %d%s\n", iClip, iClip == A.m_Clip ? "" : " (overlay)", E.m_Time, int(E.m_Kind), E.m_NameHash, Bone, pMirror ? " (mirrored: the mirror bone)" : "");
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
        std::uint64_t m_nPushes = 0;      // (the inspector's shoves so far: each one's hit id)

        struct graph_entry { xragdoll::runtime_graph m_G; std::vector<xanimation::rigid> m_Rest; };
        struct entity_state
        {
            xmath::fvec3        m_Offset;               // entity - hips (x, z) at the switch
            xanimation::rigid   m_E;                    // the entity's frame and scale at the switch: where the tracked animation is played (R1)
            float               m_Scale = 1;
            float               m_T = 0, m_KE0 = 0, m_Y0 = 0, m_Mass = 0, m_WorstEnergy = 0, m_RestAt = -1, m_Sleep = 0;
            std::vector<float>  m_Anchor;
            bool                m_bReported = false;
            double              m_TrackSum = 0, m_TrackSumW = 0;    // R1: the joints' (and the bodies' world) rotation from the animation (deg, squared, summed); the joints' worst
            int                 m_nTrack = 0;
            float               m_TrackMax = 0;
            float               m_Age = 0, m_HitAt = 0, m_Settled = 0, m_LastE = 1e9f, m_MinE = 1e9f, m_MinAt = 0;
            float               m_Supported = 0;            // the assistance ledger: seconds its root was held (kinematic, supported tracking)
            bool                m_bForced = false;          //  and a return forced by the time limit    // R3b': time since activation, of the last hit, settled for; the last tracking error
            const xanimation::body_graph* m_pGraph = nullptr;       // (the graph it was activated with: m_Graphs keeps it)
            std::vector<std::uint64_t>    m_Hits;                   // the hit events this activation took
            float                         m_ReflexAt = -1;          // D2: when the reflex pose started (-1: none) and where the threat is (world)
            xmath::fvec3                  m_Threat{ 0, 0, 0 };
            float                         m_HandStart = -1, m_HandMin = 0;
            bool                          m_bAssistLogged = false;
            // D4 catch fall (the limbs' reflex set): the hips' last position (their velocity), the reflex weight now and a step back, where the
            // hands reach (the predicted landing, world), and its measure: when a hand and the head first reached the ground, the head's speed then
            xmath::fvec3                  m_PrevHips{ 0, 0, 0 }, m_PrevHead{ 0, 0, 0 }, m_Landing{ 0, 0, 0 };
            const xanimation::skeleton_data* m_pSD = nullptr;       // (its skeleton, as the system keeps it)
            float                         m_HipsUp = -1;            // the hips' height above the stand plane at the switch (its standing height)
            float                         m_DownStill = 0;          // KD5: seconds fallen (hips more than half down) and still
            bool                          m_bPrevHips = false, m_bFallLogged = false;
            float                         m_FallW = 0, m_FallWPrev = 0, m_FallAt = -1, m_HandDownAt = -1, m_HeadLow = 1e9f, m_HeadFall = 0, m_HandLow = 1e9f;
        };
        std::unordered_map<const void*, graph_entry>                m_Graphs;       // per Ragdoll resource as loaded
        std::unordered_map<const void*, xanimation::skeleton_data>  m_Skeletons;
        std::unordered_map<std::uint64_t, entity_state>             m_State;
        xanimation::pose                                            m_P0, m_P1;
        struct reflex_now { xmath::fvec3 m_Threat; float m_Weight = 0, m_WeightPrev = 0; xanimation::reflex_pose_params m_Params; };
        const reflex_now*                                           m_pReflex = nullptr;    // (set around AnimatedWorlds while a reflex pose plays)
        struct reflex_rig_entry { xanimation::reflex_rig m_Rig; bool m_bInit = false; };
        std::unordered_map<const void*, reflex_rig_entry>           m_ReflexRigs;
        std::vector<xlioncore::physics::impact>                     m_Impacts;      // this frame's solid hits (TakeImpacts, read once per update)
        struct nerves { xanimation::nervous::drive_state m_Drives; xanimation::nervous::reflex m_Running = xanimation::nervous::reflex::NONE, m_RunningLimbs = xanimation::nervous::reflex::NONE; std::uint64_t m_Tick = 0; };
        std::unordered_map<std::uint64_t, nerves>                   m_Nerves;       // the nervous system of each character with a Ragdoll that reacts (its drives outlive an activation)

        void OnPreUpdate(void) noexcept
        {
            m_Impacts.resize(256);
            const int n = xlioncore::physics::TakeImpacts(getGameMgr(), m_Impacts.data(), int(m_Impacts.size()));
            m_Impacts.resize(std::size_t(std::min(n, 256)));
        }

        void operator()( const xecs::component::entity& Ent, ragdoll& R, const skinned_mesh& Skin, animation* pAnim, xlioncore::transform& T, const xecs::component::parent* pParent ) noexcept
        {
            auto& GameMgr = getGameMgr();
            const auto* pGame = xlioncore::game::From(GameMgr);
            if (pGame == nullptr) return;
            if (R.m_bGettingUp && !R.m_bActive && (pAnim == nullptr || !pAnim->m_bOneShot)) R.m_bGettingUp = false;      // (the get-up clip played out)
            // R3b' (G0 stage 2): a solid hit on this entity or on its parent (a player's capsule) becomes this character's Hit: one event per side
            // of each impact (the impact's id, A or B), pushed away from the other along the contact normal
            // The nervous system decides (plan 3.5, contract 5.7; nervous/xanimation_nervous.h): every jolt raises the character's drives (surprise,
            // pain), and the hit reaction is a reflex chosen by utility scoring with hysteresis; ReactSpeed sets how big a jolt surprises it enough
            auto& Nv = m_Nerves[Ent.m_Value];
            if (R.m_ReactSpeed > 0)
            {
                xanimation::nervous::drive_params NP; NP.m_SurprisePerMps = 0.5f / R.m_ReactSpeed;
                Nv.m_Drives.Decay(NP, pGame->m_Time.m_DeltaTime);
                const xlioncore::physics::impact* pJolt = nullptr; float Jolt = 0; int JoltSide = -1;
                for (const auto& I : m_Impacts)
                {
                    const std::uint64_t Me = Ent.m_Value, Up = pParent ? pParent->m_Value.m_Value : 0;
                    const int Side = (I.m_A == Me || (Up && I.m_A == Up)) ? 0 : (I.m_B == Me || (Up && I.m_B == Up)) ? 1 : -1;
                    if (Side < 0) continue;
                    // the speed change this side gets in a perfectly inelastic hit, mu v / m_own (mu: the reduced mass; a static or kinematic
                    // other: v): a ball barely moves a player
                    const float Mo = Side == 0 ? I.m_MassA : I.m_MassB, Mt = Side == 0 ? I.m_MassB : I.m_MassA;
                    const float Dv = Mo <= 0 ? 0.0f : Mt <= 0 ? I.m_Speed : I.m_Speed * Mt / (Mo + Mt);
                    Nv.m_Drives.Impact(NP, Dv);
                    if (Dv > Jolt) { Jolt = Dv; pJolt = &I; JoltSide = Side; }
                }
                static const auto s_Reflexes = xanimation::nervous::DefaultReflexes();
                xanimation::nervous::percepts Pc; Pc.m_Impact = Jolt; Pc.m_Support = R.m_bSupported ? 1.0f : 0.0f;
                const auto D = xanimation::nervous::Choose(s_Reflexes, Nv.m_Drives, Pc, Nv.m_Running, ++Nv.m_Tick);
                if (D.m_bNew)
                {
                    std::printf("[Nervous] entity 0x%llX: surprise %.2f pain %.2f -> %s (scores:", (unsigned long long)Ent.m_Value, D.m_Drives[xanimation::nervous::SURPRISE], D.m_Drives[xanimation::nervous::PAIN], xanimation::nervous::reflex_names_v[int(D.m_Chosen)]);
                    for (const auto& [Rx, Sc] : D.m_Scores) std::printf(" %s %.2f", xanimation::nervous::reflex_names_v[int(Rx)], Sc);
                    std::printf(")%c", 10); std::fflush(stdout);
                }
                const bool bReact = D.m_bNew && D.m_Chosen == xanimation::nervous::reflex::HIT_REACTION;
                // KN3: a jolt the body takes without a reaction (under ReactSpeed, a ball) flinches its spine with the push: 8 deg per m/s, at most 15
                if (pAnim && pJolt && Jolt > 0.2f && !bReact && !R.m_bActive)
                {
                    const float Sg = JoltSide == 0 ? -1.0f : 1.0f;
                    auto Dir = pJolt->m_Normal * Sg; Dir.m_Y = 0;
                    if (const float Ln = Dir.Length(); Ln > 1e-4f)
                    {
                        pAnim->m_FlinchDir = Dir * (1.0f / Ln); pAnim->m_FlinchAngle = std::min(0.26f, 0.14f * Jolt); ++pAnim->m_FlinchSerial;
                        std::printf("[Flinch] entity 0x%llX: a %.2f m/s jolt with no hit reaction - the spine flinches %.1f deg with it%c", (unsigned long long)Ent.m_Value, Jolt, pAnim->m_FlinchAngle * 57.29578f, 10);
                        std::fflush(stdout);
                    }
                }
                Nv.m_Running = D.m_Chosen;
                R.m_NvSurprise = Nv.m_Drives.m_Value[xanimation::nervous::SURPRISE]; R.m_NvPain = Nv.m_Drives.m_Value[xanimation::nervous::PAIN];
                R.m_pNvBody = xanimation::nervous::reflex_names_v[int(D.m_Chosen)];
                if (bReact && pJolt && R.m_HitEventId == R.m_LastHitId)
                {
                    const auto& I = *pJolt; const int Side = JoltSide; const float Dv = Jolt;
                    R.m_HitEventId = I.m_Id * 2 + std::uint64_t(Side); R.m_HitPoint = I.m_Point;
                    float RagMass = 60.0f;
                    if (auto* pRag = R.m_Ragdoll.empty() ? nullptr : static_cast<const xragdoll::ragdoll*>(xlioncore::resources::GetResource(ragdoll_type_guid_v.m_Value, R.m_Ragdoll.m_Instance.m_Value))) RagMass = pRag->m_TotalMass;
                    R.m_HitImpulse = I.m_Normal * ((Side == 0 ? -1.0f : 1.0f) * R.m_ImpactGain * RagMass * Dv);
                    std::printf("[Ragdoll] entity 0x%llX: impact %llu at %.1f m/s becomes hit %llu (side %c, masses %.0f/%.0f kg: speed change %.2f m/s on its %.0f kg, impulse %.0f N s)\n", (unsigned long long)Ent.m_Value, (unsigned long long)I.m_Id, I.m_Speed, (unsigned long long)R.m_HitEventId, Side == 0 ? 'A' : 'B', I.m_MassA, I.m_MassB, Dv, RagMass, R.m_HitImpulse.Length());
                    std::fflush(stdout);
                }
            }
            // R3b' slice 1: a new hit (a HitEventId not seen yet) promotes a tracking character to physics; it returns when settled
            if (R.m_Push != R.m_PushSeen)        // (a shove from the inspector: the character's facing and mass make it a hit)
            {
                R.m_PushSeen = R.m_Push;
                float Mass = 60.0f;
                if (auto* pRag = R.m_Ragdoll.empty() ? nullptr : static_cast<const xragdoll::ragdoll*>(xlioncore::resources::GetResource(ragdoll_type_guid_v.m_Value, R.m_Ragdoll.m_Instance.m_Value))) Mass = pRag->m_TotalMass;
                auto Fw = T.m_Rotation * xmath::fvec3(0, 0, 1); Fw.m_Y = 0; Fw = Fw.NormalizeSafeCopy();
                const float a = R.m_PushDir * 0.017453293f;
                const auto Dir = (Fw * std::cos(a) + xmath::fvec3(Fw.m_Z, 0, -Fw.m_X) * std::sin(a)).NormalizeSafeCopy();
                std::printf("[Push] entity 0x%llX: %.2f m/s at %.0f deg (%.0f N s on %.0f kg)%c", (unsigned long long)Ent.m_Value, R.m_PushSpeed, R.m_PushDir, R.m_PushSpeed * Mass, Mass, 10); std::fflush(stdout);
                bool bAnimated = false;
                if (R.m_bRecover && pAnim && !R.m_bActive)        // R68: answered by animation (stage 1: a standing character - its feet beside it, its COM above)
                {
                    const float S = T.m_Scale.m_Y; const xmath::fvec3 Sd(Fw.m_Z, 0, -Fw.m_X);
                    const xmath::fvec3 Feet[2] = { T.m_Position + Sd * (0.125f * S), T.m_Position - Sd * (0.125f * S) };
                    const auto Plan = xanimation::PlanRecovery(T.m_Position + xmath::fvec3(0, 0.95f * S, 0), Dir * R.m_PushSpeed, T.m_Position.m_Y, Feet, xmath::fvec3(0.05f * S, 0, 0.12f * S), Fw, 0.85f * S);
                    static constexpr const char* Kinds[] = { "sway", "one step", "stagger", "fall" }; static constexpr char Bands[] = { 'F', 'B', 'L', 'R' };
                    const char Band = Bands[int(Plan.m_Dir)];
                    std::string Clip;
                    {   // "F1=a,F2=b,F3=c,B=d,...": the clip for the band and the strength (1 nudge: sway, 2 firm: one step, 3 hard: stagger - Tomas: a
                        // recovery scales with the push); a key without a digit serves every strength of its band
                        const char Want = Plan.m_Kind == xanimation::recovery_kind::SWAY ? '1' : Plan.m_Kind == xanimation::recovery_kind::STEP1 ? '2' : '3';
                        const auto& L = R.m_RecoverClips; std::size_t i = 0; std::string Any;
                        while (i < L.size())
                        {
                            const auto j = std::min(L.find(',', i), L.size()); const auto e = L.find('=', i);
                            if (e != std::string::npos && e < j && e > i && std::toupper(static_cast<unsigned char>(L[i])) == Band)
                            {
                                std::string V = L.substr(e + 1, j - e - 1); while (!V.empty() && V.back() == ' ') V.pop_back();
                                if (e == i + 1) Any = V; else if (L[i + 1] == Want) Clip = V;
                            }
                            i = j + 1; while (i < L.size() && L[i] == ' ') ++i;
                        }
                        if (Clip.empty()) Clip = Any;
                    }
                    const std::uint64_t Pk = pAnim->m_Package.m_Instance.m_Value;
                    auto* pPkg = Pk ? static_cast<xanim_package::anim_package*>(xlioncore::resources::GetResource(anim_package_type_guid_v.m_Value, Pk)) : nullptr;
                    const int iClip = pPkg && !Clip.empty() ? pPkg->findClipIndex(xstrtool::CRC32(Clip)) : -1;
                    if ((Plan.m_Kind == xanimation::recovery_kind::STEP1 || Plan.m_Kind == xanimation::recovery_kind::STAGGER || Plan.m_Kind == xanimation::recovery_kind::SWAY) && iClip >= 0)
                    {
                        pAnim->m_ClipName = Clip; pAnim->m_RequestHash = 0; pAnim->m_bOneShot = true; pAnim->m_bPlaying = true;
                        bAnimated = true;
                        // the torso gives at the hit (KN3's flinch on the spine chain, over the clip): a small bend with the push, scaled by its speed
                        pAnim->m_FlinchDir = Dir; pAnim->m_FlinchAngle = std::min(0.26f, 0.3f * R.m_PushSpeed); ++pAnim->m_FlinchSerial;
                        // stage 2: aimed - the clip's travel (root motion, from the hips) bent so that by its end the root stands between the stance
                        // foot and the planned landing (the S6 warp window over the whole clip)
                        if (Plan.m_Foot >= 0 && Plan.m_Kind != xanimation::recovery_kind::SWAY)
                        {
                            const auto& C = pPkg->getClips()[std::size_t(iClip)];
                            const auto& Stance = Feet[Plan.m_Foot == 0 ? 1 : 0];
                            pAnim->m_bRootMotion = true;
                            pAnim->m_WarpTarget = xmath::fvec3((Plan.m_Land.m_X + Stance.m_X) * 0.5f, T.m_Position.m_Y, (Plan.m_Land.m_Z + Stance.m_Z) * 0.5f);
                            pAnim->m_WarpFrom = 0; pAnim->m_WarpTo = C.m_FPS > 0 ? float(C.m_nFrames) / float(C.m_FPS) : 1.0f;
                            pAnim->m_bWarpTurn = false; pAnim->m_bWarpHeight = false; pAnim->m_bWarp = true;
                            std::printf("[Recover] entity 0x%llX: aimed - the root to %.2f %.2f by %.2f s (the clip's own travel bent there)%c", (unsigned long long)Ent.m_Value, pAnim->m_WarpTarget.m_X, pAnim->m_WarpTarget.m_Z, pAnim->m_WarpTo, 10); std::fflush(stdout);
                        }
                    }
                    std::printf("[Recover] entity 0x%llX: %s toward %c, %d step(s), land %.2f %.2f; %s%c", (unsigned long long)Ent.m_Value, Kinds[int(Plan.m_Kind)], Band, Plan.m_Steps, Plan.m_Land.m_X, Plan.m_Land.m_Z
                        , bAnimated ? ("clip '" + Clip + "'").c_str() : Clip.empty() ? "no clip for that direction: the physical hit" : iClip < 0 ? ("no clip '" + Clip + "' in its package: the physical hit").c_str() : "a fall: the physical hit", 10);
                    std::fflush(stdout);
                }
                if (!bAnimated)
                {
                    R.m_HitPoint = T.m_Position + xmath::fvec3(0, R.m_PushHeight, 0) - Dir * 0.15f;
                    R.m_HitImpulse = Dir * (R.m_PushSpeed * Mass);
                    R.m_HitEventId = 0x7055000000000000ull + ++m_nPushes;
                }
            }
            const bool bHit = R.m_HitEventId != 0 && R.m_HitEventId != R.m_LastHitId;
            if (bHit && !R.m_bActive && R.m_Tracking > 0) { R.m_bActive = true; R.m_bReaction = true; }
            if (!R.m_bActive)
            {
                if (R.m_pRuntime)
                {
                    // the assistance ledger (contract 2.6 req. 5, 2.8; v0: logged per activation): how long its root was held by the animation
                    // (supported tracking is assistance, never hidden) and whether it was sent back to the animation by the time limit
                    if (auto It = m_State.find(Ent.m_Value); It != m_State.end() && !It->second.m_bAssistLogged && (It->second.m_Supported > 0 || It->second.m_bForced))
                    {
                        std::printf("[Assist] entity 0x%llX: root held by the animation %.2f s of %.2f s active%s\n", (unsigned long long)Ent.m_Value, It->second.m_Supported, It->second.m_Age
                            , It->second.m_bForced ? "; sent back to its animation by the time limit (not settled)" : "");
                        std::fflush(stdout);
                    }
                    xlioncore::physics::DeactivateRagdoll(GameMgr, R.m_pRuntime); R.m_pRuntime = nullptr; m_State.erase(Ent.m_Value);
                }
                return;
            }
            if (R.m_pRuntime == nullptr && !Activate(Ent, R, Skin, pAnim, T, pParent, *pGame)) { R.m_bActive = false; return; }

            auto& St = m_State[Ent.m_Value];
            St.m_Age += pGame->m_Time.m_DeltaTime;
            if (pParent) St.m_E = WorldFrame(T, pParent, St.m_Scale);       // (a child: the animation is played where its parent carries it now)
            if (bHit)       // (the impulse: LIONCore applies it right before the next step, once per event id - its ledger)
            {
                xlioncore::physics::HitRagdoll(GameMgr, *R.m_pRuntime, R.m_HitEventId, R.m_HitPoint, R.m_HitImpulse);
                if (std::find(St.m_Hits.begin(), St.m_Hits.end(), R.m_HitEventId) == St.m_Hits.end())
                {
                    St.m_Hits.push_back(R.m_HitEventId); St.m_HitAt = St.m_Age; St.m_Settled = 0; St.m_MinE = 1e9f; St.m_MinAt = St.m_Age;
                    // D2: the body protects itself from where the push came (0.5 m back along it from the point) and turns its head away
                    const float Il = R.m_HitImpulse.Length();
                    if (R.m_bReflexes && Il > 1e-4f) { St.m_ReflexAt = St.m_Age; St.m_Threat = R.m_HitPoint - R.m_HitImpulse * (0.5f / Il); }
                }   // (a repeat does not restart the settling)
                R.m_LastHitId = R.m_HitEventId;
            }
            // D4 (superset: delirium's catch fall): the limbs' reflex set decides from how fast the hips go down and whether the body is held up;
            // a catch reaches both hands for where the fall will land (the hips' ballistic arc to the ground) and turns the head away from it
            if (R.m_Tracking > 0)       // (perceived even with Reflexes off: the [Fall] measure compares)
            {
                const auto Hips = xlioncore::physics::RagdollRootBody(*R.m_pRuntime, 1.0f).m_P;
                const float Dt = std::max(pGame->m_Time.m_DeltaTime, 1e-4f);
                const auto Vel = St.m_bPrevHips ? (Hips - St.m_PrevHips) * (1.0f / Dt) : xmath::fvec3(0, 0, 0);
                St.m_PrevHips = Hips; St.m_bPrevHips = true;
                if (St.m_HipsUp < 0) St.m_HipsUp = std::max(Hips.m_Y - St.m_E.m_P.m_Y, 1e-3f);
                xanimation::nervous::percepts Pc; Pc.m_Support = R.m_bSupported ? 1.0f : 0.0f; Pc.m_Fall = std::max(0.0f, -Vel.m_Y);
                Pc.m_Drop = std::clamp(1.0f - (Hips.m_Y - St.m_E.m_P.m_Y) / St.m_HipsUp, 0.0f, 1.0f);
                static const auto s_Limbs = xanimation::nervous::LimbReflexes();
                const auto D = xanimation::nervous::Choose(s_Limbs, Nv.m_Drives, Pc, Nv.m_RunningLimbs, ++Nv.m_Tick);
                if (D.m_bNew)
                {
                    std::printf("[Nervous] entity 0x%llX limbs: falling %.2f m/s, hips %.0f %% down, %s -> %s (scores:", (unsigned long long)Ent.m_Value, Pc.m_Fall, 100 * Pc.m_Drop, R.m_bSupported ? "held up" : "unsupported", xanimation::nervous::reflex_names_v[int(D.m_Chosen)]);
                    for (const auto& [Rf, Sc] : D.m_Scores) std::printf(" %s %.2f", xanimation::nervous::reflex_names_v[int(Rf)], Sc);
                    std::printf(")%c", 10); std::fflush(stdout);
                    if (D.m_Chosen == xanimation::nervous::reflex::CATCH_FALL && St.m_FallAt < 0) St.m_FallAt = St.m_Age;
                }
                Nv.m_RunningLimbs = D.m_Chosen;
                R.m_pNvLimbs = xanimation::nervous::reflex_names_v[int(D.m_Chosen)]; R.m_NvFall = Pc.m_Fall; R.m_NvDrop = Pc.m_Drop;
                const bool bCatch = D.m_Chosen == xanimation::nervous::reflex::CATCH_FALL;
                // ponytail: the ground is the plane the character stood on at the switch; a ray down (CastRayClosest, filtered off its own bodies) for ledges and slopes
                if (bCatch) St.m_Landing = xanimation::PredictLanding(Hips, Vel, St.m_E.m_P.m_Y);
                St.m_FallWPrev = St.m_FallW;
                St.m_FallW = bCatch && R.m_bReflexes ? std::min(1.0f, St.m_FallW + Dt / 0.12f) : std::max(0.0f, St.m_FallW - Dt / 0.5f);     // (fast in, slower out)
            }
            // R1 supported tracking: the joints pull toward the animation as it plays on, where the character was at the switch
            if (R.m_Tracking > 0)
            {
                std::vector<xanimation::rigid> W0, W1;
                reflex_now Rx;
                if (St.m_ReflexAt >= 0)
                {
                    const xanimation::reflex_pose_params Rp;
                    Rx.m_Threat = St.m_Threat; Rx.m_Weight = xanimation::ReflexEnvelope(Rp, St.m_Age - St.m_ReflexAt);
                    Rx.m_WeightPrev = xanimation::ReflexEnvelope(Rp, St.m_Age - St.m_ReflexAt - pGame->m_Time.m_FixedDeltaTime);
                    if (Rx.m_Weight > 0) m_pReflex = &Rx;
                    if (St.m_Age - St.m_ReflexAt > Rp.m_In + Rp.m_Hold + Rp.m_Out) St.m_ReflexAt = -1;
                }
                if (St.m_FallW > 0 && St.m_FallW >= Rx.m_Weight)      // D4: the catch outweighs the hit's protect: the hands go to the landing instead
                {
                    Rx.m_Threat = St.m_Landing; Rx.m_Weight = St.m_FallW; Rx.m_WeightPrev = St.m_FallWPrev;
                    Rx.m_Params.m_Reach = 1.0f; Rx.m_Params.m_Flinch = 0.6f;
                    m_pReflex = &Rx;
                }
                const bool bReflexOn = m_pReflex != nullptr;
                const bool bGot = AnimatedWorlds(Skin, pAnim, St.m_E, St.m_Scale, pGame->m_Time.m_FixedDeltaTime, W0, W1);
                m_pReflex = nullptr;
                // D2's measure: how close the ragdoll's hands came to the threat during the reflex, against where they were when it started
                if (St.m_pGraph && (bReflexOn || St.m_HandStart >= 0))
                {
                    float D = 1e9f;
                    for (std::size_t i = 0; i < St.m_pGraph->m_Bodies.size(); ++i)
                    {
                        const int b = St.m_pGraph->m_Bodies[i].m_Bone; bool bHand = false;
                        for (const auto& [K, Rg] : m_ReflexRigs) for (int a = 0; a < Rg.m_Rig.m_nArms; ++a) bHand = bHand || Rg.m_Rig.m_Arms[std::size_t(a)].m_Hand == b;
                        xanimation::rigid F;
                        if (bHand && xlioncore::physics::RagdollBodyFrame(*R.m_pRuntime, int(i), F)) D = std::min(D, (F.m_P - St.m_Threat).Length());
                    }
                    if (D < 1e8f)
                    {
                        if (St.m_HandStart < 0) { St.m_HandStart = D; St.m_HandMin = D; }
                        St.m_HandMin = std::min(St.m_HandMin, D);
                        if (!bReflexOn)
                        {
                            std::printf("[Reflex] entity 0x%llX: protect + flinch done; the nearer hand came from %.0f to %.0f cm of the threat%c", (unsigned long long)Ent.m_Value, 100 * St.m_HandStart, 100 * St.m_HandMin, 10);
                            std::fflush(stdout); St.m_HandStart = -1;
                        }
                    }
                }
                // D4's measure over the fall: when a hand first came within 12 cm of the ground (the stand plane), the head's lowest height and its
                // fastest speed down; logged 2.5 s after it first chose to catch. A good catch: a hand down first, the head kept up and slow
                if (St.m_FallAt >= 0 && !St.m_bFallLogged && St.m_pGraph)
                {
                    const float G = St.m_E.m_P.m_Y, Dt = std::max(pGame->m_Time.m_DeltaTime, 1e-4f);
                    auto Sk = m_Skeletons.end();
                    for (auto It = m_Skeletons.begin(); It != m_Skeletons.end(); ++It) if (It->second.m_LookAt.valid() && It->second.m_Parent.size() == W1.size()) Sk = It;
                    for (std::size_t i = 0; i < St.m_pGraph->m_Bodies.size(); ++i)
                    {
                        const int b = St.m_pGraph->m_Bodies[i].m_Bone; bool bHand = false;
                        for (const auto& [K, Rg] : m_ReflexRigs) for (int a = 0; a < Rg.m_Rig.m_nArms; ++a) bHand = bHand || Rg.m_Rig.m_Arms[std::size_t(a)].m_Hand == b;
                        const bool bHead = Sk != m_Skeletons.end() && Sk->second.m_LookAt.m_Head == b;
                        xanimation::rigid F;
                        if (!(bHand || bHead) || !xlioncore::physics::RagdollBodyFrame(*R.m_pRuntime, int(i), F)) continue;
                        if (bHand && St.m_HandDownAt < 0 && F.m_P.m_Y - G < 0.12f) St.m_HandDownAt = St.m_Age;
                        if (bHand) St.m_HandLow = std::min(St.m_HandLow, F.m_P.m_Y - G);
                        if (bHead)
                        {
                            if (St.m_HeadLow < 1e8f) St.m_HeadFall = std::max(St.m_HeadFall, (St.m_PrevHead.m_Y - F.m_P.m_Y) / Dt);
                            St.m_HeadLow = std::min(St.m_HeadLow, F.m_P.m_Y - G); St.m_PrevHead = F.m_P;
                        }
                    }
                    if (St.m_Age - St.m_FallAt > 2.5f)       // (the whole fall, also for a body that catches more than once)
                    {
                        std::printf("[Fall] entity 0x%llX: catch %s; a hand reached the ground %s%.2f s after the fall started; the hands' lowest %.0f cm; the head's lowest %.0f cm, fastest down %.2f m/s%c", (unsigned long long)Ent.m_Value
                            , R.m_bReflexes ? "on" : "off", St.m_HandDownAt < 0 ? "never " : "", St.m_HandDownAt < 0 ? 0.0f : St.m_HandDownAt - St.m_FallAt, 100 * St.m_HandLow, 100 * St.m_HeadLow, St.m_HeadFall, 10);
                        std::fflush(stdout); St.m_bFallLogged = true;
                    }
                }
                if (bGot)
                {
                    xlioncore::physics::ragdoll_tracking P; P.m_Gain = R.m_Tracking; P.m_bSupported = R.m_bSupported; P.m_FeedForward = R.m_FeedForward; P.m_GravityFF = R.m_bSupported ? 0.0f : R.m_Balance;
                    if (R.m_bSupported) St.m_Supported += pGame->m_Time.m_DeltaTime;
                    float EW = 0;
                    const float E = xlioncore::physics::TrackRagdoll(*R.m_pRuntime, W0, W1, pGame->m_Time.m_FixedDeltaTime, P, &EW);
                    if (E >= 0) { St.m_TrackSum += double(E) * E; St.m_TrackSumW += double(EW) * EW; ++St.m_nTrack; St.m_TrackMax = std::max(St.m_TrackMax, E); }

                    // R3b' return to cheap (contract 2.4 demotion): settled = joints within ReturnBelow of the animation for 0.3 s (from 0.3 s
                    // after the hit), or 2 s at most; then the Animation poses it again. The pose jump at the switch is measured (AN10)
                    if (R.m_bReaction && St.m_Age - St.m_HitAt > 0.3f)
                    {
                        if (E >= 0) { St.m_LastE = E; if (E < 0.95f * St.m_MinE) { St.m_MinE = E; St.m_MinAt = St.m_Age; } }
                        // settled: under ReturnBelow for 0.3 s, or on a plateau - no new low (by 5 %) for 0.3 s and near the lowest it reached since
                        // the hit (a running clip never gets under ReturnBelow: its tracking floor is higher)
                        const bool bPlateau = St.m_MinE < 1e8f && St.m_Age - St.m_MinAt > 0.3f && St.m_LastE < 1.25f * St.m_MinE;     // (a measured error first)
                        St.m_Settled = St.m_LastE < R.m_ReturnBelow ? St.m_Settled + pGame->m_Time.m_DeltaTime : bPlateau ? 0.3f : 0.0f;
                        if (St.m_Settled >= 0.3f || St.m_Age - St.m_HitAt > 2.0f)
                        {
                            float Pop = 0;
                            if (St.m_pGraph)
                                for (std::size_t i = 0; i < St.m_pGraph->m_Bodies.size(); ++i)
                                {
                                    xanimation::rigid F;
                                    const auto b = std::size_t(St.m_pGraph->m_Bodies[i].m_Bone);
                                    if (b < W1.size() && xlioncore::physics::RagdollBodyFrame(*R.m_pRuntime, int(i), F))
                                        Pop = std::max(Pop, (F.m_P - xanimation::BodyFromBone(W1[b], St.m_pGraph->m_Bodies[i]).m_P).Length());
                                }
                            std::printf("[Ragdoll] entity 0x%llX returns to its animation %.2f s after its last new hit (%zu hits; %s; joints %.1f deg from it): pose jump %.1f cm\n"
                                , (unsigned long long)Ent.m_Value, St.m_Age - St.m_HitAt, St.m_Hits.size(), St.m_Settled >= 0.3f ? "settled" : "time limit", St.m_LastE, 100 * Pop);
                            std::fflush(stdout);
                            St.m_bForced = St.m_Settled < 0.3f;
                            // (the ledger line now, in the frame of the return: the deactivation comes a frame later, and a game stopped in between lost it)
                            std::printf("[Assist] entity 0x%llX: root held by the animation %.2f s of %.2f s active%s%c", (unsigned long long)Ent.m_Value, St.m_Supported, St.m_Age
                                , St.m_bForced ? "; sent back to its animation by the time limit (not settled)" : "", 10);
                            std::fflush(stdout); St.m_bAssistLogged = true;
                            HandOverToAnimation(Ent, R, Skin, T, pParent, W1);
                            R.m_bActive = false; R.m_bReaction = false;
                        }
                    }
                }
            }

            // KD5 get-up (contract v2 7.8; physical/xanimation_get_up.h): fallen (hips more than half down) and still for 0.5 s, with a get-up clip
            // for how it lies: the character is laid along the body, the ragdoll hands over to the clip (blended) and it plays once
            if (pAnim && pParent == nullptr && (!R.m_GetUpFaceDown.empty() || !R.m_GetUpFaceUp.empty()) && R.m_NvDrop > 0.5f && St.m_pSD && St.m_pSD->m_LookAt.valid())
            {
                St.m_DownStill = R.m_NvFall < 0.15f ? St.m_DownStill + pGame->m_Time.m_DeltaTime : 0.0f;
                if (St.m_DownStill >= 0.5f && TryGetUp(Ent, R, Skin, *pAnim, T, St)) return;
            }
            // the entity follows the hips across the ground
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
            if (S.m_KE > Bound && St.m_WorstEnergy <= 1.0f && R.m_Tracking <= 0)      // (a tracking ragdoll is powered: the bound is a limp one's)      // the moment it first makes energy: when, how much, which body
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
                if (St.m_nTrack) std::printf("[Ragdoll] tracking entity 0x%llX: gain %.1f, %s, joints from the animation rms %.2f deg, worst %.2f deg; bodies in the world rms %.2f deg (%d frames)\n"
                    , (unsigned long long)Ent.m_Value, R.m_Tracking, R.m_bSupported ? "supported" : "unsupported", std::sqrt(St.m_TrackSum / St.m_nTrack), St.m_TrackMax, std::sqrt(St.m_TrackSumW / St.m_nTrack), St.m_nTrack);
                std::fflush(stdout);
                St.m_bReported = true;
            }
        }

        // KD5: the clip for how the body lies, the character laid along it (its hips under the ragdoll's, turned so the clip's first pose's
        // hips->head lies along the body's), the ragdoll handed over to the clip and the clip requested once. False: no clip for it (it stays down)
        bool TryGetUp( const xecs::component::entity& Ent, ragdoll& R, const skinned_mesh& Skin, animation& A, xlioncore::transform& T, entity_state& St ) noexcept
        {
            using namespace xlioncore::resources;
            const auto& S = *St.m_pSD; const int nB = int(S.m_Parent.size());
            std::vector<xanimation::rigid> Now(static_cast<std::size_t>(nB)), Fill(static_cast<std::size_t>(nB), St.m_E);      // (not size_t(nB): that declares a function)
            xlioncore::physics::PoseRagdoll(*R.m_pRuntime, S.view(), 1.0f, Fill, Now);
            const int Hb = S.m_Warp.m_Hips >= 0 ? S.m_Warp.m_Hips : 0, Hd = S.m_LookAt.m_Head;
            // the body's front: the rest forward carried by the hips' rotation since the rest
            xanimation::details::rigid_worlds Wr; xanimation::details::Worlds(S.view(), S.m_Rest, Wr);
            const auto Front = Now[std::size_t(Hb)].m_Q * (Wr.m_R[std::size_t(Hb)].ConjugateCopy() * S.m_LookAt.m_Forward);
            const bool bDown = xanimation::IsFaceDown(Front);
            const std::string& Name = bDown ? R.m_GetUpFaceDown : R.m_GetUpFaceUp;
            if (Name.empty()) return false;
            const std::uint64_t Pkg = A.m_Package.m_Instance.m_Value;
            auto* pPkg = Pkg ? static_cast<xanim_package::anim_package*>(GetResource(anim_package_type_guid_v.m_Value, Pkg)) : nullptr;
            const int iClip = pPkg ? pPkg->findClipIndex(xstrtool::CRC32(Name)) : -1;
            if (iClip < 0) { std::printf("[GetUp] entity 0x%llX: no clip '%s' in its Animation's package: it stays down%c", (unsigned long long)Ent.m_Value, Name.c_str(), 10); std::fflush(stdout); St.m_DownStill = -1e9f; return false; }
            xanimation::pose P0, Scratch;
            if (!xanimation::SampleClip(S, *pPkg, iClip, 0.0f, false, P0, Scratch)) return false;
            xanimation::details::rigid_worlds Wc; xanimation::details::Worlds(S.view(), P0, Wc);
            const float Yaw = xanimation::GetUpYaw(Now[std::size_t(Hd)].m_P - Now[std::size_t(Hb)].m_P, Wc.m_P[std::size_t(Hd)] - Wc.m_P[std::size_t(Hb)]);
            const xmath::fquat Q(0.0f, std::sin(0.5f * Yaw), 0.0f, std::cos(0.5f * Yaw));
            const auto ClipHips = Q * (Wc.m_P[std::size_t(Hb)] * St.m_Scale);
            T.m_Rotation = Q;
            T.m_Position = xmath::fvec3(Now[std::size_t(Hb)].m_P.m_X - ClipHips.m_X, St.m_E.m_P.m_Y, Now[std::size_t(Hb)].m_P.m_Z - ClipHips.m_Z);
            std::printf("[GetUp] entity 0x%llX: down %.1f s, face %s: '%s' (clip %d), turned %.0f deg%c", (unsigned long long)Ent.m_Value, St.m_DownStill, bDown ? "down" : "up", Name.c_str(), iClip, Yaw * 57.29578f, 10);
            std::fflush(stdout);
            std::vector<xanimation::rigid> W0, W1;
            if (AnimatedWorlds(Skin, &A, xanimation::rigid{ T.m_Position, T.m_Rotation }, St.m_Scale, 0, W0, W1)) HandOverToAnimation(Ent, R, Skin, T, nullptr, W1);
            A.m_ClipName = Name; A.m_RequestHash = 0; A.m_bOneShot = true; A.m_bPlaying = true;
            R.m_bActive = false; R.m_bReaction = false; R.m_bGettingUp = true;
            return true;
        }

        // The entity's world frame and scale: its Transform for a root, the pose the transform system left in its Parent for a child
        static xanimation::rigid WorldFrame( const xlioncore::transform& T, const xecs::component::parent* pParent, float& Scale ) noexcept
        {
            if (pParent == nullptr) { Scale = T.m_Scale.m_X; return { T.m_Position, T.m_Rotation }; }
            Scale = pParent->m_WorldScale[0];
            return { xmath::fvec3(pParent->m_WorldPosition[0], pParent->m_WorldPosition[1], pParent->m_WorldPosition[2])
                   , xmath::fquat(pParent->m_WorldRotation[0], pParent->m_WorldRotation[1], pParent->m_WorldRotation[2], pParent->m_WorldRotation[3]) };
        }

        // R3b' return to cheap without a pop: the ragdoll's last two captures as local poses become the Animation's starting point, which then
        // eases into the clip over 0.3 s (the poser's inertialization: anim_poser::HandOver). Animated: this frame's animated world rigid frames
        // (the bones the ragdoll does not own keep them).
        void HandOverToAnimation( const xecs::component::entity& Ent, const ragdoll& R, const skinned_mesh& Skin, const xlioncore::transform& T, const xecs::component::parent* pParent, std::span<const xanimation::rigid> Animated ) noexcept
        {
            using namespace xlioncore::resources;
            auto* pAnimSys = findSystem<animation_system>();
            auto* pGeom = Skin.m_Geom.empty() ? nullptr : static_cast<xgeom_skin::xgpu::geom*>(GetResource(geom_skin_type_guid_v.m_Value, Skin.m_Geom.m_Instance.m_Value));
            auto* pSkel = pGeom ? static_cast<const xskeleton::skeleton*>(GetResource(skeleton_type_guid_v.m_Value, pGeom->getSkeletonRef().m_Instance.m_Value)) : nullptr;
            const auto* pGame = xlioncore::game::From(getGameMgr());
            if (pAnimSys == nullptr || pSkel == nullptr || pGame == nullptr || R.m_pRuntime == nullptr) return;
            auto& S = m_Skeletons[pSkel];
            const auto n = std::size_t(S.m_Parent.size());
            if (n == 0 || Animated.size() != n) return;
            float Es = 1; const auto E = WorldFrame(T, pParent, Es);
            std::vector<xanimation::rigid> W(n); std::vector<float> Sc(n, Es);
            xanimation::pose Cur = m_P1, Prev = m_P1;       // (the scales: the animation's)
            if (Cur.count() != int(n)) return;
            xlioncore::physics::PoseRagdoll(*R.m_pRuntime, S.view(), 1.0f, Animated, W); xanimation::LocalFromRigidWorlds(S.view(), W, Sc, E, Es, Cur);
            xlioncore::physics::PoseRagdoll(*R.m_pRuntime, S.view(), 0.0f, Animated, W); xanimation::LocalFromRigidWorlds(S.view(), W, Sc, E, Es, Prev);
            pAnimSys->m_Poser.HandOver(Ent.m_Value, pSkel, Cur, Prev, pGame->m_Time.m_FixedDeltaTime, 0.3f);
        }

        // The animated world rigid bone frames of a character played at E (its frame) and Scale: now (W1) and Dtb back (W0); the rest pose
        // when it has no clip. False: its skin or skeleton is not loaded.
        bool AnimatedWorlds( const skinned_mesh& Skin, const animation* pAnim, const xanimation::rigid& E, float Scale, float Dtb
                           , std::vector<xanimation::rigid>& W0, std::vector<xanimation::rigid>& W1 ) noexcept
        {
            using namespace xlioncore::resources;
            auto* pGeom = Skin.m_Geom.empty() ? nullptr : static_cast<xgeom_skin::xgpu::geom*>(GetResource(geom_skin_type_guid_v.m_Value, Skin.m_Geom.m_Instance.m_Value));
            auto* pSkel = pGeom ? static_cast<const xskeleton::skeleton*>(GetResource(skeleton_type_guid_v.m_Value, pGeom->getSkeletonRef().m_Instance.m_Value)) : nullptr;
            if (pSkel == nullptr) return false;
            const int nB = int(pSkel->getBones().size());
            auto& S = m_Skeletons[pSkel];
            if (int(S.m_Parent.size()) != nB) S = xanimation::MakeSkeleton(*pSkel);
            auto Sample = [&](float Back, xanimation::pose& P)
            {
                const std::uint64_t Pkg = pAnim == nullptr ? 0 : pAnim->m_Package.empty() ? pGeom->getAnimationRef().m_Instance.m_Value : pAnim->m_Package.m_Instance.m_Value;
                auto* pPkg = Pkg ? static_cast<xanim_package::anim_package*>(GetResource(anim_package_type_guid_v.m_Value, Pkg)) : nullptr;
                if (pPkg && pAnim->m_Clip >= 0 && pAnim->m_Clip < int(pPkg->getClips().size()) && int(pPkg->getBoneNameHashes().size()) == nB)
                {
                    const auto& C = pPkg->getClips()[std::size_t(pAnim->m_Clip)];
                    const float Len = C.m_FPS > 0 ? float(C.m_nFrames) / float(C.m_FPS) : 0.0f;
                    float t = pAnim->m_Time - (pAnim->m_bPlaying ? Back * pAnim->m_Speed : 0.0f);
                    if (Len > 0) t = pAnim->m_bLoop ? t - Len * std::floor(t / Len) : std::clamp(t, 0.0f, Len);
                    xanimation::pose Scratch;
                    if (xanimation::SampleClip(S, *pPkg, pAnim->m_Clip, t, pAnim->m_bMirror, P, Scratch)) return;
                }
                P = S.m_Rest;
            };
            Sample(Dtb, m_P0); Sample(0, m_P1);
            if (m_pReflex && m_pReflex->m_Weight > 0)     // D2: a reflex pose on the animation the ragdoll tracks (protect, flinch)
            {
                auto& Rig = m_ReflexRigs[pSkel];
                if (!Rig.m_bInit) { Rig.m_bInit = true; if (const auto Ro = pSkel->getRoles(); int(Ro.size()) == nB) Rig.m_Rig = xanimation::FindReflexRig(S.view(), Ro); }
                const auto ThreatModel = E.Inverse().m_Q * (m_pReflex->m_Threat - E.m_P) * (1.0f / std::max(Scale, 1e-6f));
                xanimation::look_at_state L0, L1;
                xanimation::ApplyReflexPose(S.view(), Rig.m_Rig, S.m_LookAt, m_P0, ThreatModel, m_pReflex->m_WeightPrev, m_pReflex->m_Params, L0);
                xanimation::ApplyReflexPose(S.view(), Rig.m_Rig, S.m_LookAt, m_P1, ThreatModel, m_pReflex->m_Weight, m_pReflex->m_Params, L1);
            }
            const auto n = static_cast<std::size_t>(nB);
            W0.resize(n); W1.resize(n); std::vector<float> Sc(n);
            xanimation::RigidBoneWorlds(S.view(), m_P0, E, Scale, W0, Sc);
            xanimation::RigidBoneWorlds(S.view(), m_P1, E, Scale, W1, Sc);
            return true;
        }

        bool Activate( const xecs::component::entity& Ent, ragdoll& R, const skinned_mesh& Skin, const animation* pAnim, const xlioncore::transform& T, const xecs::component::parent* pParent, const xlioncore::game& Game ) noexcept
        {
            using namespace xlioncore::resources;
            auto Fail = [&](const char* Why) { std::printf("[Ragdoll] entity 0x%llX not activated: %s\n", (unsigned long long)Ent.m_Value, Why); std::fflush(stdout); return false; };
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
            std::vector<xanimation::rigid> W0, W1;
            float Scale = 1; const auto EW = WorldFrame(T, pParent, Scale);       // (a child (a player's Lilly on its capsule): its world pose)
            if (!AnimatedWorlds(Skin, pAnim, EW, Scale, Dt, W0, W1)) return Fail("its skeleton is not loaded");

            xlioncore::physics::ragdoll_params Params;
            Params.m_Size = 2 * (pRag->m_Capsule.m_HalfSegment + pRag->m_Capsule.m_Radius) * Scale;
            if (Params.m_Size <= 0.01f) Params.m_Size = 1.7f;
            if (const char* p = std::getenv("XLION_RAGDOLL_SUBSTEPS"); p && *p) Params.m_SubSteps = std::max(1, std::atoi(p));      // (a narrowing switch)
            std::string Note;
            R.m_pRuntime = xlioncore::physics::ActivateRagdoll(getGameMgr(), GE.m_G.m_Graph, GE.m_Rest, W0, W1, Dt, GE.m_G.m_Filter, Params, &Note);
            if (R.m_pRuntime == nullptr) return Fail("the game has no physics world");
            if (pParent) xlioncore::physics::RagdollIgnoreEntity(getGameMgr(), *R.m_pRuntime, pParent->m_Value.m_Value);     // (its carrier's body: the same character)

            auto& St = m_State[Ent.m_Value]; St = {};
            const auto Hips = xlioncore::physics::RagdollRootBody(*R.m_pRuntime, 1.0f);
            St.m_Offset = T.m_Position - Hips.m_P;
            St.m_E = EW; St.m_Scale = Scale; St.m_pGraph = &GE.m_G.m_Graph; St.m_pSD = &S;
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
        std::pair<int, view>                            m_PrimitivesInScene{ -1, view::SCENE };     // (the frame and view whose RenderScene drew the Primitives: Collect leaves them out)
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
            if (m_PrimitivesInScene.second == View && m_Frame - m_PrimitivesInScene.first <= 3) return;     // (the scene of this view draws them: RenderScene, a Collect or two ago - views collect in turn)
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
            {   // the Primitives, as Meshes of this scene (one depth with everything else); Collect leaves them out of the old overlay for this view
                xecs::query::instance Query;
                Query.m_Must.AddFromComponents<xlioncore::transform, primitive>();
                LeaveOut(Query, View);
                auto S = Search(Query);
                Foreach(S, [&](const xecs::component::entity& Ent, const xlioncore::transform& Local, const primitive& Prim, const xecs::component::parent* pParent, const xlioncore::render_transform* pRender) noexcept
                {
                    if (Hidden(Ent.m_Value)) return;
                    const auto T = xlioncore::WorldOf(Local, pParent, pRender, Interpolate);
                    if (!T.m_Scale.isFinite() || T.m_Scale.m_X == 0.0f || T.m_Scale.m_Y == 0.0f || T.m_Scale.m_Z == 0.0f) return;     // (the same guard as Collect)
                    R.AddPrimitive(Prim.m_Shape, Prim.m_Color, T, Ent.m_Value);
                });
                m_PrimitivesInScene = { m_Frame, View };
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