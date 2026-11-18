#ifndef XLIONRENDER_ANIM_POSE_H
#define XLIONRENDER_ANIM_POSE_H
#pragma once
// K0c: the animation pose stack, out of the scene renderer: the library's pose of a skeleton (a clip's sample, the transition when its source
// changes, stride/orientation warping, the ground fit, foot locking, look-at) and the per-entity state it keeps. The Animation system owns
// one per world and poses every animated character once per update (its game time); the scene renderer is one too, for what the system
// did not pose (no update ran, a preview). Moved verbatim from xlionrender_scene_renderer.h; only the clock is new (m_GameDt).
#include "dependencies/xAnimation/source/xanimation.h"
#include "dependencies/xAnimation/source/blend/xanimation_overlay.h"
#include "dependencies/xAnimation/source/motion/xanimation_springs.h"
#include "dependencies/xAnimation/source/motion/xanimation_lean.h"
#include "dependencies/xAnimation/source/motion/xanimation_reflex_pose.h"
#include "dependencies/xAnimation/source/motion/xanimation_bone_ops.h"
#include "dependencies/xAnimation/source/motion/xanimation_stabilize.h"
#include "dependencies/xAnimation/source/ik/xanimation_spline_ik.h"
#include "dependencies/xstrtool/source/xstrtool.h"       // (the overlay's FromBone by name)
#include "dependencies/xAnimation/source/adapters/xanimation_resources.h"
#include "xlionrender_scene_components.h"      // (the Animation component the pose reads)
#include <array>
#include <chrono>
#include <span>
#include <unordered_map>
#include <vector>

namespace xlionrender
{
    // KU3: the fix-up stages of a character's pose, and the built-in order they run in (each rule of it was a measured bug when broken:
    // warping before the feet, the feet before the upper body, lean before the aim, stabilize after it, offset before copy, springs last)
    enum class fixup : std::uint8_t { OVERLAY, WARP, GROUND, FOOT_LOCK, LEAN, AIM, STABILIZE, BONE_OPS, SPRINGS, TRAIL };
    inline constexpr std::array<const char*, 10> fixup_names_v{ "Overlay", "Warp", "Ground", "FootLock", "Lean", "Aim", "Stabilize", "BoneOps", "Springs", "Trail" };
    // (TRAIL after the feet, before the upper body: a body laid along its path, then aimed and leaned like any other)
    inline constexpr std::array<fixup, 10> builtin_stack_v{ fixup::OVERLAY, fixup::WARP, fixup::GROUND, fixup::FOOT_LOCK, fixup::TRAIL, fixup::LEAN, fixup::AIM, fixup::STABILIZE, fixup::BONE_OPS, fixup::SPRINGS };

    struct anim_poser
    {
        std::vector<fixup>                              m_Stack;                        // KU3: the order the fix-ups run in (empty: builtin_stack_v)
        xanimation::pose                                m_AnimPose;                     // (scratch: the library's local pose of the character being added)
        xanimation::pose                                m_AnimScratch;                  // (scratch: the unmirrored sample of a mirrored clip)
        const xanimation::pose*                         m_pAnimUsed = nullptr;          // (the pose PoseWithLibrary posed: m_AnimPose or the rest)
        std::unordered_map<const void*, xanimation::skeleton_data> m_AnimSkeleton;      // per skeleton resource: the library's copy of its hierarchy and rest pose
        float                                           m_GameDt = -1;                  // >= 0: the per-entity state advances by this (the update's game time), not the wall clock

        // What an update posed: per entity, its bone worlds (model space) and local pose, stamped with the update that made them
        // ponytail: a map keyed by entity, swept of entities not posed for 256 updates; a slot in the component if levels churn entities
        struct posed { std::uint32_t m_Update = 0; std::vector<xmath::fmat4> m_Worlds; xanimation::pose m_Local; const xskeleton::skeleton* m_pSkel = nullptr; };
        std::unordered_map<std::uint64_t, posed>        m_Posed;
        std::uint32_t                                   m_Update = 0;
        const posed* Posed(std::uint64_t Entity) const noexcept
        {
            const auto It = m_Posed.find(Entity);
            return It != m_Posed.end() && It->second.m_Update == m_Update ? &It->second : nullptr;
        }
        void SweepPosed(void) noexcept
        {
            if ((m_Update & 255) != 0) return;
            for (auto It = m_Posed.begin(); It != m_Posed.end(); ) if (m_Update - It->second.m_Update > 256) It = m_Posed.erase(It); else ++It;
        }

        // A skinned character's pose from its Animation component as the renderer always made it: the clip at its time, the transition, the
        // Animation system's warp and ground casts, foot locking and look-at in the frame L2W. Entity 0: no per-entity state (no transitions).
        void PoseEntity(const xskeleton::skeleton& Skel, const xanim_package::anim_package* pPkg, bool bClip, const animation* pAnim, const xmath::fmat4& L2W
                       , std::uint64_t Entity, std::vector<xmath::fmat4>& Out) noexcept
        {
            xanimation::warp_params Wp;          // K2: the Animation system's warping this frame
            if (bClip) { Wp.m_Stride = pAnim->m_WarpStride; Wp.m_Angle = pAnim->m_WarpAngle; Wp.m_Dir = pAnim->m_WarpDir; }
            std::array<xanimation::ground_contact, 4> Ground; std::size_t nGround = 0;        // K3: the Animation system's casts this frame
            if (bClip && pAnim->m_bFootIK) for (; nGround < pAnim->m_nGround && nGround < Ground.size(); ++nGround)
                Ground[nGround] = { pAnim->m_Ground[nGround].m_bHit, pAnim->m_Ground[nGround].m_Height, pAnim->m_Ground[nGround].m_Normal };
            m_pOverlayAnim = bClip ? pAnim : nullptr; m_pSpringAnim = pAnim;
            m_Stack.clear();
            if (pAnim && !pAnim->m_Stack.empty() && Entity)       // KU3: its own order (parsed when it changes; what it breaks said once)
            {
                auto& Tr = m_Tracks[Entity];
                if (Tr.m_StackText != pAnim->m_Stack) { Tr.m_StackText = pAnim->m_Stack; Tr.m_Stack = ParseStack(pAnim->m_Stack, Entity); }
                m_Stack = Tr.m_Stack;
            }
            PoseWithLibrary(Skel, bClip ? pPkg : nullptr, bClip ? pAnim->m_Clip : 0, bClip ? pAnim->m_Time : 0.0f, bClip && pAnim->m_bMirror, Out
                           , pAnim ? Entity : 0, pAnim ? pAnim->m_BlendTime : 0.0f, &Wp, std::span<const xanimation::ground_contact>(Ground.data(), nGround)
                           , pAnim ? pAnim->m_MatchJumps : 0u, bClip && pAnim->m_bFootLock ? &L2W : nullptr
                           , &L2W, pAnim ? pAnim->m_LookAt : xmath::fvec3(0, 0, 0), pAnim ? pAnim->m_LookAtWeight : 0.0f);
        }
        float Elapsed(std::chrono::steady_clock::time_point Last, std::chrono::steady_clock::time_point Now) const noexcept
            { return m_GameDt >= 0 ? std::min(m_GameDt, 0.1f) : std::clamp(std::chrono::duration<float>(Now - Last).count(), 0.0f, 0.1f); }

        // The library's pose of a skeleton at a clip's time (no clip: its rest pose) as bone worlds. The skeleton's parent array is copied once
        // per skeleton resource (re-checked against the resource each time: a reload with other parents rebuilds it).
        // Entity / BlendTime (K1): when what poses the entity changes (another clip or package, mirrored or not, the rest), the new pose starts
        // from the one shown and eases into the new source over BlendTime (inertialization: the old clip is not sampled any more; 0: a cut).
        void PoseWithLibrary(const xskeleton::skeleton& Skel, const xanim_package::anim_package* pPkg, int iClip, float Time, bool bMirror, std::vector<xmath::fmat4>& Out, std::uint64_t Entity = 0, float BlendTime = 0, const xanimation::warp_params* pWarp = nullptr
                            , std::span<const xanimation::ground_contact> Ground = {}, std::uint32_t Jumps = 0      // (Jumps: motion matching's jump count - a jump within a clip is a new source too)
                            , const xmath::fmat4* pLockL2W = nullptr                                                  // (set: foot locking, in this world frame)
                            , const xmath::fmat4* pL2W = nullptr, const xmath::fvec3& LookAt = {}, float LookAtWeight = 0)    // (look-at: a world point, in the frame pL2W)
        {
            auto& S = m_AnimSkeleton[&Skel];
            const auto Bones = Skel.getBones();
            bool bSame = S.m_Parent.size() == Bones.size();
            for (std::size_t i = 0; bSame && i < Bones.size(); ++i) bSame = S.m_Parent[i] == Bones[i].m_iParent;
            if (!bSame) S = xanimation::MakeSkeleton(Skel);
            Out.resize(Bones.size());
            const bool bClip = pPkg && xanimation::SampleClip(S, *pPkg, iClip, Time, bMirror, m_AnimPose, m_AnimScratch);
            if (Entity == 0) { xanimation::ComputeBoneWorlds(S.view(), bClip ? m_AnimPose : S.m_Rest, Out); m_pAnimUsed = bClip ? &m_AnimPose : &S.m_Rest; return; }
            if (!bClip) m_AnimPose = S.m_Rest;
            Transition(Entity, bClip ? (std::uint64_t(reinterpret_cast<std::uintptr_t>(pPkg)) * 1000003u + std::uint64_t(iClip) * 2 + (bMirror ? 1 : 0) + std::uint64_t(Jumps) * 0x9E3779B97F4A7C15ull) : 1, &Skel, BlendTime, m_AnimPose);
            // KU3 step 1 (areas/animation/plans/fixup_stack_as_data.md): the fix-ups run from an ordered stack - the built-in one is exactly the order
            // they always ran in (equivalence first); a recipe resource will give another order and several instances (step 2)
            for (const fixup F : m_Stack.empty() ? std::span<const fixup>(builtin_stack_v) : std::span<const fixup>(m_Stack))
            {
                switch (F)
                {
                case fixup::OVERLAY:
                            if (bClip && m_pOverlayAnim) ApplyOverlaySlot(Skel, S, *pPkg, Entity, *m_pOverlayAnim);     // (S1: after the blend, before the fix-ups: warping and the feet see the overlaid pose)
                    break;
                case fixup::WARP:
                            if (bClip && pWarp && pWarp->active()) xanimation::WarpPose(S.view(), S.m_Warp, m_AnimPose, *pWarp);     // (after the blend: the blend's poses stay the clip's own)
                    break;
                case fixup::GROUND:
                            if (bClip && Ground.size() == S.m_Warp.m_Legs.size() && !Ground.empty()) xanimation::FitGround(S.view(), S.m_Warp, m_AnimPose, Ground, S.m_FootHeights);
                    break;
                case fixup::FOOT_LOCK:
                            if (bClip && pLockL2W)                     // foot locking: last, on the pose as drawn (per entity: its pins and when it was last drawn)
                            {
                                auto& Tr = m_Tracks[Entity];
                                const auto Now = std::chrono::steady_clock::now();
                                const float Dt = Tr.m_LockT.time_since_epoch().count() ? Elapsed(Tr.m_LockT, Now) : 0.0f;
                                Tr.m_LockT = Now;
                                // (through a transition's blend a planted foot is held farther: the blend moves the pose's foot between two clips while the body
                                // goes on, and at the usual 25 cm the lock let go mid-blend - SoccerAnim's feet slid ~125 cm/s around matching jumps)
                                xanimation::foot_lock_params LockPm; if (Tr.m_In.m_bActive) LockPm.m_MaxPull = 0.6f;
                                if (m_pSpringAnim && m_pSpringAnim->m_bFootSteps) LockPm.m_StepAt = 0.12f;      // (KS5: model units; Lilly's metres)
                                // (the clip's own Footstep/Contact events say which feet are down: a foot locks when it lands, however fast the body goes - by
                                // world speed alone a foot that landed while the body moved never locked; clips without such events keep the speed test)
                                std::array<std::uint8_t, 8> Down{}; std::size_t nDown = 0;
                                // (a paused clip is a standing pose: its events say nothing about which feet are down now - the speed test decides; KS5's turn on
                                // the spot found it: paused on a frame between steps, no foot ever locked)
                                if (!bMirror && (m_pSpringAnim == nullptr || m_pSpringAnim->m_bPlaying))
                                {
                                    const auto Ev = pPkg->getClipEvents(iClip);
                                    bool bAny = false;
                                    for (const auto& E : Ev) bAny |= E.m_Kind == xanim_package::event_kind::FOOTSTEP || E.m_Kind == xanim_package::event_kind::CONTACT;
                                    if (bAny) for (; nDown < S.m_Warp.m_Legs.size() && nDown < Down.size(); ++nDown)
                                        for (const auto& E : Ev)
                                            if (E.m_iBone == S.m_Warp.m_Legs[nDown].m_Foot && (E.m_Kind == xanim_package::event_kind::FOOTSTEP || E.m_Kind == xanim_package::event_kind::CONTACT)
                                                && Time >= E.m_Time && Time <= E.m_Time + E.m_Duration) Down[nDown] = 1;
                                }
                                const int nStepsBefore = Tr.m_Lock.m_nSteps;
                                xanimation::LockFeet(S.view(), S.m_Warp, m_AnimPose, pLockL2W->ExtractPosition(), pLockL2W->ExtractRotation().NormalizeSafeCopy(), pLockL2W->ExtractScale(), S.m_FootHeights, Dt, Tr.m_Lock, LockPm
                                                    , std::span<const std::uint8_t>(Down.data(), nDown));
                                if (Tr.m_Lock.m_nSteps != nStepsBefore) { std::printf("[AnimSteps] entity %llu: step %d%c", (unsigned long long)Entity, Tr.m_Lock.m_nSteps, 10); std::fflush(stdout); }
                            }
                    break;
                case fixup::LEAN:
                            if (m_pSpringAnim && pL2W && m_pSpringAnim->m_LeanAmount > 0) ApplyLeanTo(Skel, S, Entity, *m_pSpringAnim, *pL2W);
                            if (m_pSpringAnim && pL2W && m_pSpringAnim->m_FlinchSerial) ApplyFlinch(Skel, S, Entity, *m_pSpringAnim, *pL2W);       // (KN3: with the lean, the same chain)
                    break;
                case fixup::AIM:
                            if (bClip && pL2W && S.m_LookAt.valid())  // look-at: after the feet (the chain is above the hips), toward a world point in the entity's frame
                            {
                                auto& Tr = m_Tracks[Entity];
                                if (LookAtWeight > 0 || Tr.m_Look.m_W > 0)
                                {
                                    const auto Now = std::chrono::steady_clock::now();
                                    const float Dt = Tr.m_LookT.time_since_epoch().count() ? Elapsed(Tr.m_LookT, Now) : 0.0f;
                                    Tr.m_LookT = Now;
                                    const auto Inv = pL2W->InverseSRT();
                                    // S2: onto another bone (the spine turns it) and/or from angles; the head's chain otherwise
                                    const auto* pChain = &S.m_LookAt;
                                    if (m_pSpringAnim && !m_pSpringAnim->m_AimBone.empty())
                                    {
                                        if (Tr.m_AimBone != m_pSpringAnim->m_AimBone || (Tr.m_AimAxis - m_pSpringAnim->m_AimAxis).LengthSq() > 0)
                                        {
                                            Tr.m_AimBone = m_pSpringAnim->m_AimBone; Tr.m_AimAxis = m_pSpringAnim->m_AimAxis;
                                            const auto Ro = Skel.getRoles();
                                            Tr.m_AimChain = xanimation::FindAimChain(S.view(), Ro, S.m_LookAt, S.m_Rest, Skel.findBoneIndex(xstrtool::CRC32(Tr.m_AimBone)), Tr.m_AimAxis);
                                            std::printf("[AnimAim] entity %llu: aiming %s (axis %.2f %.2f %.2f) with %zu spine bones%s%c", (unsigned long long)Entity, Tr.m_AimBone.c_str(), Tr.m_AimAxis.m_X, Tr.m_AimAxis.m_Y, Tr.m_AimAxis.m_Z
                                                , Tr.m_AimChain.m_Bones.size(), Tr.m_AimChain.valid() ? "" : ": NO such bone or no spine - the head aims instead", 10);
                                            std::fflush(stdout);
                                        }
                                        if (Tr.m_AimChain.valid()) pChain = &Tr.m_AimChain;
                                    }
                                    if (m_pSpringAnim && m_pSpringAnim->m_bLookAngles)
                                        xanimation::LookAtAngles(S.view(), *pChain, m_AnimPose, m_pSpringAnim->m_LookYaw * 0.017453293f, m_pSpringAnim->m_LookPitch * 0.017453293f, LookAtWeight, Dt, Tr.m_Look);
                                    else
                                        xanimation::LookAt(S.view(), *pChain, m_AnimPose, Inv * LookAt, LookAtWeight, Dt, Tr.m_Look);
                                    if (pChain != &S.m_LookAt && !m_pSpringAnim->m_bLookAngles)      // (S2's measure: how far the aimed axis is from the target, every 120 frames)
                                    {
                                        xanimation::details::rigid_worlds Wa; xanimation::details::Worlds(S.view(), m_AnimPose, Wa);
                                        const auto Ax = (Wa.m_R[std::size_t(pChain->m_Head)] * Tr.m_AimAxis).NormalizeSafeCopy(), To = (Inv * LookAt - Wa.m_P[std::size_t(pChain->m_Head)]).NormalizeSafeCopy();
                                        Tr.m_AimErr = std::max(Tr.m_AimErr, std::acos(std::clamp(Ax.Dot(To), -1.0f, 1.0f)) * 57.29578f);
                                        if (++Tr.m_AimN % 120 == 0) { std::printf("[AnimAim] entity %llu: %s points %.1f deg from the target at most over 120 frames%c", (unsigned long long)Entity, Tr.m_AimBone.c_str(), Tr.m_AimErr, 10); std::fflush(stdout); Tr.m_AimErr = 0; }
                                    }
                                }
                            }
                    break;
                case fixup::STABILIZE:
                            if (m_pSpringAnim && pL2W && m_pSpringAnim->m_StabilizeWeight > 0 && S.m_LookAt.valid()) ApplyStabilize(S, Entity, *m_pSpringAnim, *pL2W);     // (KN4: after the aim: it eases what the look-at made)
                    break;
                case fixup::BONE_OPS:
                            if (m_pSpringAnim && (!m_pSpringAnim->m_OffsetBone.empty() || !m_pSpringAnim->m_CopyTo.empty())) ApplyBoneOps(Skel, S, Entity, *m_pSpringAnim);
                    break;
                case fixup::TRAIL:
                            if (m_pSpringAnim && pL2W && !m_pSpringAnim->m_TrailFrom.empty() && !m_pSpringAnim->m_TrailTo.empty()) ApplyTrail(Skel, S, Entity, *m_pSpringAnim, *pL2W);
                    break;
                case fixup::SPRINGS:
                            if (m_pSpringAnim && pL2W && !m_pSpringAnim->m_SpringFrom.empty()) ApplySprings(Skel, S, Entity, *m_pSpringAnim, *pL2W);
                    break;
                }
            }
            m_pOverlayAnim = nullptr; m_Stack.clear();
            m_pSpringAnim = nullptr;
            xanimation::ComputeBoneWorlds(S.view(), m_AnimPose, Out); m_pAnimUsed = &m_AnimPose;
        }

        // N1 (superset: KINEMATION's sway, Unreal's Spring Bone / AnimDynamics): the Animation's spring chain - SpringFrom and its first-child
        // descendants lag behind the body's accelerations and swing back (motion/xanimation_springs.h). Last in the stack: it moves what was posed.
        // ponytail: one chain per character; a list of chains when a character needs hair and a tail at once
        const animation*                            m_pSpringAnim = nullptr;
        void ApplySprings(const xskeleton::skeleton& Skel, const xanimation::skeleton_data& S, std::uint64_t Entity, const animation& A, const xmath::fmat4& L2W)
        {
            auto& Tr = m_Tracks[Entity];
            if (Tr.m_SpringFrom != A.m_SpringFrom)
            {
                Tr.m_SpringFrom = A.m_SpringFrom; Tr.m_SpringChain.clear(); Tr.m_SpringTip = -1; Tr.m_Spring = {};
                int B = BoneOrSay(Skel, A.m_SpringFrom, Entity, "Springs.From");
                const auto Par = S.view().m_Parent;
                while (B >= 0 && Tr.m_SpringChain.size() < 32)
                {
                    Tr.m_SpringChain.push_back(B);
                    int Child = -1; for (int c = B + 1; c < int(Par.size()); ++c) if (Par[std::size_t(c)] == B) { Child = c; break; }
                    B = Child;
                }
                if (Tr.m_SpringChain.size() >= 2) { Tr.m_SpringTip = Tr.m_SpringChain.back(); Tr.m_SpringChain.pop_back(); }
            }
            if (Tr.m_SpringChain.empty()) return;
            xanimation::spring_params P; P.m_Hz = A.m_SpringHz; P.m_Damping = A.m_SpringDamping; P.m_Weight = A.m_SpringWeight; P.m_MaxAngle = A.m_SpringMaxAngle * 0.017453293f;
            P.m_Gravity = xmath::fvec3(0, -A.m_SpringDroop, 0);
            const xanimation::rigid E{ L2W.ExtractPosition(), L2W.ExtractRotation().NormalizeSafeCopy() };
            std::vector<xmath::fquat> Before; for (int b : Tr.m_SpringChain) Before.push_back(m_AnimPose.m_Rotation[std::size_t(b)]);
            xanimation::UpdateSprings(S.view(), Tr.m_SpringChain, Tr.m_SpringTip, m_AnimPose, E, L2W.ExtractScale().m_X, m_GameDt >= 0 ? std::min(m_GameDt, 0.1f) : 1.0f / 60.0f, P, Tr.m_Spring);
            for (std::size_t k = 0; k < Before.size(); ++k)
                Tr.m_SpringMax = std::max(Tr.m_SpringMax, 2 * std::acos(std::min(1.0f, std::abs(Before[k].Dot(m_AnimPose.m_Rotation[std::size_t(Tr.m_SpringChain[k])])))) * 57.29578f);
            if (++Tr.m_SpringN % 120 == 0)
            {
                std::printf("[AnimSprings] entity %llu: chain of %zu bones from %s, largest swing over the last 120 frames %.1f deg%c", (unsigned long long)Entity, Tr.m_SpringChain.size() + 1, Tr.m_SpringFrom.c_str(), Tr.m_SpringMax, 10);
                std::fflush(stdout); Tr.m_SpringMax = 0;
            }
        }

        // KU3: an order by names ("Overlay, Warp, ..." - commas or spaces, any case); an unknown name and an order that breaks a rule of
        // builtin_stack_v are said ([AnimStack]); the order runs as given (a person may try it)
        static std::vector<fixup> ParseStack(const std::string& Text, std::uint64_t Entity) noexcept
        {
            std::vector<fixup> Out; std::string Word;
            auto Lower = [](std::string S) { for (auto& c : S) c = char(std::tolower(static_cast<unsigned char>(c))); return S; };
            auto Take = [&]
            {
                if (Word.empty()) return;
                int k = -1; for (int i = 0; i < int(fixup_names_v.size()); ++i) if (Lower(fixup_names_v[std::size_t(i)]) == Lower(Word)) k = i;
                if (k < 0) std::printf("[AnimStack] entity %llu: '%s' is no fix-up (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s): left out%c", (unsigned long long)Entity, Word.c_str()
                    , fixup_names_v[0], fixup_names_v[1], fixup_names_v[2], fixup_names_v[3], fixup_names_v[4], fixup_names_v[5], fixup_names_v[6], fixup_names_v[7], fixup_names_v[8], fixup_names_v[9], 10);
                else Out.push_back(fixup(k));
                Word.clear();
            };
            for (char c : Text) { if (c == ',' || c == ' ' || c == ';') Take(); else Word += c; }
            Take();
            struct rule { fixup m_A, m_B; const char* m_pWhy; };
            static constexpr rule Rules[] =
            { { fixup::WARP, fixup::FOOT_LOCK, "the warp moves the feet after they were pinned: they slide" }
            , { fixup::GROUND, fixup::FOOT_LOCK, "the ground fit moves the feet after they were pinned" }
            , { fixup::FOOT_LOCK, fixup::LEAN, "the lean is undone by the feet's reach" }
            , { fixup::LEAN, fixup::AIM, "the lean turns an aimed bone off its target" }
            , { fixup::AIM, fixup::STABILIZE, "the aim moves the head after it was steadied" }
            , { fixup::BONE_OPS, fixup::SPRINGS, "the springs swing before the copy and offset: a held bone does not follow the swing" }
            , { fixup::TRAIL, fixup::AIM, "the trail lays the body after it was aimed: the aim is lost" } };
            auto At = [&](fixup F) { for (std::size_t i = 0; i < Out.size(); ++i) if (Out[i] == F) return int(i); return -1; };
            for (const auto& R : Rules)
                if (const int a = At(R.m_A), b = At(R.m_B); a >= 0 && b >= 0 && a > b)
                    std::printf("[AnimStack] entity %llu: %s after %s - %s%c", (unsigned long long)Entity, fixup_names_v[std::size_t(R.m_A)], fixup_names_v[std::size_t(R.m_B)], R.m_pWhy, 10);
            std::printf("[AnimStack] entity %llu: %zu fix-ups%c", (unsigned long long)Entity, Out.size(), 10); std::fflush(stdout);
            return Out;
        }

        // KUX (R66): a bone named in an Animation field, or a line saying it is not one (the runtime skeleton keeps only the names' hashes:
        // the editor's skeleton lists the names). Called when the field changes, so once per change.
        static int BoneOrSay(const xskeleton::skeleton& Skel, const std::string& Name, std::uint64_t Entity, const char* pField) noexcept
        {
            const int B = Skel.findBoneIndex(xstrtool::CRC32(Name));
            if (B < 0) { std::printf("[AnimBones] entity %llu: %s '%s' is not a bone of its skeleton (names are exact, case included)%c", (unsigned long long)Entity, pField, Name.c_str(), 10); std::fflush(stdout); }
            return B;
        }

        // S3 (superset: KINEMATION's copy/modify bone, Unreal's Copy Bone / Transform Bone; motion/xanimation_bone_ops.h): the Animation's offset on
        // one bone, then its copy (To follows From, with the offset it had when the copy started); [AnimCopy] reports how well To follows
        void ApplyBoneOps(const xskeleton::skeleton& Skel, const xanimation::skeleton_data& S, std::uint64_t Entity, const animation& A)
        {
            auto& Tr = m_Tracks[Entity];
            if (Tr.m_OffsetBone != A.m_OffsetBone) { Tr.m_OffsetBone = A.m_OffsetBone; Tr.m_iOffset = A.m_OffsetBone.empty() ? -1 : BoneOrSay(Skel, A.m_OffsetBone, Entity, "Offset.Bone"); }
            if (Tr.m_iOffset >= 0)
            {
                const auto& R = A.m_OffsetRotation;
                const auto Q = xanimation::details::AxisAngle(xmath::fvec3(0, 0, 1), R.m_Z * 0.017453293f) * xanimation::details::AxisAngle(xmath::fvec3(0, 1, 0), R.m_Y * 0.017453293f) * xanimation::details::AxisAngle(xmath::fvec3(1, 0, 0), R.m_X * 0.017453293f);
                xanimation::OffsetBone(S.view(), m_AnimPose, Tr.m_iOffset, Q, A.m_OffsetPosition, A.m_OffsetWeight, A.m_bOffsetModel, A.m_OffsetSpread);
            }
            if (Tr.m_CopyFrom != A.m_CopyFrom || Tr.m_CopyTo != A.m_CopyTo)
            {
                Tr.m_CopyFrom = A.m_CopyFrom; Tr.m_CopyTo = A.m_CopyTo; Tr.m_bCopyOffset = false;
                Tr.m_iCopyFrom = A.m_CopyFrom.empty() ? -1 : Skel.findBoneIndex(xstrtool::CRC32(A.m_CopyFrom));
                Tr.m_iCopyTo = A.m_CopyTo.empty() ? -1 : Skel.findBoneIndex(xstrtool::CRC32(A.m_CopyTo));
                const auto Ro = Skel.getRoles(); Tr.m_CopyArms = int(Ro.size()) == S.view().count() ? xanimation::FindReflexRig(S.view(), Ro) : xanimation::reflex_rig{};
                if (!A.m_CopyTo.empty())
                {
                    bool bArm = false; for (int a = 0; a < Tr.m_CopyArms.m_nArms; ++a) bArm = bArm || Tr.m_CopyArms.m_Arms[std::size_t(a)].m_Hand == Tr.m_iCopyTo;
                    std::printf("[AnimCopy] entity %llu: %s follows %s%s%c", (unsigned long long)Entity, A.m_CopyTo.c_str(), A.m_CopyFrom.c_str()
                        , Tr.m_iCopyFrom < 0 || Tr.m_iCopyTo < 0 ? ": NO such bone - off" : bArm ? " (a hand: its arm reaches)" : " (moved alone)", 10);
                    std::fflush(stdout);
                }
            }
            if (Tr.m_iCopyFrom < 0 || Tr.m_iCopyTo < 0) return;
            if (!Tr.m_bCopyOffset) { Tr.m_CopyOffset = A.m_bCopyKeepOffset ? xanimation::CopyOffset(S.view(), m_AnimPose, Tr.m_iCopyFrom, Tr.m_iCopyTo) : xanimation::rigid{}; Tr.m_bCopyOffset = true; }
            const xanimation::reflex_arm* pArm = nullptr;
            for (int a = 0; a < Tr.m_CopyArms.m_nArms; ++a) if (Tr.m_CopyArms.m_Arms[std::size_t(a)].m_Hand == Tr.m_iCopyTo) pArm = &Tr.m_CopyArms.m_Arms[std::size_t(a)];
            xanimation::bone_copy_params P; P.m_bRotation = A.m_bCopyRotation; P.m_bPosition = A.m_bCopyPosition; P.m_Weight = A.m_CopyWeight;
            Tr.m_CopyMiss = std::max(Tr.m_CopyMiss, xanimation::CopyBone(S.view(), m_AnimPose, Tr.m_iCopyFrom, Tr.m_iCopyTo, Tr.m_CopyOffset, P, pArm));
            {   // the measure: where To is against where the copy puts it
                xanimation::details::rigid_worlds W; xanimation::details::Worlds(S.view(), m_AnimPose, W);
                const auto Want = W.m_P[std::size_t(Tr.m_iCopyFrom)] + W.m_R[std::size_t(Tr.m_iCopyFrom)] * Tr.m_CopyOffset.m_P;
                Tr.m_CopyOff = std::max(Tr.m_CopyOff, (W.m_P[std::size_t(Tr.m_iCopyTo)] - Want).Length());
            }
            if (++Tr.m_CopyN % 120 == 0)
            {
                std::printf("[AnimCopy] entity %llu: %s from where it is kept at most %.1f cm over 120 frames (out of reach by %.1f cm)%c", (unsigned long long)Entity, Tr.m_CopyTo.c_str(), 100 * Tr.m_CopyOff, 100 * Tr.m_CopyMiss, 10);
                std::fflush(stdout); Tr.m_CopyOff = Tr.m_CopyMiss = 0;
            }
        }

        // KU2 (superset: Unreal's Spline IK; a snake's slither, a tail following its body; ik/xanimation_spline_ik.h): the chain From..To laid along
        // the path From has travelled in the world (the trail starts as the pose's own joints: a still body stays as posed); [AnimTrail] says
        // the chain once and, every 120 frames, how far the joints are from the path
        void ApplyTrail(const xskeleton::skeleton& Skel, const xanimation::skeleton_data& S, std::uint64_t Entity, const animation& A, const xmath::fmat4& L2W)
        {
            auto& Tr = m_Tracks[Entity];
            if (Tr.m_TrailFrom != A.m_TrailFrom || Tr.m_TrailTo != A.m_TrailTo)
            {
                Tr.m_TrailFrom = A.m_TrailFrom; Tr.m_TrailTo = A.m_TrailTo; Tr.m_TrailChain.clear(); Tr.m_Trail.m_Points.clear();
                const int F = BoneOrSay(Skel, A.m_TrailFrom, Entity, "Trail.From"), T = BoneOrSay(Skel, A.m_TrailTo, Entity, "Trail.To");
                for (int b = T; b >= 0 && F >= 0; b = S.m_Parent[std::size_t(b)]) { Tr.m_TrailChain.insert(Tr.m_TrailChain.begin(), b); if (b == F) break; }
                if (Tr.m_TrailChain.empty() || Tr.m_TrailChain.front() != F) Tr.m_TrailChain.clear();
                std::printf("[AnimTrail] entity %llu: %s..%s, %zu bones%s%c", (unsigned long long)Entity, A.m_TrailFrom.c_str(), A.m_TrailTo.c_str(), Tr.m_TrailChain.size()
                    , F >= 0 && T >= 0 && Tr.m_TrailChain.size() < 2 ? ": NO chain (From must be an ancestor of To) - off" : "", 10);
                std::fflush(stdout);
            }
            if (Tr.m_TrailChain.size() < 2) return;
            xanimation::details::rigid_worlds W; xanimation::details::Worlds(S.view(), m_AnimPose, W);
            float Len = 0;
            for (std::size_t i = 1; i < Tr.m_TrailChain.size(); ++i) Len += (W.m_P[std::size_t(Tr.m_TrailChain[i])] - W.m_P[std::size_t(Tr.m_TrailChain[i - 1])]).Length();
            const float Spacing = Len / float(Tr.m_TrailChain.size() - 1) * 0.5f;
            const auto Head = L2W * W.m_P[std::size_t(Tr.m_TrailChain.front())];
            if (!Tr.m_Trail.m_Points.empty() && (Head - Tr.m_Trail.m_Points.front()).Length() > Len) Tr.m_Trail.m_Points.clear();     // (a teleport: the body as posed where it lands, not stretched back to where it was)
            if (Tr.m_Trail.m_Points.empty()) for (const int b : Tr.m_TrailChain) Tr.m_Trail.m_Points.push_back(L2W * W.m_P[std::size_t(b)]);
            xanimation::UpdateTrail(Tr.m_Trail, Head, Spacing, Len * 1.25f);
            const auto Inv = L2W.InverseSRT();
            std::vector<xmath::fvec3> Ctrl; xanimation::TrailControl(Tr.m_Trail, Head, Ctrl);
            for (auto& C : Ctrl) C = Inv * C;
            xanimation::SolveSplineIK(S.view(), Tr.m_TrailChain, m_AnimPose, Ctrl);
            if (++Tr.m_TrailN % 120 == 0)       // the measure: the joints against the trail's polyline, in the model
            {
                xanimation::details::Worlds(S.view(), m_AnimPose, W);
                float Worst = 0;
                for (const int b : Tr.m_TrailChain)
                {
                    float Best = 1e9f;
                    for (std::size_t k = 0; k + 1 < Ctrl.size(); ++k)
                    {
                        const auto D = Ctrl[k + 1] - Ctrl[k]; const float t = std::clamp((W.m_P[std::size_t(b)] - Ctrl[k]).Dot(D) / std::max(D.LengthSq(), 1e-12f), 0.0f, 1.0f);
                        Best = std::min(Best, (Ctrl[k] + D * t - W.m_P[std::size_t(b)]).Length());
                    }
                    Worst = std::max(Worst, Best);
                }
                std::printf("[AnimTrail] entity %llu: joints off the path at most %.1f cm of a %.2f body (%zu trail points)%c", (unsigned long long)Entity, 100 * Worst, Len, Tr.m_Trail.m_Points.size(), 10);
                std::fflush(stdout);
            }
        }

        // KN4 (superset: KINEMATION's stabilize bone): the head kept steady in the world (motion/xanimation_stabilize.h); [AnimStabilize] compares the
        // head's turn rate in the world before and after, every 120 frames
        void ApplyStabilize(const xanimation::skeleton_data& S, std::uint64_t Entity, const animation& A, const xmath::fmat4& L2W)
        {
            auto& Tr = m_Tracks[Entity];
            const float Dt = m_GameDt >= 0 ? std::min(m_GameDt, 0.1f) : 1.0f / 60.0f;
            const auto E = L2W.ExtractRotation().NormalizeSafeCopy();
            const auto HeadWorld = [&]() { xanimation::details::rigid_worlds W; xanimation::details::Worlds(S.view(), m_AnimPose, W); return (E * W.m_R[std::size_t(S.m_LookAt.m_Head)]).NormalizeCopy(); };
            const auto In = HeadWorld();
            xanimation::stabilize_params P; P.m_Weight = A.m_StabilizeWeight; P.m_Smooth = A.m_StabilizeSmooth;
            xanimation::StabilizeHead(S.view(), S.m_LookAt, m_AnimPose, E, Dt, Tr.m_Stab, P);
            const auto Out = HeadWorld();
            const auto Turn = [](const xmath::fquat& a, const xmath::fquat& b) { return 2 * std::acos(std::min(1.0f, std::fabs(a.Dot(b)))); };
            if (Tr.m_bStabPrev && Dt > 0) { const double a = Turn(Tr.m_StabPrevIn, In) / Dt, b = Turn(Tr.m_StabPrevOut, Out) / Dt; Tr.m_StabIn += a * a; Tr.m_StabOut += b * b; ++Tr.m_StabN; }
            Tr.m_StabPrevIn = In; Tr.m_StabPrevOut = Out; Tr.m_bStabPrev = true;
            if (Tr.m_StabN >= 120)
            {
                std::printf("[AnimStabilize] entity %llu: the head's turn rate in the world rms %.1f deg/s from the animation, %.1f deg/s stabilized%c", (unsigned long long)Entity
                    , std::sqrt(Tr.m_StabIn / Tr.m_StabN) * 57.29578, std::sqrt(Tr.m_StabOut / Tr.m_StabN) * 57.29578, 10);
                std::fflush(stdout); Tr.m_StabIn = Tr.m_StabOut = 0; Tr.m_StabN = 0;
            }
        }

        // KN3 (superset: KINEMATION's procedural recoil): a small jolt tilts the spine with the push - fast in, a hold, a slower return (the reflex
        // envelope), on the lean's chain; no physics
        void ApplyFlinch(const xskeleton::skeleton& Skel, const xanimation::skeleton_data& S, std::uint64_t Entity, const animation& A, const xmath::fmat4& L2W)
        {
            auto& Tr = m_Tracks[Entity];
            if (Tr.m_pLeanSkel != &Skel) { Tr.m_pLeanSkel = &Skel; const auto Ro = Skel.getRoles(); Tr.m_LeanChain = int(Ro.size()) == S.view().count() ? xanimation::FindLeanChain(S.view(), Ro) : std::vector<int>{}; Tr.m_Lean = {}; Tr.m_bLeanPos = false; }
            if (Tr.m_LeanChain.empty()) return;
            if (Tr.m_FlinchSerial != A.m_FlinchSerial) { Tr.m_FlinchSerial = A.m_FlinchSerial; Tr.m_FlinchAge = 0; }
            xanimation::reflex_pose_params P; P.m_In = 0.06f; P.m_Hold = 0.05f; P.m_Out = 0.35f;
            const float W = xanimation::ReflexEnvelope(P, Tr.m_FlinchAge);
            Tr.m_FlinchAge += m_GameDt >= 0 ? std::min(m_GameDt, 0.1f) : 1.0f / 60.0f;
            if (W <= 0) return;
            const auto Q = L2W.ExtractRotation().NormalizeSafeCopy();
            const auto Dir = Q.ConjugateCopy() * A.m_FlinchDir;          // (model space)
            const auto Axis = xmath::fvec3(0, 1, 0).Cross(Dir).NormalizeSafeCopy();
            xanimation::ApplyLean(S.view(), Tr.m_LeanChain, m_AnimPose, Axis, A.m_FlinchAngle * W);
        }

        // N2 (superset: KINEMATION's pivot/lean, Unreal's lean): the spine leans into the body's acceleration (motion/xanimation_lean.h), the body's
        // velocity from its frame's position update to update, in its own frame
        void ApplyLeanTo(const xskeleton::skeleton& Skel, const xanimation::skeleton_data& S, std::uint64_t Entity, const animation& A, const xmath::fmat4& L2W)
        {
            auto& Tr = m_Tracks[Entity];
            if (Tr.m_pLeanSkel != &Skel) { Tr.m_pLeanSkel = &Skel; const auto Ro = Skel.getRoles(); Tr.m_LeanChain = int(Ro.size()) == S.view().count() ? xanimation::FindLeanChain(S.view(), Ro) : std::vector<int>{}; Tr.m_Lean = {}; Tr.m_bLeanPos = false; }
            if (Tr.m_LeanChain.empty()) return;
            const float Dt = m_GameDt >= 0 ? std::min(m_GameDt, 0.1f) : 1.0f / 60.0f;
            const auto Pos = L2W.ExtractPosition(); const auto Q = L2W.ExtractRotation().NormalizeSafeCopy();
            const float Sc = std::max(L2W.ExtractScale().m_X, 1e-6f);
            auto V = Tr.m_bLeanPos && Dt > 0 ? (Pos - Tr.m_LeanPrevPos) * (1.0f / Dt) : xmath::fvec3(0, 0, 0);
            if (V.Length() > 20.0f * Sc) V = Tr.m_Lean.m_PrevVel * Sc;      // (a teleport: no acceleration)
            Tr.m_LeanPrevPos = Pos; Tr.m_bLeanPos = true;
            xanimation::lean_params P; P.m_Amount = A.m_LeanAmount; P.m_MaxAngle = A.m_LeanMaxAngle * 0.017453293f; P.m_Smooth = A.m_LeanSmooth;
            xmath::fvec3 Axis;
            const float Angle = xanimation::UpdateLean(Tr.m_Lean, Q.ConjugateCopy() * V * (1.0f / Sc), Dt, P, Axis);      // (model space: the character's own units, g in them)
            xanimation::ApplyLean(S.view(), Tr.m_LeanChain, m_AnimPose, Axis, Angle);
            Tr.m_LeanMax = std::max(Tr.m_LeanMax, Angle * 57.29578f); Tr.m_LeanSeen.push_back(Angle * 57.29578f);
            if (++Tr.m_LeanN % 120 == 0)
            {
                std::printf("[AnimLean] entity %llu: %zu spine bones, largest lean over the last 120 frames %.1f deg (median %.1f)%c", (unsigned long long)Entity, Tr.m_LeanChain.size(), Tr.m_LeanMax, (std::nth_element(Tr.m_LeanSeen.begin(), Tr.m_LeanSeen.begin() + std::ptrdiff_t(Tr.m_LeanSeen.size() / 2), Tr.m_LeanSeen.end()), Tr.m_LeanSeen[Tr.m_LeanSeen.size() / 2]), 10);
                std::fflush(stdout); Tr.m_LeanMax = 0; Tr.m_LeanSeen.clear();
            }
        }

        // S1 (KINEMATION superset): the Animation's overlay slot on m_AnimPose - a new Play starts it, Clip -1 eases it out; its mask is FromBone and
        // everything under it, faded in over the first two bones (blend/xanimation_overlay.h, the authoring layers' own mix)
        const animation*                            m_pOverlayAnim = nullptr;
        xanimation::pose                            m_OvPose, m_OvRef;
        void ApplyOverlaySlot(const xskeleton::skeleton& Skel, const xanimation::skeleton_data& S, const xanim_package::anim_package& Pkg, std::uint64_t Entity, const animation& A)
        {
            auto& Tr = m_Tracks[Entity];
            if (A.m_OverlayPlay != Tr.m_OvPlay)
            {
                Tr.m_OvPlay = A.m_OverlayPlay;
                if (A.m_OverlayClip >= 0)
                {
                    Tr.m_Ov.Start(); Tr.m_OvClip = A.m_OverlayClip;
                    std::printf("[AnimOverlay] entity %llu: clip %d over the pose from %s (%s%s)%c", (unsigned long long)Entity, A.m_OverlayClip, A.m_OverlayFrom.empty() ? "the root" : A.m_OverlayFrom.c_str(), A.m_bOverlayAdditive ? "additive" : "override", A.m_bOverlayLoop ? ", loop" : "", 10);
                    std::fflush(stdout);
                }
            }
            const bool bWasActive = Tr.m_Ov.m_bActive;
            if (A.m_OverlayClip < 0 || A.m_OverlayClip != Tr.m_OvClip) Tr.m_Ov.Stop();
            if (!Tr.m_Ov.m_bActive || Tr.m_OvClip < 0 || Tr.m_OvClip >= int(Pkg.getClips().size())) return;
            const auto& C = Pkg.getClips()[std::size_t(Tr.m_OvClip)];
            const float Len = C.m_FPS > 0 ? float(C.m_nFrames) / float(C.m_FPS) : 0.0f;
            xanimation::overlay_params P; P.m_BlendIn = A.m_OverlayBlendIn; P.m_BlendOut = A.m_OverlayBlendOut; P.m_Speed = A.m_OverlaySpeed; P.m_bLoop = A.m_bOverlayLoop;
            const float W = xanimation::AdvanceOverlay(Tr.m_Ov, P, Len, m_GameDt >= 0 ? std::min(m_GameDt, 0.1f) : 1.0f / 60.0f);
            Tr.m_OvPeak = Tr.m_Ov.m_Age <= 0.05f ? W : std::max(Tr.m_OvPeak, W);
            if (bWasActive && !Tr.m_Ov.m_bActive) { std::printf("[AnimOverlay] entity %llu: clip %d done after %.2f s (peak weight %.2f)%c", (unsigned long long)Entity, Tr.m_OvClip, Tr.m_Ov.m_Age, Tr.m_OvPeak, 10); std::fflush(stdout); }
            if (W <= 0) return;
            if (Tr.m_OvMask.size() != std::size_t(S.view().count()) || Tr.m_OvFrom != A.m_OverlayFrom)
            {
                Tr.m_OvFrom = A.m_OverlayFrom;
                xanimation::mask_rules M;
                const int B = A.m_OverlayFrom.empty() ? 0 : BoneOrSay(Skel, A.m_OverlayFrom, Entity, "Overlay.FromBone");
                if (B >= 0) M.m_Entries.push_back({ B, 1.0f, true, A.m_OverlayFrom.empty() ? 0 : 2 });
                Tr.m_OvMask = xanimation::FlattenMask(S.view(), M);
            }
            if (!xanimation::SampleClip(S, Pkg, Tr.m_OvClip, Tr.m_Ov.m_Time, false, m_OvPose, m_AnimScratch)) return;
            if (A.m_bOverlayAdditive && !xanimation::SampleClip(S, Pkg, Tr.m_OvClip, 0.0f, false, m_OvRef, m_AnimScratch)) return;
            xanimation::ApplyOverlay(m_AnimPose, m_OvPose, Tr.m_OvMask, W, A.m_bOverlayAdditive, &m_OvRef);
        }

        // the poses an entity showed on its last two frames, what posed it, and its transition (K1)
        struct anim_track
        {
            std::uint64_t                           m_Source = 0;
            const void*                             m_pSkel  = nullptr;
            xanimation::pose                        m_Cur, m_Prev;
            std::chrono::steady_clock::time_point   m_T;
            xanimation::inertializer                m_In;
            xanimation::foot_lock_state             m_Lock;         // (FootLock: the feet's pins)
            std::chrono::steady_clock::time_point   m_LockT{};
            xanimation::look_at_state               m_Look;         // (LookAt: the aim's eased angles and weight)
            std::chrono::steady_clock::time_point   m_LookT{};
            float                                   m_ForceBlend = 0, m_StartDt = 0;    // (HandOver: the next transition's blend and its poses' spacing)
            xanimation::overlay_state               m_Ov;           // S1: the overlay slot (its time and weight), what started it, its mask
            std::uint32_t                           m_OvPlay = 0;
            int                                     m_OvClip = -1;
            std::string                             m_OvFrom;
            std::vector<float>                      m_OvMask;
            float                                   m_OvPeak = 0;
            xanimation::spring_state                m_Spring;       // N1: the secondary-motion chain (its bones by name, its points)
            std::string                             m_SpringFrom;
            std::vector<int>                        m_SpringChain;
            int                                     m_SpringTip = -1;
            float                                   m_SpringMax = 0; int m_SpringN = 0;
            xanimation::lean_state                  m_Lean;         // N2: the eased acceleration, the body's last position, the spine chain (for its skeleton)
            xmath::fvec3                            m_LeanPrevPos{ 0, 0, 0 }; bool m_bLeanPos = false;
            const xskeleton::skeleton*              m_pLeanSkel = nullptr;
            std::vector<int>                        m_LeanChain;
            float                                   m_LeanMax = 0; int m_LeanN = 0; std::vector<float> m_LeanSeen;
            std::uint32_t                           m_FlinchSerial = 0; float m_FlinchAge = 1e9f;      // KN3
            std::string                             m_AimBone; xmath::fvec3 m_AimAxis{ 0, 0, 0 };     // S2: the aim chain for this bone and axis (rebuilt when they change)
            xanimation::look_at_chain               m_AimChain;
            int                                     m_AimN = 0; float m_AimErr = 0;
            std::string                             m_CopyFrom, m_CopyTo; int m_iCopyFrom = -1, m_iCopyTo = -1; bool m_bCopyOffset = false;     // S3
            xanimation::rigid                       m_CopyOffset; xanimation::reflex_rig m_CopyArms;
            int                                     m_CopyN = 0; float m_CopyMiss = 0, m_CopyOff = 0;
            std::string                             m_OffsetBone; int m_iOffset = -1;
            std::string                             m_TrailFrom, m_TrailTo; std::vector<int> m_TrailChain; xanimation::spline_trail m_Trail; int m_TrailN = 0;      // KU2
            std::string                             m_StackText; std::vector<fixup> m_Stack;       // KU3: its own order, as parsed
            xanimation::stabilize_state             m_Stab;         // KN4: the eased head; the measure (the head's turn rate before and after, squared, summed)
            xmath::fquat                            m_StabPrevIn, m_StabPrevOut; bool m_bStabPrev = false; double m_StabIn = 0, m_StabOut = 0; int m_StabN = 0;
        };
        std::unordered_map<std::uint64_t, anim_track>   m_Tracks;

        void Transition(std::uint64_t Entity, std::uint64_t Source, const void* pSkel, float BlendTime, xanimation::pose& P)
        {
            if (m_Tracks.size() > 4096) m_Tracks.clear();       // ponytail: entities that went away are forgotten in bulk; a per-frame sweep if levels spawn that many
            auto& Tr = m_Tracks[Entity];
            const auto Now = std::chrono::steady_clock::now();
            const bool bHad = Tr.m_pSkel == pSkel && Tr.m_Cur.count() == P.count();
            const float Dt = bHad ? Elapsed(Tr.m_T, Now) : 0.0f;
            if (Tr.m_ForceBlend > 0.0f) BlendTime = Tr.m_ForceBlend;          // (a hand-over's own blend: HandOver)
            if (bHad && Tr.m_Source != Source && BlendTime > 0.0f)
            {
                Tr.m_In.Start(Tr.m_Cur, Tr.m_Prev, P, P, Tr.m_StartDt > 0.0f ? Tr.m_StartDt : Dt, BlendTime * 0.25f);     // (a half-life of a quarter of the blend time: ~94 % gone by its end)
                static const bool bLog = [] { const char* e = std::getenv("XLION_ANIM_EVENTS"); return e && *e == '1'; }();
                if (bLog) { float W = 0; for (const auto& R : Tr.m_In.m_RotX) W = std::max(W, std::sqrt(R.Dot(R))); std::printf("[AnimBlend] entity %llu: a new source eases in over %.2f s (largest turn to absorb %.1f deg)\n", (unsigned long long)Entity, BlendTime, W * 57.29578f); }
            }
            const bool bHandOver = Tr.m_ForceBlend > 0.0f && bHad;
            float RawJump = 0;
            if (bHandOver) for (int i = 0; i < P.count(); ++i) RawJump = std::max(RawJump, 2 * std::acos(std::min(1.0f, std::abs(P.m_Rotation[i].Dot(Tr.m_Cur.m_Rotation[i])))));
            if (bHad) Tr.m_In.Apply(P, Dt); else Tr.m_In.m_bActive = false;
            if (bHandOver)      // (R3b' AN10: how far the first animated pose is from the last ragdoll pose, with the blend and without it)
            {
                float Jump = 0;
                for (int i = 0; i < P.count(); ++i) Jump = std::max(Jump, 2 * std::acos(std::min(1.0f, std::abs(P.m_Rotation[i].Dot(Tr.m_Cur.m_Rotation[i])))));
                std::printf("[AnimBlend] entity %llu: hand-over from a ragdoll, largest bone turn at the switch %.1f deg (without the blend %.1f deg)\n", (unsigned long long)Entity, Jump * 57.29578f, RawJump * 57.29578f);
                std::fflush(stdout);
            }
            if (bHad) Tr.m_Prev = Tr.m_Cur; else Tr.m_Prev = P;
            Tr.m_Cur = P; Tr.m_Source = Source; Tr.m_pSkel = pSkel; Tr.m_T = Now; Tr.m_ForceBlend = 0.0f; Tr.m_StartDt = 0.0f;
        }

        // R3b' return to cheap: the poses another source showed last (a ragdoll's two last captures: Cur, Prev, StepDt apart, local) become
        // the entity's track, so its next animated pose starts from them with their velocity and eases in over Blend (inertialization)
        void HandOver(std::uint64_t Entity, const void* pSkel, const xanimation::pose& Cur, const xanimation::pose& Prev, float StepDt, float Blend)
        {
            auto& Tr = m_Tracks[Entity];
            Tr.m_Cur = Cur; Tr.m_Prev = Prev; Tr.m_pSkel = pSkel; Tr.m_Source = 0;     // (0: no clip's source: the next pose is a new source)
            Tr.m_T = std::chrono::steady_clock::now(); Tr.m_ForceBlend = Blend; Tr.m_StartDt = StepDt;
        }
    };
}
#endif
