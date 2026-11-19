#ifndef XLIONRENDER_RECOVERY_VS_H
#define XLIONRENDER_RECOVERY_VS_H
#pragma once
// Push recovery's plan and Visual Script (R84 first slice; the decision document's gates 1-2): the system calls RecoveryPlan instead of
// xanimation::PlanRecovery. With no switch set it is PlanRecovery, nothing else.
//   XLION_RECOVERY_CAPTURE=<file>      every plan's inputs and result appended as a typed fixture line (the gate-1 harness replays them)
//   XLION_RECOVERY_EVALUATOR=shadow    both evaluate, the hand function actuates; a difference is logged with the field that differs
//   XLION_RECOVERY_EVALUATOR=generated the generated function (Visual Script) actuates; the hand one is not called
// The generated evaluator exists only when xvisual_script.plugin's generated header is there (v0 binds at build time); without it the
// switch says so once and the hand function runs.
#include "xlionrender_env.h"
#include "dependencies/xAnimation/source/motion/xanimation_recovery.h"
#include <bit>
#include <cstdio>
#include <cstring>
#if __has_include("plugins/xvisual_script.plugin/generated/PlanRecoveryVS.h")
    #include "plugins/xvisual_script.plugin/generated/PlanRecoveryVS.h"
    #define XLIONRENDER_RECOVERY_VS 1
#endif

namespace xlionrender
{
    namespace recovery_vs
    {
        enum class evaluator : int { HAND, SHADOW, GENERATED };
        inline evaluator Evaluator() noexcept
        {
            static const evaluator E = []
            {
                const char* e = Env("XLION_RECOVERY_EVALUATOR");
                const evaluator Want = !e ? evaluator::HAND : !std::strcmp(e, "shadow") ? evaluator::SHADOW : !std::strcmp(e, "generated") ? evaluator::GENERATED : evaluator::HAND;
#ifndef XLIONRENDER_RECOVERY_VS
                if (Want != evaluator::HAND) { std::printf("[VisualScript] recovery: '%s' asked, but this build has no generated PlanRecoveryVS: the hand function runs\n", e); std::fflush(stdout); return evaluator::HAND; }
#endif
                if (Want != evaluator::HAND) { std::printf("[VisualScript] recovery evaluator: %s\n", Want == evaluator::SHADOW ? "shadow (both run, the hand function actuates)" : "generated (the graph actuates)"); std::fflush(stdout); }
                return Want;
            }();
            return E;
        }
        inline const char* Diff(const xanimation::recovery_plan& a, const xanimation::recovery_plan& b) noexcept
        {
            const auto F = [](float x, float y) { return std::bit_cast<std::uint32_t>(x) != std::bit_cast<std::uint32_t>(y); };
            const auto V = [&](const xmath::fvec3& x, const xmath::fvec3& y) { return F(x.m_X, y.m_X) || F(x.m_Y, y.m_Y) || F(x.m_Z, y.m_Z); };
            if (a.m_Kind != b.m_Kind) return "Kind"; if (a.m_Dir != b.m_Dir) return "Dir"; if (a.m_Foot != b.m_Foot) return "Foot"; if (a.m_Steps != b.m_Steps) return "Steps";
            if (V(a.m_Land, b.m_Land)) return "Land"; if (F(a.m_Time, b.m_Time)) return "Time"; if (V(a.m_Xi, b.m_Xi)) return "Xi"; if (F(a.m_Imbalance, b.m_Imbalance)) return "Imbalance";
            return nullptr;
        }
        // one fixture line: hex floats (exact); a new file starts with its schema
        inline void Capture(const xmath::fvec3& Com, const xmath::fvec3& Vel, float Floor, std::span<const xmath::fvec3> Feet, const xmath::fvec3& Sole, const xmath::fvec3& Facing, float Leg
                           , const xanimation::recovery_params& P, const xanimation::recovery_plan& R) noexcept
        {
            static const char* pFile = Env("XLION_RECOVERY_CAPTURE");
            if (!pFile || !*pFile) return;
            std::FILE* f = Open(pFile, "a"); if (!f) return;
            std::fseek(f, 0, SEEK_END);
            if (std::ftell(f) == 0) std::fprintf(f, "# xvisual_script fixture v1: PlanRecovery\n"
                "# schema in: Com:fvec3 Vel:fvec3 Floor:float Feet:fvec3[n] SoleHalf:fvec3 Facing:fvec3 Leg:float P:recovery_params(StepTime Reach MaxSteps Margin Past)"
                " | out: Kind:int Dir:int Foot:int Land:fvec3 Time:float Steps:int Xi:fvec3 Imbalance:float (floats as C hex)\n");
            const auto V3 = [&](const xmath::fvec3& v) { std::fprintf(f, " %a %a %a", v.m_X, v.m_Y, v.m_Z); };
            std::fprintf(f, "case"); V3(Com); V3(Vel); std::fprintf(f, " %a %zu", Floor, Feet.size()); for (auto& x : Feet) V3(x); V3(Sole); V3(Facing);
            std::fprintf(f, " %a %a %a %a %a %a", Leg, P.m_StepTime, P.m_Reach, P.m_MaxSteps, P.m_Margin, P.m_Past);
            std::fprintf(f, " | %d %d %d", int(R.m_Kind), int(R.m_Dir), R.m_Foot); V3(R.m_Land); std::fprintf(f, " %a %d", R.m_Time, R.m_Steps); V3(R.m_Xi); std::fprintf(f, " %a\n", R.m_Imbalance);
            std::fclose(f);
        }
    }

    inline xanimation::recovery_plan RecoveryPlan(std::uint64_t Entity, const xmath::fvec3& Com, const xmath::fvec3& Vel, float Floor, std::span<const xmath::fvec3> Feet, const xmath::fvec3& Sole
                                                 , const xmath::fvec3& Facing, float Leg, const xanimation::recovery_params& P = {}) noexcept
    {
        using namespace recovery_vs;
        xanimation::recovery_plan R;
        const auto E = Evaluator();
#ifdef XLIONRENDER_RECOVERY_VS
        if (E == evaluator::GENERATED) R = xvs_generated::PlanRecoveryVS(Com, Vel, Floor, Feet, Sole, Facing, Leg, P);
        else
#endif
            R = xanimation::PlanRecovery(Com, Vel, Floor, Feet, Sole, Facing, Leg, P);
#ifdef XLIONRENDER_RECOVERY_VS
        if (E == evaluator::SHADOW)
        {
            const auto G = xvs_generated::PlanRecoveryVS(Com, Vel, Floor, Feet, Sole, Facing, Leg, P);
            const char* d = Diff(R, G);
            std::printf("[VisualScript] entity 0x%llX shadow: %s%s\n", (unsigned long long)Entity, d ? "DIFFERS in " : "the generated plan equals the hand plan", d ? d : ""); std::fflush(stdout);
        }
#endif
        (void)Entity;
        Capture(Com, Vel, Floor, Feet, Sole, Facing, Leg, P, R);
        return R;
    }
}

#endif
