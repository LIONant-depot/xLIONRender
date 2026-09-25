#include "xlionrender_api.h"
#include "xlionrender_renderer.h"
#include "xlionrender_internal.h"
#include "xlionrender_system.h"

namespace xlionrender
{
    inline renderer g_Renderer;
    inline system*  g_pSystem = nullptr;

    void SetActiveSystemInternal(system* pSystem) noexcept { g_pSystem = pSystem; }

    bool Init(xgpu::device& Device) noexcept
    {
        return g_Renderer.Init(Device);
    }

    void Draw(xgpu::cmd_buffer& CmdBuffer, const xmath::fmat4& W2C, float ViewportW, float ViewportH) noexcept
    {
        if (g_pSystem) g_pSystem->Collect();
        g_Renderer.Draw(CmdBuffer, W2C, ViewportW, ViewportH);
    }

    std::uint64_t Pick(const xmath::fvec3& Origin, const xmath::fvec3& Dir) noexcept
    {
        return g_pSystem ? g_pSystem->Pick(Origin, Dir) : xecs::component::entity::invalid_entity_v;
    }

    void SetSelectedEntity(std::uint64_t EntityValue) noexcept
    {
        g_Renderer.SetSelected(EntityValue);
    }

    // Not exported (no header declares it outside this DLL) - only system::Collect, compiled into
    // this same DLL (xlionrender_system.h), ever calls this. "No one submits shapes except internally
    // to this DLL" - this is the one function that makes that literally true.
    void SubmitInternal(shape Shape, const xmath::fmat4& L2W, const xmath::fvec3& Color, std::uint64_t EntityValue) noexcept
    {
        g_Renderer.Submit(Shape, L2W, Color, EntityValue);
    }
}
