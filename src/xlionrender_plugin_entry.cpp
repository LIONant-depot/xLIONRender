// Self-registration entry points - same ABI Game.dll and xLIONCore already use
// (dependencies/xECSV2/src/xecs_plugin_api.h), plus XScript_GetComponentDisplayInfo for inspector
// category/priority (Primitive -> Rendering).
#include "xlionrender_system.h"
#include "dependencies/xECSV2/src/xecs_plugin_api.h"
#include "plugins/xscript_module.plugin/source/Runtime/xscript_registration.h"

extern "C" __declspec(dllexport)
void XecsPlugin_RegisterComponents(xecs::game_mgr::instance& GameMgr, xecs::plugin::token Token) noexcept
{
    for (auto* p = xscript::self_registration<xscript::component_entry>::s_pHead; p; p = p->m_pNext)
        p->m_Value.m_pRegisterFn(GameMgr, Token);
}

extern "C" __declspec(dllexport)
void XecsPlugin_RegisterSystems(xecs::game_mgr::instance& GameMgr) noexcept
{
    // transform is registered by LIONCore.dll, so only ITS info_v copy got a bit at Lock - resolve
    // this DLL's own copy (and its built-ins) by GUID before the system below is created and queries it.
    GameMgr.m_ComponentMgr.LockComponentTypes();
    xecs::component::mgr::SyncLocalBitIDs<xlioncore::transform, xlioncore::render_transform, xlioncore::no_render_tag, xecs::editor::no_render_tag>();
    GameMgr.RegisterSystems<xlionrender::system>();
}

extern "C" __declspec(dllexport)
void XScript_GetComponentDisplayInfo(xscript::pfn_component_display_visitor pVisitor, void* pUserData) noexcept
{
    for (auto* p = xscript::self_registration<xscript::component_entry>::s_pHead; p; p = p->m_pNext)
        pVisitor(pUserData, p->m_Value.m_Guid, p->m_Value.m_pName, p->m_Value.m_pCategory, p->m_Value.m_Priority);
}
