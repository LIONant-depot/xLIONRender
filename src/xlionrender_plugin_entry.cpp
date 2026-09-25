// Self-registration entry points - same ABI Game.dll and xLIONCore already use
// (dependencies/xECSV2/src/xecs_plugin_api.h). xLION.exe resolves these via
// GetModuleHandle+GetProcAddress, never naming xlionrender::primitive/system directly. This is the
// ONLY translation unit that calls RegisterComponents<primitive>()/RegisterSystems<system>() - both
// compiled into THIS DLL, so info_v<primitive> gets populated in the same binary every query in
// xlionrender_system.h also runs in.
//
// Geometry note: primitive shapes are generated from dependencies/xprim_geom (a real, dependency-free
// depot), never from xGPU's own Examples/E19_MaterialEditor/E19_mesh_manager.h - that header is
// example code the engine intends to eventually remove; it's useful only as a reference for HOW it
// calls the same xprim_geom generators, not as something to include.
#include "xlionrender_system.h"
#include "dependencies/xECSV2/src/xecs_plugin_api.h"

extern "C" __declspec(dllexport)
void XecsPlugin_RegisterComponents(xecs::game_mgr::instance& GameMgr, xecs::plugin::token Token) noexcept
{
    GameMgr.RegisterComponents<xlionrender::primitive>(Token);
}

extern "C" __declspec(dllexport)
void XecsPlugin_RegisterSystems(xecs::game_mgr::instance& GameMgr) noexcept
{
    // rigid_body is registered by LIONCore.dll, so only ITS info_v copy got a bit at Lock - resolve
    // this DLL's own copy (and its built-ins) by GUID before the system below is created and queries it.
    GameMgr.m_ComponentMgr.LockComponentTypes();
    xecs::component::mgr::SyncLocalBitIDs<xlioncore::physics::rigid_body>();
    GameMgr.RegisterSystems<xlionrender::system>();
}
