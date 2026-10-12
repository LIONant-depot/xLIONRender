#ifndef XLIONRENDER_H
#define XLIONRENDER_H
#pragma once

// The one include a tool needs. Typical use:
//
//      xlionrender::instance Renderer;     Renderer.Init(Device);
//      xlionrender::mesh_id  Sphere;       Renderer.CreateMesh(Sphere, Vertices, Indices);
//      xlionrender::render_view View;      Renderer.CreateView(View, { .m_Width = 512, .m_Height = 512 });
//      xlionrender::scene    Scene;        Scene.m_Draws.push_back({ .m_Mesh = Sphere });
//      ... each frame, between Window.BeginRendering() and Window.PageFlip():
//      Renderer.Render(Window, Scene, View, Camera);      // View.m_Output holds the tone-mapped image
//
// Source layout and direction: documentation/xlion_renderer_research.md

#include "xlionrender_instance.h"

#endif
