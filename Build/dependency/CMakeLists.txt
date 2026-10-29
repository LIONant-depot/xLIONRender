include(${CMAKE_BINARY_DIR}/_deps/xcmake_tools/Common.cmake)

# Defensive - FetchAndPopulate is idempotent, a no-op if the host project already fetched these. Not
# fetching xLIONCore here (unlike xECSV2/xGPU/xprim_geom): its own Build/dependency/CMakeLists.txt has
# no guard against being add_subdirectory'd twice (unconditional add_library), so fetching it again
# from here would redefine the xLIONCore target - xLION's own top-level CMakeLists.txt already fetches
# it, and does so before this depot (ordering requirement, noted here since it's easy to miss).
FetchAndPopulate("https://github.com/LIONant-depot/xECSV2.git"     "main")
FetchAndPopulate("https://github.com/LIONant-depot/xGPU.git"       "main")
FetchAndPopulate("https://github.com/LIONant-depot/xprim_geom.git" "main")

# Shader compilation - self-contained (not relying on xLION's own GLSLC variable, which may not be
# set yet at the point this depot gets fetched - resolves the same relative path xLION's own
# CMakeLists.txt uses, independent of fetch ordering).
if(WIN32)
  set(XLIONRENDER_GLSLC "${CMAKE_SOURCE_DIR}/dependencies/Shaderc/bin/glslc.exe")
else()
  find_program(XLIONRENDER_GLSLC glslc REQUIRED) # Linux port: the distro's glslc
endif()
set(XLIONRENDER_SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../src/shaders")
set(XLIONRENDER_GENERATED_HEADERS)
foreach(STAGE_PAIR "frag:primitive_frag" "vert:primitive_vert" "frag:outline_frag" "vert:outline_vert" "frag:outline_obb_frag" "vert:outline_obb_vert" "frag:capsule_frag" "vert:capsule_vert" "frag:text_frag" "vert:text_vert")
  string(REPLACE ":" ";" STAGE_PAIR_LIST ${STAGE_PAIR})
  list(GET STAGE_PAIR_LIST 0 STAGE)
  list(GET STAGE_PAIR_LIST 1 NAME)
  set(SHADER_SRC "${XLIONRENDER_SHADER_DIR}/xlionrender_${NAME}.glsl")
  set(SHADER_OUT "${CMAKE_BINARY_DIR}/sprv_shaders_h/xlionrender_${NAME}.h")
  add_custom_command(
    OUTPUT ${SHADER_OUT}
    COMMAND ${XLIONRENDER_GLSLC} -fshader-stage=${STAGE} "${SHADER_SRC}" -mfmt=num -o ${SHADER_OUT}
    DEPENDS "${SHADER_SRC}"
    COMMENT "Compiling xLIONRender ${STAGE} shader ${NAME}"
  )
  list(APPEND XLIONRENDER_GENERATED_HEADERS ${SHADER_OUT})
endforeach()

# The new renderer (xlionrender::instance, see documentation/xlion_renderer_research.md): an INTERFACE library like
# xGPU - every binary that links it compiles its own copy (instances are per binary, there are no globals). Its
# shaders include the shared library in src/shaders, hence -I.
set(XLIONRENDER_LIB_SHADERS)
foreach(STAGE_PAIR "vert:xlionrender_forward_vert" "frag:xlionrender_forward_standard_frag" "vert:xlionrender_fullscreen_vert"
                   "frag:xlionrender_tonemap_frag" "frag:xlionrender_blit_frag" "frag:xlionrender_default_material_frag" "frag:xlionrender_depth_frag" "frag:xlionrender_gprepass_frag" "vert:xlionrender_gi_gather_vert" "frag:xlionrender_gi_classify_frag" "frag:xlionrender_gi_gather_frag" "vert:xlionrender_clear_depth_vert" "frag:xlionrender_geom_standard_frag" "frag:xlionrender_env_sky_frag" "frag:xlionrender_env_prefilter_frag" "frag:xlionrender_env_dfg_frag" "frag:xlionrender_env_downsample_frag" "frag:xlionrender_env_cube_to_oct_frag" "frag:xlionrender_env_cubemap_frag" "frag:xlionrender_sky_background_frag" "frag:xlionrender_outline_mask_frag" "frag:xlionrender_outline_composite_frag" "vert:xlionrender_static_geom_vert" "vert:xlionrender_static_geom_gi_vert" "vert:xlionrender_static_geom_gi_records_vert" "vert:xlionrender_skin_geom_vert" "vert:xlionrender_gi_capture_vert" "comp:xlionrender_gi_surfel_pass_comp" "comp:xlionrender_gi_scatter_comp" "comp:xlionrender_gi_commit_comp" "comp:xlionrender_gi_ops_comp" "comp:xlionrender_gi_filter_comp" "comp:xlionrender_gi_expand_comp" "comp:xlionrender_gi_means_comp" "comp:xlionrender_gi_cube_comp" "comp:xlionrender_gi_migrate_comp" "comp:xlionrender_gi_readback_comp" "comp:xlionrender_cull_comp" "comp:xlionrender_hiz_comp" "comp:xlionrender_ao_comp" "comp:xlionrender_vsm_request_comp" "comp:xlionrender_shadow_pairs_comp" "comp:xlionrender_local_request_comp" "comp:xlionrender_light_cells_comp" "frag:xlionrender_tail_screen_frag" "comp:xlionrender_tail_denoise_comp" "vert:xlionrender_shadow_pairs_vert" "comp:xlionrender_gi_links_comp" "comp:xlionrender_gi_solve_comp" "comp:xlionrender_gi_ghosts_comp" "comp:xlionrender_gi_probes_comp" "comp:xlionrender_refl_trace_comp" "comp:xlionrender_refl_denoise_comp" "comp:xlionrender_refl_ssr_comp" "comp:xlionrender_refl_hits_comp" "comp:xlionrender_refl_planar_mips_comp")
  string(REPLACE ":" ";" STAGE_PAIR_LIST ${STAGE_PAIR})
  list(GET STAGE_PAIR_LIST 0 STAGE)
  list(GET STAGE_PAIR_LIST 1 NAME)
  set(SHADER_OUT "${CMAKE_BINARY_DIR}/sprv_shaders_h/${NAME}.h")
  file(GLOB XLIONRENDER_SHADER_LIBRARY CONFIGURE_DEPENDS "${XLIONRENDER_SHADER_DIR}/*")
  add_custom_command(
    OUTPUT ${SHADER_OUT}
    COMMAND ${XLIONRENDER_GLSLC} -fshader-stage=${STAGE} --target-env=vulkan1.2 -I "${XLIONRENDER_SHADER_DIR}" "${XLIONRENDER_SHADER_DIR}/${NAME}.glsl" -mfmt=num -o ${SHADER_OUT}
    DEPENDS ${XLIONRENDER_SHADER_LIBRARY}
    COMMENT "Compiling xLIONRender ${STAGE} shader ${NAME}"
  )
  list(APPEND XLIONRENDER_LIB_SHADERS ${SHADER_OUT})
endforeach()
# The built-in lit fragment shaders' G-buffer variants (XLION_GBUFFER_OUTPUT: the half-size thin G-buffer of the screen tail,
# documentation/lighting_p3_design.md "The decision"): <name>_gbuffer.h
foreach(NAME xlionrender_forward_standard_frag xlionrender_geom_standard_frag xlionrender_default_material_frag)
  set(SHADER_OUT "${CMAKE_BINARY_DIR}/sprv_shaders_h/${NAME}_gbuffer.h")
  file(GLOB XLIONRENDER_SHADER_LIBRARY CONFIGURE_DEPENDS "${XLIONRENDER_SHADER_DIR}/*")
  add_custom_command(
    OUTPUT ${SHADER_OUT}
    COMMAND ${XLIONRENDER_GLSLC} -fshader-stage=frag --target-env=vulkan1.2 -DXLION_GBUFFER_OUTPUT -I "${XLIONRENDER_SHADER_DIR}" "${XLIONRENDER_SHADER_DIR}/${NAME}.glsl" -mfmt=num -o ${SHADER_OUT}
    DEPENDS ${XLIONRENDER_SHADER_LIBRARY}
    COMMENT "Compiling xLIONRender G-buffer variant ${NAME}"
  )
  list(APPEND XLIONRENDER_LIB_SHADERS ${SHADER_OUT})
endforeach()

# The compiled shaders are built by their own target: a custom command's output is not visible to the consumers'
# directories, so it cannot be an INTERFACE source (CMake 3.19+ propagates the dependency to whoever links the library)
add_custom_target(xLIONRender_lib_shaders DEPENDS ${XLIONRENDER_LIB_SHADERS})
add_library(xLIONRender_lib INTERFACE)
add_dependencies(xLIONRender_lib xLIONRender_lib_shaders)
target_sources(xLIONRender_lib INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}/../../src/xlionrender_instance.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../src/plans/forward/xlionrender_plan_forward.cpp"
)
target_include_directories(xLIONRender_lib INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}/../../src"
  "${CMAKE_BINARY_DIR}/sprv_shaders_h"
)

# xLIONRender's own shared library - proves this DLL is a real home for components + a system, same
# shape as xLIONCore. Self-registers via XecsPlugin_RegisterComponents/RegisterSystems (see
# xlionrender_plugin_entry.cpp's own comment) - xLION.exe never includes primitive/system directly.
# xtextfile.cpp/my_properties.cpp: not exported/dllimport'd from anywhere - every binary that touches
# xecs.h's types compiles its own private copy (the exact same pattern xLION.exe's own CMakeLists.txt
# and xLIONCore's own Build/dependency/CMakeLists.txt already use - see their own comments).
add_library(xLIONRender SHARED
  "${CMAKE_SOURCE_DIR}/dependencies/xtextfile/source/xtextfile.cpp"
  "${CMAKE_SOURCE_DIR}/dependencies/xproperty/source/xcore/my_properties.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../src/xlionrender_renderer.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../src/xlionrender_api.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../src/xlionrender_text_renderer.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../src/xlionrender_plugin_entry.cpp"
  ${XLIONRENDER_GENERATED_HEADERS}
)
set_target_properties(xLIONRender PROPERTIES OUTPUT_NAME "LIONRender")
target_include_directories(xLIONRender PRIVATE
  "${CMAKE_SOURCE_DIR}"
  "${CMAKE_SOURCE_DIR}/dependencies/xECSV2/src"
  "${CMAKE_SOURCE_DIR}/dependencies/xerr"
  "${CMAKE_SOURCE_DIR}/dependencies/xresource_guid"
  "${CMAKE_SOURCE_DIR}/dependencies/xtextfile"
  "${CMAKE_SOURCE_DIR}/dependencies/xproperty"
  "${CMAKE_SOURCE_DIR}/dependencies/xresource_pipeline_v2"
  "${CMAKE_SOURCE_DIR}/dependencies/xstrtool"
  "${CMAKE_SOURCE_DIR}/dependencies/xcontainer"
  "${CMAKE_SOURCE_DIR}/dependencies/xdelegate"
  "${CMAKE_SOURCE_DIR}/dependencies/xscheduler"
  "${CMAKE_SOURCE_DIR}/dependencies/xmath"
  "${CMAKE_SOURCE_DIR}/dependencies/xbits"
  "${CMAKE_SOURCE_DIR}/dependencies/box3d/include"
  "${CMAKE_SOURCE_DIR}/dependencies/xGPU"                # the resource plugins' runtime headers include "source/xGPU.h"
  "${CMAKE_BINARY_DIR}/sprv_shaders_h"
)
target_compile_definitions(xLIONRender PRIVATE XECS_BUILD_SHARED XLIONRENDER_BUILD_SHARED XLIONRENDER_EXPORTS XLIONCORE_BUILD_SHARED)
# xecs.h (pulled in transitively via xlioncore's physics_body_properties component) is built with XECS_BUILD_SHARED
# here, so xtextfile/xproperty symbols it declares dllimport need xLIONCore's own import lib to
# resolve at link time - xLIONCore.dll is the one binary that actually exports them.
# xLIONRender_lib: the forward renderer (xlionrender::instance), compiled into this DLL for the Levels' Meshes and lights (xlionrender_scene_renderer.h)
target_link_libraries(xLIONRender PRIVATE xGPU xprim_geom xLIONCore xLIONRender_lib)
if(WIN32)
  target_link_libraries(xLIONRender PRIVATE Advapi32.lib)
endif()
if(MSVC)
  target_compile_options(xLIONRender PRIVATE /bigobj)   # xlionrender_api.cpp (the renderer + GI headers) passes the Debug section limit
endif()

# Every header under src/ (xlionrender_api.h, xlionrender_internal.h, xlionrender_primitive.h,
# xlionrender_system.h) was never listed above - only the .cpp files that actually compile into
# this DLL were - so none of it was visible in xLIONRender's own Visual Studio project, same gap
# fixed in xLIONCore's own Build/dependency/CMakeLists.txt (see its comment). GLOB_RECURSE here (not
# a hand-maintained list, unlike add_library's own source list above) is deliberate: this is IDE
# browsability only, never build correctness, so a new header just needs one reconfigure to appear.
file(GLOB_RECURSE XLIONRENDER_HEADERS CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/../../src/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../src/*.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../src/*.inl"
)
target_sources(xLIONRender PRIVATE ${XLIONRENDER_HEADERS})
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/../../src" PREFIX "src"
  FILES ${XLIONRENDER_HEADERS}
        "${CMAKE_CURRENT_SOURCE_DIR}/../../src/xlionrender_renderer.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../src/xlionrender_api.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../src/xlionrender_plugin_entry.cpp"
)
source_group("Shaders" FILES ${XLIONRENDER_GENERATED_HEADERS})
