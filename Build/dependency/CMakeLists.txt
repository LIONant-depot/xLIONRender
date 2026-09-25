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
set(XLIONRENDER_GLSLC "${CMAKE_SOURCE_DIR}/dependencies/Shaderc/bin/glslc.exe")
set(XLIONRENDER_SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../src/shaders")
set(XLIONRENDER_GENERATED_HEADERS)
foreach(STAGE_PAIR "frag:primitive_frag" "vert:primitive_vert")
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
  "${CMAKE_BINARY_DIR}/sprv_shaders_h"
)
target_compile_definitions(xLIONRender PRIVATE XECS_BUILD_SHARED XLIONRENDER_BUILD_SHARED XLIONRENDER_EXPORTS)
# xecs.h (pulled in transitively via xlioncore's rigid_body component) is built with XECS_BUILD_SHARED
# here, so xtextfile/xproperty symbols it declares dllimport need xLIONCore's own import lib to
# resolve at link time - xLIONCore.dll is the one binary that actually exports them.
target_link_libraries(xLIONRender PRIVATE xGPU xprim_geom xLIONCore)
if(WIN32)
  target_link_libraries(xLIONRender PRIVATE Advapi32.lib)
endif()
