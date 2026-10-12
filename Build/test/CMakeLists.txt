cmake_minimum_required(VERSION 3.20)
project(xLIONRender_tests LANGUAGES CXX)

# Tests for the new renderer (src/ minus the old editor renderer and ecs/). Builds against the working trees next
# to this depot (<root>/dependencies/xGPU, xmath, xprim_geom) so unpushed xGPU changes are tested too.
#   cmake -S Build/test -B Build/test/vs2022 -G "Visual Studio 17 2022" -A x64
#   cmake --build Build/test/vs2022 --config Debug && ctest --test-dir Build/test/vs2022 -C Debug
get_filename_component(XLIONRENDER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
get_filename_component(XLION_ROOT_DEFAULT "${XLIONRENDER_DIR}/../.." ABSOLUTE)
set(XLION_ROOT "${XLION_ROOT_DEFAULT}" CACHE PATH "Folder whose dependencies/ holds xGPU, xmath, xprim_geom")

if(NOT DEFINED ENV{VULKAN_SDK})
  message(FATAL_ERROR "VULKAN_SDK is not set")
endif()
set(VULKAN_SDK "$ENV{VULKAN_SDK}")
set(GLSLC "${VULKAN_SDK}/Bin/glslc.exe")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SHADER_DIR "${XLIONRENDER_DIR}/src/shaders")
set(SHADER_OUT_DIR "${CMAKE_BINARY_DIR}/sprv_shaders_h")
set(SHADER_HEADERS)
foreach(PAIR xlionrender_forward_vert:vertex xlionrender_forward_standard_frag:fragment xlionrender_fullscreen_vert:vertex
             xlionrender_tonemap_frag:fragment xlionrender_blit_frag:fragment xlionrender_default_material_frag:fragment xlionrender_depth_frag:fragment xlionrender_clear_depth_vert:vertex xlionrender_shadow_remap_frag:fragment xlionrender_static_geom_vert:vertex xlionrender_skin_geom_vert:vertex)
  string(REPLACE ":" ";" PAIR ${PAIR})
  list(GET PAIR 0 NAME)
  list(GET PAIR 1 STAGE)
  set(OUT "${SHADER_OUT_DIR}/${NAME}.h")
  file(GLOB SHADER_LIBRARY "${SHADER_DIR}/*")    # includes: any change rebuilds every shader (there are few)
  add_custom_command(OUTPUT ${OUT}
    COMMAND ${GLSLC} -fshader-stage=${STAGE} -I "${SHADER_DIR}" "${SHADER_DIR}/${NAME}.glsl" -mfmt=num -o ${OUT}
    DEPENDS ${SHADER_LIBRARY}
    COMMENT "glslc ${NAME}")
  list(APPEND SHADER_HEADERS ${OUT})
endforeach()

# Test-only shaders (src/tests/shaders), compiled against the same shader library
set(TEST_SHADER_DIR "${XLIONRENDER_DIR}/src/tests/shaders")
foreach(PAIR xlionrender_test_material_frag:fragment xlionrender_test_normal_frag:fragment)
  string(REPLACE ":" ";" PAIR ${PAIR})
  list(GET PAIR 0 NAME)
  list(GET PAIR 1 STAGE)
  set(OUT "${SHADER_OUT_DIR}/${NAME}.h")
  add_custom_command(OUTPUT ${OUT}
    COMMAND ${GLSLC} -fshader-stage=${STAGE} -I "${SHADER_DIR}" "${TEST_SHADER_DIR}/${NAME}.glsl" -mfmt=num -o ${OUT}
    DEPENDS "${TEST_SHADER_DIR}/${NAME}.glsl" ${SHADER_LIBRARY}
    COMMENT "glslc ${NAME}")
  list(APPEND TEST_SHADER_HEADERS ${OUT})
endforeach()

# The renderer itself, shared by every test
add_library(xlionrender_new STATIC
  "${XLIONRENDER_DIR}/src/xlionrender_instance.cpp"
  "${XLIONRENDER_DIR}/src/plans/forward/xlionrender_plan_forward.cpp"
  "${XLION_ROOT}/dependencies/xGPU/source/xGPU.cpp"
  ${SHADER_HEADERS})
target_include_directories(xlionrender_new PUBLIC
  "${XLIONRENDER_DIR}/src" "${XLION_ROOT}" "${XLION_ROOT}/dependencies/xGPU/source" "${VULKAN_SDK}/Include" "${SHADER_OUT_DIR}")
target_link_directories(xlionrender_new PUBLIC "${VULKAN_SDK}")   # xGPU links "/Lib/vulkan-1.lib" via #pragma
target_compile_definitions(xlionrender_new PUBLIC UNICODE _UNICODE NOMINMAX)
target_compile_options(xlionrender_new PUBLIC /MP)

enable_testing()
foreach(TEST xlionrender_test_forward xlionrender_test_static_geom xlionrender_test_skin_geom xlionrender_test_shadows xlionrender_test_local_shadows xlionrender_test_light_lab)
  add_executable(${TEST} "${XLIONRENDER_DIR}/src/tests/${TEST}.cpp" ${TEST_SHADER_HEADERS})
  target_link_libraries(${TEST} PRIVATE xlionrender_new)
  add_test(NAME ${TEST} COMMAND ${TEST})
endforeach()
