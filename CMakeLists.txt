cmake_minimum_required(VERSION 3.10)
project(xLIONRender_unit_test LANGUAGES C CXX)

# Standalone buildable project for xLIONRender on its own - generates its own solution the same way
# dependencies/xLIONCore's own top-level CMakeLists.txt does. A consuming project (xLION) instead
# fetches this depot via FetchAndPopulate, which picks up Build/dependency/CMakeLists.txt directly -
# this file add_subdirectory's that same file rather than duplicating its add_library block.
if(POLICY CMP0169)
  cmake_policy(SET CMP0169 OLD)
endif()

include(FetchContent)
FetchContent_Declare(
  xcmake_tools
  GIT_REPOSITORY https://github.com/LIONant-depot/xcmake_tools.git
  GIT_TAG main
  GIT_SHALLOW TRUE
  SOURCE_DIR "${CMAKE_BINARY_DIR}/_deps/xcmake_tools"
)
FetchContent_MakeAvailable(xcmake_tools)
include(${CMAKE_BINARY_DIR}/_deps/xcmake_tools/Common.cmake)

# Transitive dependencies xecs.h/xgpu.h/xprim_geom.h need - same list xLION's own top-level
# CMakeLists.txt already fetches directly; everything else comes in transitively.
FetchAndPopulate("https://github.com/LIONant-depot/xresource_mgr.git"    "main")
FetchAndPopulate("https://github.com/LIONant-depot/xcontainer.git"       "main")
FetchAndPopulate("https://github.com/LIONant-depot/xdelegate.git"        "main")
FetchAndPopulate("https://github.com/LIONant-depot/xprim_geom.git"       "main")
FetchAndPopulate("https://github.com/LIONant-depot/xproperty.git"        "master")
FetchAndPopulate("https://github.com/LIONant-depot/xstrtool.git"         "main")
FetchAndPopulate("https://github.com/LIONant-depot/xserializer.git"      "main")
FetchAndPopulate("https://github.com/LIONant-depot/xscheduler.git"       "main")
FetchAndPopulate("https://github.com/LIONant-depot/xundo.git"            "main")
FetchAndPopulate("https://github.com/LIONant-depot/xsource_control.git"  "main")
FetchAndPopulate("https://github.com/LIONant-depot/xGPU.git"             "main")
# Unlike Build/dependency/CMakeLists.txt (deliberately does NOT fetch this - see its own comment on
# why, when xLION has already fetched it first), this standalone entry point is the only fetcher in
# this mode, so it needs to bring xLIONCore in itself.
FetchAndPopulate("https://github.com/LIONant-depot/xLIONCore.git"        "main")

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/Build/dependency" "xLIONRender_build")
