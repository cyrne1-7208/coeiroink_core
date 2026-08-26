cmake_minimum_required(VERSION 3.31)
project(coeiroink_opencl LANGUAGES CXX)

include(FetchContent)
find_package(Git REQUIRED)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "coeiroink-opencl is supported only on Linux")
endif()

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(SQLite3 REQUIRED)

# OpenCL SDKはビルドホストから参照し、wheelへloaderやベンダードライバを同梱しない。
if(OCL_PATH AND NOT OCL_PATH MATCHES "-NOTFOUND$")
  find_path(
    COEIROINK_OPENCL_INCLUDE_DIR
    NAMES CL/opencl.hpp CL/cl2.hpp
    PATHS "${OCL_PATH}"
    NO_DEFAULT_PATH
    REQUIRED
  )
else()
  find_path(
    COEIROINK_OPENCL_INCLUDE_DIR
    NAMES CL/opencl.hpp CL/cl2.hpp
    REQUIRED
  )
endif()

if(OCL_LIB AND EXISTS "${OCL_LIB}")
  set(COEIROINK_OPENCL_LIBRARY "${OCL_LIB}")
elseif(OCL_LIB AND NOT OCL_LIB MATCHES "-NOTFOUND$")
  find_library(
    COEIROINK_OPENCL_LIBRARY
    NAMES OpenCL
    PATHS "${OCL_LIB}"
    NO_DEFAULT_PATH
    REQUIRED
  )
else()
  find_library(COEIROINK_OPENCL_LIBRARY NAMES OpenCL REQUIRED)
endif()

# 上流も同じ変数名を参照するため、検証済みのパスをサブプロジェクトへ渡す。
set(OCL_PATH "${COEIROINK_OPENCL_INCLUDE_DIR}" CACHE PATH "OpenCL include directory" FORCE)
set(OCL_LIB "${COEIROINK_OPENCL_LIBRARY}" CACHE FILEPATH "OpenCL loader library" FORCE)
# DLPrimitivesのキャッシュ機能を有効にし、対応するデバイスでは再起動後もコンパイル結果を再利用する。
set(USE_SQLITE3 ON CACHE BOOL "Enable DLPrimitives kernel cache" FORCE)
set(SQLITE3_PATH "${SQLite3_INCLUDE_DIRS}" CACHE PATH "SQLite3 include directory" FORCE)
set(SQLITE3_LIB "${SQLite3_LIBRARIES}" CACHE FILEPATH "SQLite3 library" FORCE)
set(BUILD_CORE_ONLY ON CACHE BOOL "Build only DLPrimitives core" FORCE)
set(DLPRIM_STATIC_BUILD ON CACHE BOOL "Link DLPrimitives statically" FORCE)

# 公開MIT実装を固定し、BatchSGEMMの未初期化値と行列積・畳み込みの入力長依存コンパイルを修正する。
FetchContent_Declare(
  pytorch_dlprim
  GIT_REPOSITORY https://github.com/artyom-beilis/pytorch_dlprim.git
  GIT_TAG 1af48d4966d83b5b43344288999343e690ea7037
  GIT_SHALLOW FALSE
  GIT_SUBMODULES_RECURSE TRUE
  PATCH_COMMAND
    "${GIT_EXECUTABLE}" -C dlprimitives apply "${CMAKE_CURRENT_LIST_DIR}/patches/dlprimitives-opencl.patch"
  EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(pytorch_dlprim)

# VITSでpytorch_dlprimに不足する演算だけをCore側でPrivateUse1へ登録する。
# 未登録演算はpytorch_dlprim側の警告とCPUフォールバックに任せる。
target_sources(pt_ocl PRIVATE "${CMAKE_CURRENT_LIST_DIR}/src/vits_ops.cpp")
target_include_directories(pt_ocl PRIVATE "${pytorch_dlprim_SOURCE_DIR}/src")

# Linux拡張は実行中インタープリターからPython C APIを解決し、libpythonへ直接依存させない。
get_target_property(_coeiroink_pt_ocl_link_libraries pt_ocl LINK_LIBRARIES)
list(REMOVE_ITEM _coeiroink_pt_ocl_link_libraries ${Python3_LIBRARIES})
set_property(TARGET pt_ocl PROPERTY LINK_LIBRARIES "${_coeiroink_pt_ocl_link_libraries}")

add_custom_target(coeiroink_opencl_build ALL DEPENDS pt_ocl)
install(TARGETS pt_ocl LIBRARY DESTINATION pytorch_ocl)
install(FILES "${CMAKE_CURRENT_LIST_DIR}/python/pytorch_ocl/__init__.py" DESTINATION pytorch_ocl)
# OpenCL wheel単体で再配布条件を確認できるよう、リポジトリのLGPL本文も同梱する。
install(
  FILES "${CMAKE_CURRENT_LIST_DIR}/../../LICENSE"
  DESTINATION pytorch_ocl
  RENAME COEIROINK_LICENSE.txt
)
