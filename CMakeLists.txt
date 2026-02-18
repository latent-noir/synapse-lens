cmake_minimum_required(VERSION 3.20)
project(nnviz_core VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ─── Options ─────────────────────────────────────────────────────────────────
option(NNVIZ_BUILD_TESTS "Build C++ unit tests" ON)
option(NNVIZ_BUILD_PYTHON "Build pybind11 Python bindings" ON)

# ─── Dependencies ────────────────────────────────────────────────────────────
find_package(Eigen3 3.4 REQUIRED NO_MODULE)

if(NNVIZ_BUILD_PYTHON)
    find_package(pybind11 CONFIG REQUIRED)
endif()

if(NNVIZ_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    enable_testing()
endif()

# ─── Subdirectories ──────────────────────────────────────────────────────────
add_subdirectory(core)
