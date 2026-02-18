# ─── Core C++ Library ────────────────────────────────────────────────────────

# Static library for the core engine
add_library(nnviz_core_lib STATIC
    src/linalg.cpp
    src/statistics.cpp
    src/activations.cpp
    src/matrix.cpp
    src/dense_layer.cpp
    src/network.cpp
    src/loss.cpp
)

target_include_directories(nnviz_core_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(nnviz_core_lib PUBLIC
    Eigen3::Eigen
)

target_compile_definitions(nnviz_core_lib PRIVATE NNVIZ_CORE_EXPORTS)

# ─── pybind11 Python Module ─────────────────────────────────────────────────
if(NNVIZ_BUILD_PYTHON)
    pybind11_add_module(nnviz_core bindings/bind_core.cpp)

    target_link_libraries(nnviz_core PRIVATE
        nnviz_core_lib
    )

    # Install the Python module into the nnviz package directory
    install(TARGETS nnviz_core
        LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/nnviz
    )
endif()

# ─── C++ Unit Tests ──────────────────────────────────────────────────────────
if(NNVIZ_BUILD_TESTS)
    add_subdirectory(tests)
endif()
