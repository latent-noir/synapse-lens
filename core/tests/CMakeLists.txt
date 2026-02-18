# ─── Core C++ Tests ──────────────────────────────────────────────────────────

add_executable(test_linalg test_linalg.cpp)
target_link_libraries(test_linalg PRIVATE nnviz_core_lib GTest::gtest_main)
add_test(NAME LinalgTests COMMAND test_linalg)

add_executable(test_statistics test_statistics.cpp)
target_link_libraries(test_statistics PRIVATE nnviz_core_lib GTest::gtest_main)
add_test(NAME StatisticsTests COMMAND test_statistics)

add_executable(test_activations test_activations.cpp)
target_link_libraries(test_activations PRIVATE nnviz_core_lib GTest::gtest_main)
add_test(NAME ActivationTests COMMAND test_activations)

add_executable(test_nn_engine test_nn_engine.cpp)
target_link_libraries(test_nn_engine PRIVATE nnviz_core_lib GTest::gtest_main)
add_test(NAME NNEngineTests COMMAND test_nn_engine)

