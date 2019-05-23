cmake_minimum_required(VERSION 3.5.1)
project(T_ASE_SERVER)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")
#Benchmark configure


set(LOCAL_LIB_DIRS /home/ar103/.local/usr/local/lib)
find_package(GoogleBenchmark REQUIRED)
find_package (Threads REQUIRED)
link_directories(${LOCAL_LIB_DIRS})


add_library(pomdp_domain
        lib/my_utils.cpp
        lib/belief_state.cpp
        lib/pomdps/pomdp_domain.cpp

        )

add_library(partial_solver
        lib/plan.cpp
        lib/partial_policy.cpp
        lib/solvers/partial_solver.cpp
        )
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

set(SOURCE_FILES
        ${PROJECT_SOURCE_DIR}/src/kitchen/kitchen_main.cpp
        ${PROJECT_SOURCE_DIR}/include/kitchen/kitchen_main.h
        ${PROJECT_SOURCE_DIR}/src/kitchen/kitchen_belief.cpp
        ${PROJECT_SOURCE_DIR}/include/kitchen/kitchen_belief.h
        ${PROJECT_SOURCE_DIR}/src/kitchen/kitchen_tau.cpp
        ${PROJECT_SOURCE_DIR}/include/kitchen/kitchen_tau.h

        ${PROJECT_SOURCE_DIR}/include/pomdps/corridor_domain.h
        ${PROJECT_SOURCE_DIR}/lib/pomdps/corridor_domain.cpp
        main.cpp
        )
include_directories(
        include
        /home/ar103/.local/usr/local/include
)
add_executable(${PROJECT_NAME}_Benchmark ${SOURCE_FILES} )
target_link_libraries(
        ${PROJECT_NAME}_Benchmark
        pomdp_domain
        partial_solver
        z3
        ${benchmark_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT}
)

