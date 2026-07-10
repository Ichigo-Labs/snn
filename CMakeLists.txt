cmake_minimum_required(VERSION 3.22)
project(snn LANGUAGES C)

option(SNN_ENABLE_CUDA "Build CUDA backend" ON)
option(SNN_BUILD_TESTS "Build test suite" ON)
option(SNN_BUILD_BENCHMARKS "Build benchmark executables" OFF)
option(SNN_BUILD_EXAMPLES "Build example executables" OFF)
option(SNN_BUILD_TOOLS "Build the MNIST BPTT training tool (needs zlib)" OFF)
option(SNN_ENABLE_COVERAGE "Build with gcov line coverage instrumentation" OFF)
option(SNN_ENABLE_OPENMP "Parallelize the CPU simulation step with OpenMP" OFF)
option(SNN_ENABLE_NATIVE_ARCH "Compile the CPU library with -march=native (faster, non-portable binaries)" OFF)

# An empty build type compiles the hot simulation loops with no optimization;
# default to Release so a plain `cmake -S . -B build` produces a fast library.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(snn_cpu OBJECT src/snn.c src/snn_bptt.c)
target_include_directories(snn_cpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_options(snn_cpu PRIVATE -Wall -Wextra -Wpedantic -Werror)
if(SNN_ENABLE_NATIVE_ARCH)
    target_compile_options(snn_cpu PRIVATE -march=native)
endif()
if(SNN_ENABLE_COVERAGE)
    target_compile_options(snn_cpu PRIVATE --coverage -O0 -g)
endif()
if(SNN_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED COMPONENTS C)
    target_compile_definitions(snn_cpu PRIVATE SNN_ENABLE_OPENMP=1)
    target_link_libraries(snn_cpu PUBLIC OpenMP::OpenMP_C)
    # snn_cpu is an OBJECT library, so its link deps are not transitively applied
    # to consumers of the aggregate `snn` target; attach OpenMP there too.
    set(SNN_OPENMP_LINK OpenMP::OpenMP_C)
else()
    set(SNN_OPENMP_LINK "")
endif()

set(SNN_CUDA_TARGETS snn_cuda_stub)
if(SNN_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        # Imported CUDA::cudart carries the toolkit library dir; a bare
        # `cudart` links only on machines where it happens to be ldconfig'd.
        find_package(CUDAToolkit REQUIRED)
        add_library(snn_cuda_obj OBJECT src/snn_cuda.cu)
        set_target_properties(snn_cuda_obj PROPERTIES CUDA_STANDARD 14 CUDA_STANDARD_REQUIRED ON CUDA_SEPARABLE_COMPILATION OFF)
        target_include_directories(snn_cuda_obj PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src)
        # --fmad=false keeps the integrate kernel's arithmetic bit-identical to
        # the host build (which targets baseline x86-64 without FMA), so the
        # CPU-vs-GPU parity tests can assert exact voltage equality.
        target_compile_options(snn_cuda_obj PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --fmad=false --compiler-options=-Wall,-Wextra>)
        if(SNN_ENABLE_COVERAGE)
            # Instrument the host-side code of the .cu translation unit so gcov
            # can measure the CUDA backend's control flow (kernels excluded).
            target_compile_options(snn_cuda_obj PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=--coverage -O0 -g>)
        endif()
        add_library(snn STATIC $<TARGET_OBJECTS:snn_cpu> $<TARGET_OBJECTS:snn_cuda_obj>)
        target_include_directories(snn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
        target_link_libraries(snn PUBLIC CUDA::cudart m ${SNN_OPENMP_LINK})
        target_compile_definitions(snn PUBLIC SNN_WITH_CUDA=1)
    else()
        message(WARNING "CUDA requested but nvcc not found; building CPU library with CUDA stubs")
        add_library(snn STATIC $<TARGET_OBJECTS:snn_cpu> src/snn_cuda_stub.c)
        target_include_directories(snn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
        target_link_libraries(snn PUBLIC m ${SNN_OPENMP_LINK})
        target_compile_definitions(snn PUBLIC SNN_WITH_CUDA=0)
    endif()
else()
    add_library(snn STATIC $<TARGET_OBJECTS:snn_cpu> src/snn_cuda_stub.c)
    target_include_directories(snn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(snn PUBLIC m ${SNN_OPENMP_LINK})
    target_compile_definitions(snn PUBLIC SNN_WITH_CUDA=0)
endif()

if(SNN_ENABLE_COVERAGE)
    target_compile_options(snn PRIVATE --coverage -O0 -g)
    target_link_options(snn INTERFACE --coverage)
endif()

if(SNN_BUILD_TESTS)
    target_compile_definitions(snn_cpu PUBLIC SNN_ENABLE_TEST_HOOKS=1)
    target_compile_definitions(snn PUBLIC SNN_ENABLE_TEST_HOOKS=1)
    if(SNN_ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
        target_compile_definitions(snn_cuda_obj PUBLIC SNN_ENABLE_TEST_HOOKS=1)
    endif()
    enable_testing()
    add_executable(test_snn tests/test_snn.c tests/test_bptt.c)
    target_link_libraries(test_snn PRIVATE snn)
    target_include_directories(test_snn PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    if(SNN_ENABLE_COVERAGE)
        target_compile_options(test_snn PRIVATE --coverage -O0 -g)
        # The test executable is linked by the host compiler (gcc/g++) in both
        # the CPU-only and CUDA-enabled configurations, so plain --coverage is
        # the correct link flag for the gcov runtime.
        target_link_options(test_snn PRIVATE --coverage)
    endif()
    add_test(NAME test_snn COMMAND test_snn)
endif()

if(SNN_BUILD_BENCHMARKS)
    add_executable(step_throughput benchmarks/step_throughput.c)
    target_link_libraries(step_throughput PRIVATE snn)
    target_include_directories(step_throughput PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_options(step_throughput PRIVATE -Wall -Wextra)
endif()

if(SNN_BUILD_EXAMPLES)
    add_executable(minimal examples/minimal.c)
    target_link_libraries(minimal PRIVATE snn)
    target_include_directories(minimal PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_options(minimal PRIVATE -Wall -Wextra)
endif()

if(SNN_BUILD_TOOLS)
    # The trainer reads the committed MNIST idx.gz files directly.
    find_package(ZLIB REQUIRED)
    if(SNN_ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
        # The GPU training backend is a tool convenience, not a library
        # feature: unlike snn_cuda_obj it compiles with fmad ON, because the
        # tool's --mode gputest verifies it against the CPU trainer to float
        # tolerance rather than bitwise. sm_60 is the floor for the double
        # atomicAdd it uses; newer devices JIT the PTX forward.
        add_executable(mnist_bptt tools/mnist_bptt.c tools/bptt_cuda.cu)
        set_target_properties(mnist_bptt PROPERTIES CUDA_STANDARD 14 CUDA_STANDARD_REQUIRED ON
                              CUDA_ARCHITECTURES 60 CUDA_SEPARABLE_COMPILATION OFF)
        target_compile_options(mnist_bptt PRIVATE
                               $<$<COMPILE_LANGUAGE:CUDA>:-O3 --compiler-options=-Wall,-Wextra>)
        target_link_libraries(mnist_bptt PRIVATE CUDA::cudart CUDA::cublas)
    else()
        add_executable(mnist_bptt tools/mnist_bptt.c tools/bptt_cuda_stub.c)
    endif()
    target_link_libraries(mnist_bptt PRIVATE snn ZLIB::ZLIB m)
    target_include_directories(mnist_bptt PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_options(mnist_bptt PRIVATE -Wall -Wextra)
    # The BPTT library is thread-agnostic (one workspace and one gradient
    # accumulator per thread), so the tool parallelizes over the minibatch even
    # when the library itself was built without OpenMP.
    find_package(OpenMP QUIET COMPONENTS C)
    if(OpenMP_C_FOUND)
        target_link_libraries(mnist_bptt PRIVATE OpenMP::OpenMP_C)
    endif()
endif()
