#
#   Find packages
#

include(FindPackageHandleStandardArgs)


################################################################################
#
#                               Threading
#
################################################################################

if(CMAKE_C_COMPILER_IS_INTEL OR CMAKE_CXX_COMPILER_IS_INTEL)
    if(NOT WIN32)
        set(THREADS_PREFER_PTHREAD_FLAG OFF CACHE BOOL "Use -pthread vs. -lpthread" FORCE)
    endif()

    find_package(Threads)
    if(Threads_FOUND)
        list(APPEND EXTERNAL_PRIVATE_LIBRARIES Threads::Threads)
    endif()
endif()


################################################################################
#
#        PTL submodule
#
################################################################################

if(TOMOPY_USE_PTL)
    checkout_git_submodule(RECURSIVE TEST_FILE CMakeLists.txt
        RELATIVE_PATH source/PTL WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
    add_subdirectory(source/PTL)
    if(BUILD_STATIC_LIBS)
        list(APPEND EXTERNAL_LIBRARIES ptl-static)
    else()
        list(APPEND EXTERNAL_LIBRARIES ptl-shared)
    endif()
    list(APPEND ${PROJECT_NAME}_DEFINITIONS TOMOPY_USE_PTL)
endif()


################################################################################
#
#        Prefix path to Anaconda installation
#
################################################################################
#
find_package(PythonInterp)
if(PYTHON_EXECUTABLE)
    get_filename_component(PYTHON_ROOT_DIR ${PYTHON_EXECUTABLE} DIRECTORY)
    get_filename_component(PYTHON_ROOT_DIR ${PYTHON_ROOT_DIR} DIRECTORY)
    set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH}
        ${PYTHON_ROOT_DIR}
        ${PYTHON_ROOT_DIR}/bin
        ${PYTHON_ROOT_DIR}/lib
        ${PYTHON_ROOT_DIR}/include)
endif()


################################################################################
#
#        MKL (optional)
#
################################################################################
if(TOMOPY_USE_MKL)
    find_package(MKL)
    if(MKL_FOUND)
        add_library(tomopy-mkl INTERFACE)
        set_target_properties(tomopy-mkl PROPERTIES
            INTERFACE_COMPILE_DEFINITIONS    TOMOPY_USE_MKL
            INTERFACE_INCLUDE_DIRECTORIES    "${MKL_INCLUDE_DIRS}"
            INTERFACE_LINK_LIBRARIES         "${MKL_LIBRARIES}")
        list(APPEND EXTERNAL_LIBRARIES tomopy-mkl)
    endif()
endif()

################################################################################
#
#        OpenCV (required)
#
################################################################################

set(OpenCV_COMPONENTS opencv_core opencv_imgproc)
find_package(OpenCV REQUIRED COMPONENTS ${OpenCV_COMPONENTS})
list(APPEND EXTERNAL_LIBRARIES ${OpenCV_LIBRARIES})


################################################################################
#
#        GCov
#
################################################################################

if(TOMOPY_USE_COVERAGE)
    find_library(GCOV_LIBRARY gcov)
    if(GCOV_LIBRARY)
        list(APPEND EXTERNAL_LIBRARIES ${GCOV_LIBRARY})
    else()
        list(APPEND EXTERNAL_LIBRARIES gcov)
    endif()
    add(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lgcov")
endif()


################################################################################
#
#                               TiMemory
#
################################################################################

if(TOMOPY_USE_TIMEMORY)
    find_package(TiMemory)

    if(TiMemory_FOUND)
        list(APPEND EXTERNAL_INCLUDE_DIRS ${TiMemory_INCLUDE_DIRS})
        list(APPEND EXTERNAL_LIBRARIES
            ${TiMemory_LIBRARIES} ${TiMemory_C_LIBRARIES})
        list(APPEND ${PROJECT_NAME}_DEFINITIONS TOMOPY_USE_TIMEMORY)
    endif()

endif()


################################################################################
#
#        Google PerfTools
#
################################################################################

if(TOMOPY_USE_GPERF)
    find_package(GPerfTools COMPONENTS profiler)

    if(GPerfTools_FOUND)
        list(APPEND EXTERNAL_INCLUDE_DIRS ${GPerfTools_INCLUDE_DIRS})
        list(APPEND EXTERNAL_LIBRARIES ${GPerfTools_LIBRARIES})
        list(APPEND ${PROJECT_NAME}_DEFINITIONS TOMOPY_USE_GPERF)
    endif()

endif()


################################################################################
#
#        OpenMP
#
################################################################################

if(TOMOPY_USE_OPENMP)

    if(NOT WIN32)
        find_package(OpenMP)

        if(OpenMP_FOUND)
            if(CMAKE_C_COMPILER_IS_PGI)
                string(REPLACE "-mp" "-mp${OpenMP_C_IMPL}" OpenMP_C_FLAGS "${OpenMP_C_FLAGS}")
            endif()

            if(CMAKE_CXX_COMPILER_IS_PGI)
                string(REPLACE "-mp" "-mp${OpenMP_C_IMPL}" OpenMP_CXX_FLAGS "${OpenMP_CXX_FLAGS}")
            endif()

            # C
            if(OpenMP_C_FOUND)
                list(APPEND ${PROJECT_NAME}_C_FLAGS ${OpenMP_C_FLAGS})
            endif()

            # C++
            if(OpenMP_CXX_FOUND)
                list(APPEND ${PROJECT_NAME}_CXX_FLAGS ${OpenMP_CXX_FLAGS})
                list(APPEND EXTERNAL_LIBRARIES ${OpenMP_CXX_FLAGS})
            endif()
        else()
            message(WARNING "OpenMP not found")
            set(TOMOPY_USE_OPENMP OFF)
        endif()
    elseif(WIN32)
        message(STATUS "Ignoring TOMOPY_USE_OPENMP=ON because Windows + omp simd is not supported")
        set(TOMOPY_USE_OPENMP OFF)
    else()
        message(STATUS "Ignoring TOMOPY_USE_OPENMP=ON because '-fopenmp-simd' is supported")
        set(TOMOPY_USE_OPENMP OFF)
    endif()

endif()


################################################################################
#
#        CUDA
#
################################################################################

if(TOMOPY_USE_CUDA)

    get_property(LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)

    if("CUDA" IN_LIST LANGUAGES)
        add_interface_library(tomopy-cuda)

        target_compile_definitions(tomopy-cuda INTERFACE TIMEMORY_USE_CUDA)
        target_include_directories(tomopy-cuda INTERFACE ${CUDA_INCLUDE_DIRS}
            ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

        set_target_properties(tomopy-cuda PROPERTIES
            INTERFACE_CUDA_STANDARD                 ${CMAKE_CUDA_STANDARD}
            INTERFACE_CUDA_STANDARD_REQUIRED        ${CMAKE_CUDA_STANDARD_REQUIRED}
            INTERFACE_CUDA_RESOLVE_DEVICE_SYMBOLS   ON
            INTERFACE_CUDA_SEPARABLE_COMPILATION    ON)

        set(CUDA_AUTO_ARCH "auto")
        set(CUDA_ARCHITECTURES auto kepler tesla maxwell pascal volta turing)
        set(CUDA_ARCH "${CUDA_AUTO_ARCH}" CACHE STRING
            "CUDA architecture (options: ${CUDA_ARCHITECTURES})")
        add_feature(CUDA_ARCH "CUDA architecture (options: ${CUDA_ARCHITECTURES})")
        set_property(CACHE CUDA_ARCH PROPERTY STRINGS ${CUDA_ARCHITECTURES})

        set(cuda_kepler_arch    30)
        set(cuda_tesla_arch     35)
        set(cuda_maxwell_arch   50)
        set(cuda_pascal_arch    60)
        set(cuda_volta_arch     70)
        set(cuda_turing_arch    75)

        if(NOT "${CUDA_ARCH}" STREQUAL "${CUDA_AUTO_ARCH}")
            if(NOT "${CUDA_ARCH}" IN_LIST CUDA_ARCHITECTURES)
                message(WARNING
                    "CUDA architecture \"${CUDA_ARCH}\" not known. Options: ${CUDA_ARCH}")
                unset(CUDA_ARCH CACHE)
                set(CUDA_ARCH "${CUDA_AUTO_ARCH}")
            else()
                set(_ARCH_NUM ${cuda_${CUDA_ARCH}_arch})
            endif()
        endif()

        option(TIMEMORY_DEPRECATED_CUDA_SUPPORT "Enable support for old CUDA flags" OFF)
        mark_as_advanced(TIMEMORY_DEPRECATED_CUDA_SUPPORT)

        if(TIMEMORY_DEPRECATED_CUDA_SUPPORT)
            add_interface_library(tomopy-cuda-7)
            target_compile_options(tomopy-cuda-7 INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:
                $<IF:$<STREQUAL:${CUDA_ARCH},${CUDA_AUTO_ARCH}>,-arch=sm_35,-arch=sm_${_ARCH_NUM}>
                -gencode=arch=compute_20,code=sm_20
                -gencode=arch=compute_30,code=sm_30
                -gencode=arch=compute_50,code=sm_50
                -gencode=arch=compute_52,code=sm_52
                -gencode=arch=compute_52,code=compute_52
                >)

            add_interface_library(tomopy-cuda-8)
            target_compile_options(tomopy-cuda-8 INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:
                $<IF:$<STREQUAL:${CUDA_ARCH},${CUDA_AUTO_ARCH}>,-arch=sm_35,-arch=sm_${_ARCH_NUM}>
                -gencode=arch=compute_20,code=sm_20
                -gencode=arch=compute_30,code=sm_30
                -gencode=arch=compute_50,code=sm_50
                -gencode=arch=compute_52,code=sm_52
                -gencode=arch=compute_60,code=sm_60
                -gencode=arch=compute_61,code=sm_61
                -gencode=arch=compute_61,code=compute_61
                >)
        endif()

        add_interface_library(tomopy-cuda-9)
        target_compile_options(tomopy-cuda-9 INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:
            $<IF:$<STREQUAL:${CUDA_ARCH},${CUDA_AUTO_ARCH}>,-arch=sm_50,-arch=sm_${_ARCH_NUM}>
            -gencode=arch=compute_50,code=sm_50
            -gencode=arch=compute_52,code=sm_52
            -gencode=arch=compute_60,code=sm_60
            -gencode=arch=compute_61,code=sm_61
            -gencode=arch=compute_70,code=sm_70
            -gencode=arch=compute_70,code=compute_70
            >)

        add_interface_library(tomopy-cuda-10)
        target_compile_options(tomopy-cuda-10 INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:
            $<IF:$<STREQUAL:"${CUDA_ARCH}","${CUDA_AUTO_ARCH}">,-arch=sm_50,-arch=sm_${_ARCH_NUM}>
            -gencode=arch=compute_60,code=sm_60
            -gencode=arch=compute_61,code=sm_61
            -gencode=arch=compute_70,code=sm_70
            -gencode=arch=compute_75,code=sm_75
            -gencode=arch=compute_75,code=compute_75
            >)

        string(REPLACE "." ";" CUDA_MAJOR_VERSION "${CUDA_VERSION}")
        list(GET CUDA_MAJOR_VERSION 0 CUDA_MAJOR_VERSION)

        if(CUDA_MAJOR_VERSION VERSION_GREATER 10 OR CUDA_MAJOR_VERSION MATCHES 10)
            target_link_libraries(tomopy-cuda INTERFACE tomopy-cuda-10)
        elseif(CUDA_MAJOR_VERSION MATCHES 9)
            target_link_libraries(tomopy-cuda INTERFACE tomopy-cuda-9)
        else()
            if(TIMEMORY_DEPRECATED_CUDA_SUPPORT)
                if(CUDA_MAJOR_VERSION MATCHES 8)
                    target_link_libraries(tomopy-cuda INTERFACE tomopy-cuda-8)
                elseif(CUDA_MAJOR_VERSION MATCHES 7)
                    target_link_libraries(tomopy-cuda INTERFACE tomopy-cuda-7)
                endif()
            else()
                message(WARNING "CUDA version < 9 detected. Enable TIMEMORY_DEPRECATED_CUDA_SUPPORT")
            endif()
        endif()

        list(APPEND EXTERNAL_LIBRARIES tomopy-cuda)
        target_compile_definitions(tomopy-cuda INTERFACE TOMOPY_USE_CUDA)

        if(TOMOPY_USE_NVTX)
            find_library(NVTX_LIBRARY
                NAMES nvToolsExt
                PATHS ${CUDA_TOOLKIT_ROOT_DIR} ${CUDA_SDK_ROOT_DIR} /usr/local/cuda
                HINTS ${CUDA_TOOLKIT_ROOT_DIR} ${CUDA_SDK_ROOT_DIR} /usr/local/cuda
                PATH_SUFFIXES lib lib64)
            if(NVTX_LIBRARY)
                list(APPEND EXTERNAL_LIBRARIES ${NVTX_LIBRARY})
                list(APPEND ${PROJECT_NAME}_DEFINITIONS TOMOPY_USE_NVTX)
            else()
                set(TOMOPY_USE_NVTX OFF)
            endif()
        endif()

        list(APPEND ${PROJECT_NAME}_CUDA_FLAGS
            --default-stream per-thread
            --expt-extended-lambda)

        if(NOT WIN32)
            list(APPEND ${PROJECT_NAME}_CUDA_FLAGS
                --compiler-bindir=${CMAKE_CXX_COMPILER})
        endif()

        if(TOMOPY_CUDA_LINEINFO)
            list(APPEND ${PROJECT_NAME}_CUDA_FLAGS -lineinfo)
            list(APPEND ${PROJECT_NAME}_CXX_FLAGS -fno-omit-frame-pointer -funwind-tables -g)
        endif()

        if(TOMOPY_USE_CUDA_MAX_REGISTER_COUNT)
            set(CUDA_MAX_REGISTER_COUNT "24" CACHE STRING "CUDA maximum register count")
            list(APPEND ${PROJECT_NAME}_CUDA_FLAGS --maxrregcount=${CUDA_MAX_REGISTER_COUNT})
        endif()

    endif()

    find_package(CUDA REQUIRED QUIET)
    list(APPEND EXTERNAL_LIBRARIES ${CUDA_npp_LIBRARY})
    list(APPEND EXTERNAL_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS}
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()


################################################################################
#
#        External variables
#
################################################################################

# user customization to force link libs
to_list(_LINKLIBS "${TOMOPY_USER_LIBRARIES};$ENV{TOMOPY_USER_LIBRARIES}")
foreach(_LIB ${_LINKLIBS})
    list(APPEND EXTERNAL_LIBRARIES ${_LIB})
endforeach()

# including the directories
safe_remove_duplicates(EXTERNAL_INCLUDE_DIRS ${EXTERNAL_INCLUDE_DIRS})
safe_remove_duplicates(EXTERNAL_LIBRARIES ${EXTERNAL_LIBRARIES})
foreach(_DIR ${EXTERNAL_INCLUDE_DIRS})
    include_directories(SYSTEM ${_DIR})
endforeach()

# include dirs
set(TARGET_INCLUDE_DIRECTORIES
    ${PROJECT_SOURCE_DIR}/source/include
    ${PROJECT_SOURCE_DIR}/source/PTL/source
    ${EXTERNAL_INCLUDE_DIRS})
