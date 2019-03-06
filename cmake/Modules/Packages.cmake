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
        set(CMAKE_THREAD_PREFER_PTHREAD ON)
        set(THREADS_PREFER_PTHREAD_FLAG OFF CACHE BOOL "Use -pthread vs. -lpthread" FORCE)
    endif()

    find_package(Threads)
    if(Threads_FOUND)
        list(APPEND EXTERNAL_LIBRARIES Threads::Threads)
    endif()
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
    if(UNIX)
        get_filename_component(PYTHON_ROOT_DIR ${PYTHON_ROOT_DIR} DIRECTORY)
    endif()
    list(APPEND CMAKE_PREFIX_PATH
        ${PYTHON_ROOT_DIR}              # common path for UNIX
        ${PYTHON_ROOT_DIR}/Library      # common path for Windows
        $ENV{CONDA_PREFIX}              # fallback if set
    )
endif()


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
        endif()

        # only define if GPU enabled
        if(TOMOPY_USE_GPU)
            list(APPEND ${PROJECT_NAME}_DEFINITIONS TOMOPY_USE_OPENMP)
        endif()
    else()
        message(WARNING "OpenMP not found")
        set(TOMOPY_USE_OPENMP OFF)
    endif()

endif()


################################################################################
#
#        OpenACC
#
################################################################################

if(TOMOPY_USE_OPENACC AND TOMOPY_USE_GPU)
    find_package(OpenACC)

    if(OpenACC_FOUND)
        foreach(LANG C CXX)
            if(OpenACC_${LANG}_FOUND)
                list(APPEND ${PROJECT_NAME}_${LANG}_FLAGS ${OpenACC_${LANG}_FLAGS})
                list(APPEND ${PROJECT_NAME}_DEFINITIONS TOMOPY_USE_OPENACC)
            endif()
        endforeach()
    else()
        message(WARNING "OpenACC not found")
        set(TOMOPY_USE_OPENACC OFF)
    endif()

endif()


################################################################################
#
#        TBB
#
################################################################################

if(TOMOPY_USE_TBB)
    set(TBB_ROOT_DIR ${PYTHON_ROOT_DIR})
    find_package(TBB COMPONENTS malloc)

    if(TBB_malloc_FOUND)
        list(APPEND EXTERNAL_INCLUDE_DIRS ${TBB_INCLUDE_DIRS})
        list(APPEND EXTERNAL_LIBRARIES ${TBB_LIBRARIES})
        list(APPEND ${PROJECT_NAME}_DEFINITIONS TOMOPY_USE_TBB)
    else()
        message(WARNING "TBB not found")
        set(TOMOPY_USE_TBB OFF)
    endif()

endif()


################################################################################
#
#        MKL
#
################################################################################

find_package(MKL REQUIRED)

if(MKL_FOUND)
    list(APPEND EXTERNAL_INCLUDE_DIRS ${MKL_INCLUDE_DIRS})
    list(APPEND EXTERNAL_LIBRARIES ${MKL_LIBRARIES})
    list(APPEND ${PROJECT_NAME}_DEFINITIONS TOMOPY_USE_MKL)
    list(APPEND ${PROJECT_NAME}_DEFINITIONS USE_MKL)
endif()


################################################################################
#
#        CUDA
#
################################################################################

if(TOMOPY_USE_CUDA AND TOMOPY_USE_GPU)

    get_property(LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)

    if("CUDA" IN_LIST LANGUAGES)
        list(APPEND ${PROJECT_NAME}_DEFINITIONS TOMOPY_USE_CUDA)
        add_feature(${PROJECT_NAME}_CUDA_FLAGS "CUDA NVCC compiler flags")
        add_feature(CUDA_ARCH "CUDA architecture (e.g. sm_35)")

        set(CUDA_ARCH "sm_35" CACHE STRING "CUDA architecture flag")

        if(TOMOPY_USE_NVTX)
            find_library(NVTX_LIBRARY
                NAMES nvToolsExt
                PATHS /usr/local/cuda
                HINTS /usr/local/cuda
                PATH_SUFFIXES lib lib64)
        else()
            unset(NVTX_LIBRARY CACHE)
        endif()

        if(NVTX_LIBRARY)
            list(APPEND EXTERNAL_CUDA_LIBRARIES ${NVTX_LIBRARY})
            list(APPEND ${PROJECT_NAME}_DEFINITIONS TOMOPY_USE_NVTX)
        endif()

        list(APPEND ${PROJECT_NAME}_CUDA_FLAGS
            -arch=${CUDA_ARCH}
            --default-stream per-thread
            --compiler-bindir=${CMAKE_CXX_COMPILER})

        add_option(TOMOPY_USE_CUDA_MAX_REGISTER_COUNT "Enable setting maximum register count" OFF)
        if(TOMOPY_USE_CUDA_MAX_REGISTER_COUNT)
            add_feature(CUDA_MAX_REGISTER_COUNT "CUDA maximum register count")
            set(CUDA_MAX_REGISTER_COUNT "24" CACHE STRING "CUDA maximum register count")
            list(APPEND ${PROJECT_NAME}_CUDA_FLAGS
            --maxrregcount=${CUDA_MAX_REGISTER_COUNT})
        endif()

    endif()

    find_package(CUDA REQUIRED)
    if(CUDA_FOUND)
        list(APPEND EXTERNAL_CUDA_LIBRARIES ${CUDA_npp_LIBRARY})
        list(APPEND EXTERNAL_CUDA_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS}
            ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    else()
        set(TOMOPY_USE_CUDA OFF)
    endif()
endif()


################################################################################
#
#        OpenCV
#
################################################################################
if(TOMOPY_USE_OPENCV)
    set(OpenCV_COMPONENTS opencv_core opencv_imgproc)
    find_package(OpenCV COMPONENTS ${OpenCV_COMPONENTS})

    if(OpenCV_FOUND)
        list(APPEND EXTERNAL_LIBRARIES ${OpenCV_LIBRARIES})
        list(APPEND ${PROJECT_NAME}_DEFINITIONS TOMOPY_USE_OPENCV)
    else()
        message(STATUS "OpenCV not found")
        set(TOMOPY_USE_OPENCV OFF)
    endif()
endif()


################################################################################
#
#        Intel IPP
#
################################################################################

if(TOMOPY_USE_IPP)
    find_package(IPP COMPONENTS core i s cv)

    if(IPP_FOUND)
        list(APPEND EXTERNAL_INCLUDE_DIRS ${IPP_INCLUDE_DIRS})
        list(APPEND EXTERNAL_LIBRARIES ${IPP_LIBRARIES})
        list(APPEND ${PROJECT_NAME}_DEFINITIONS TOMOPY_USE_IPP)
    else()
        set(TOMOPY_USE_IPP OFF)
    endif()
endif()


################################################################################
#
#        ITTNOTIFY (for VTune)
#
################################################################################
if(TOMOPY_USE_ITTNOTIFY)
    find_package(ittnotify)

    if(ittnotify_FOUND)
        list(APPEND EXTERNAL_INCLUDE_DIRS ${ITTNOTIFY_INCLUDE_DIRS})
        list(APPEND EXTERNAL_LIBRARIES ${ITTNOTIFY_LIBRARIES})
    else()
        message(WARNING "ittnotify not found. Set \"VTUNE_AMPLIFIER_201{7,8,9}_DIR\" or \"VTUNE_AMPLIFIER_XE_201{7,8,9}_DIR\" in environment")
    endif()
endif()


################################################################################
#
#        External variables
#
################################################################################

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
    ${EXTERNAL_INCLUDE_DIRS}
    ${EXTERNAL_CUDA_INCLUDE_DIRS})
