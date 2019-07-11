//  Copyright (c) 2015, UChicago Argonne, LLC. All rights reserved.
//  Copyright 2015. UChicago Argonne, LLC. This software was produced
//  under U.S. Government contract DE-AC02-06CH11357 for Argonne National
//  Laboratory (ANL), which is operated by UChicago Argonne, LLC for the
//  U.S. Department of Energy. The U.S. Government has rights to use,
//  reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR
//  UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR
//  ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is
//  modified to produce derivative works, such modified software should
//  be clearly marked, so as not to confuse it with the version available
//  from ANL.
//  Additionally, redistribution and use in source and binary forms, with
//  or without modification, are permitted provided that the following
//  conditions are met:
//      * Redistributions of source code must retain the above copyright
//        notice, this list of conditions and the following disclaimer.
//      * Redistributions in binary form must reproduce the above copyright
//        notice, this list of conditions and the following disclaimer in
//        the documentation andwith the
//        distribution.
//      * Neither the name of UChicago Argonne, LLC, Argonne National
//        Laboratory, ANL, the U.S. Government, nor the names of its
//        contributors may be used to endorse or promote products derived
//        from this software without specific prior written permission.
//  THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS
//  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
//  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
//  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago
//  Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
//  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
//  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
//  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
//  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
//  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//  POSSIBILITY OF SUCH DAMAGE.
// ---------------------------------------------------------------
//  TOMOPY class header
//

#pragma once

#include "backend/device.hh"
#include "backend/ranges.hh"
#include "environment.hh"

#include <atomic>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <npp.h>
#include <nppi.h>
#include <vector>
#include <vector_types.h>

//======================================================================================//
//
//      NVTX
//
//======================================================================================//

#if defined(TOMOPY_USE_NVTX)

#    include <nvToolsExt.h>

#    define TOMOPY_NVXT_RANGE_PUSH(obj) nvtxRangePushEx(obj)
#    define TOMOPY_NVXT_RANGE_POP(obj)                                                   \
        cudaStreamSynchronize(obj);                                                      \
        nvtxRangePop()
#    define TOMOPY_NVXT_NAME_THREAD(num, name) nvtxNameOsThread(num, name)
#    define TOMOPY_NVXT_MARK(msg) nvtxMark(name)

extern void
init_nvtx();

extern nvtxEventAttributes_t nvtx_total;
extern nvtxEventAttributes_t nvtx_iteration;
extern nvtxEventAttributes_t nvtx_slice;
extern nvtxEventAttributes_t nvtx_projection;
extern nvtxEventAttributes_t nvtx_update;
extern nvtxEventAttributes_t nvtx_rotate;

#else
#    define TOMOPY_NVXT_RANGE_PUSH(...)
#    define TOMOPY_NVXT_RANGE_POP(...)
#    define TOMOPY_NVXT_NAME_THREAD(...)
#    define TOMOPY_NVXT_MARK(...)
#endif

//======================================================================================//
//
//      CUDA error checking
//
//======================================================================================//

// this is always defined, even in release mode
#if !defined(CUDA_CHECK_CALL)
#    define CUDA_CHECK_CALL(err)                                                         \
        {                                                                                \
            if(cudaSuccess != err)                                                       \
            {                                                                            \
                std::stringstream ss;                                                    \
                ss << "cudaCheckError() failed at " << __FUNCTION__ << "@'" << __FILE__  \
                   << "':" << __LINE__ << " : " << cudaGetErrorString(err);              \
                fprintf(stderr, "%s\n", ss.str().c_str());                               \
                throw std::runtime_error(ss.str().c_str());                              \
            }                                                                            \
        }
#endif

// this is only defined in debug mode
#if defined(DEBUG)
#    if !defined(CUDA_CHECK_LAST_ERROR)
#        define CUDA_CHECK_LAST_ERROR(stream)                                            \
            {                                                                            \
                cudaStreamSynchronize(stream);                                           \
                cudaError err = cudaGetLastError();                                      \
                if(cudaSuccess != err)                                                   \
                {                                                                        \
                    std::stringstream ss;                                                \
                    ss << "cudaCheckError() failed at " << __FUNCTION__ << "@'"          \
                       << __FILE__ << "':" << __LINE__ << " : "                          \
                       << cudaGetErrorString(err);                                       \
                    fprintf(stderr, "%s\n", ss.str().c_str());                           \
                    throw std::runtime_error(ss.str().c_str());                          \
                }                                                                        \
            }
#    endif
#else
#    if !defined(CUDA_CHECK_LAST_ERROR)
#        define CUDA_CHECK_LAST_ERROR(...)
#    endif
#endif

namespace cuda
{
//--------------------------------------------------------------------------------------//
//
// define some types for when CUDA is enabled
using stream_t = cudaStream_t;
using event_t  = cudaEvent_t;
using error_t  = cudaError_t;
using memcpy_t = cudaMemcpyKind;
// define some values for when CUDA is enabled
static const decltype(cudaSuccess)       success_v          = cudaSuccess;
static const decltype(cudaErrorNotReady) err_not_ready_v    = cudaErrorNotReady;
static const cudaMemcpyKind              host_to_host_v     = cudaMemcpyHostToHost;
static const cudaMemcpyKind              host_to_device_v   = cudaMemcpyHostToDevice;
static const cudaMemcpyKind              device_to_host_v   = cudaMemcpyDeviceToHost;
static const cudaMemcpyKind              device_to_device_v = cudaMemcpyDeviceToDevice;

template <bool B, typename T = char>
using enable_if_t = impl::enable_if_t<B, T>;

template <typename _Tp>
using array_t = std::vector<_Tp>;

inline int
device_count();

inline int
device_sm_count(int device = 0);

//--------------------------------------------------------------------------------------//
//
struct interpolation
{
    static int nn() { return NPPI_INTER_NN; }
    static int linear() { return NPPI_INTER_LINEAR; }
    static int cubic() { return NPPI_INTER_CUBIC; }

    static int mode(const std::string& preferred)
    {
        EnvChoiceList<int> choices = {
            EnvChoice<int>(nn(), "NN", "nearest neighbor interpolation"),
            EnvChoice<int>(linear(), "LINEAR", "bilinear interpolation"),
            EnvChoice<int>(cubic(), "CUBIC", "bicubic interpolation")
        };
        return GetChoice<int>(choices, preferred);
    }
};

//--------------------------------------------------------------------------------------//

struct kernel_params
{
    kernel_params() = default;

    explicit kernel_params(uint32_t _block, uint32_t _grid = 0)
    : block(_block)
    , grid(_grid)
    {}

    int compute(const int& size)
    {
        return (grid == 0) ? ((size + block - 1) / block) : grid;
    }

    static int compute(const int& size, const int& block_size)
    {
        return ((size + block_size - 1) / block_size);
    }

    uint32_t block = get_env<uint32_t>("TOMOPY_BLOCK_SIZE", 32);
    uint32_t grid  = get_env<uint32_t>("TOMOPY_GRID_SIZE", 0);  // 0 == compute
};

//--------------------------------------------------------------------------------------//

inline uint32_t&
this_thread_device()
{
    // this creates a globally accessible function for determining the device
    // the thread is assigned to
    //
    static std::atomic<uint32_t> _ntid(0);
    static thread_local uint32_t _instance =
        (device_count() > 0) ? ((_ntid++) % device_count()) : 0;
    return _instance;
}

//--------------------------------------------------------------------------------------//
//
//      functions dealing with cuda errors
//
//--------------------------------------------------------------------------------------//

/// check the success of a cudaError_t
inline bool
check_call(error_t err)
{
    return (err == success_v);
}

//--------------------------------------------------------------------------------------//
/// get last error but don't reset last error to cudaSuccess
inline error_t
peek_at_last_error()
{
    return cudaPeekAtLastError();
}

//--------------------------------------------------------------------------------------//
/// get last error and reset to cudaSuccess
inline error_t
get_last_error()
{
    return cudaGetLastError();
}

//--------------------------------------------------------------------------------------//
/// get the error string
inline const char*
get_error_string(error_t err)
{
    return cudaGetErrorString(err);
}

//--------------------------------------------------------------------------------------//
//
//      functions dealing with the device
//
//--------------------------------------------------------------------------------------//

/// get the number of devices available
inline int
device_count()
{
    int  dc  = 0;
    auto ret = cudaGetDeviceCount(&dc);
    consume_parameters(ret);
    return dc;
}

//--------------------------------------------------------------------------------------//

inline cudaDeviceProp
device_properties(int device = 0)
{
    using map_t                             = std::map<int, cudaDeviceProp>;
    using pointer_t                         = std::unique_ptr<map_t>;
    static thread_local pointer_t _instance = pointer_t(new map_t());

    if(_instance->find(device) != _instance->end())
        return _instance->find(device)->second;

    cudaSetDevice(device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    return ((*_instance)[device] = deviceProp);
}

/// print info about devices available (only does this once per process)
extern void
device_query();

//--------------------------------------------------------------------------------------//
/// get the number of devices available
inline int
device_sm_count(int device)
{
    if(device_count() == 0)
        return 0;
    return device_properties(device).multiProcessorCount;
}

//--------------------------------------------------------------------------------------//
/// sets the thread to a specific device
inline void
set_device(int device)
{
    CUDA_CHECK_CALL(cudaSetDevice(device));
}

//--------------------------------------------------------------------------------------//
/// sync the device
inline void
device_sync()
{
    CUDA_CHECK_CALL(cudaDeviceSynchronize());
}

//--------------------------------------------------------------------------------------//
/// reset the device
inline void
device_reset()
{
    CUDA_CHECK_CALL(cudaDeviceReset());
}

//--------------------------------------------------------------------------------------//
//
//      functions dealing with cuda streams
//
//--------------------------------------------------------------------------------------//

/// create a cuda stream
inline array_t<stream_t>
stream_create(size_t nstreams, unsigned int flags = cudaStreamNonBlocking)
{
    array_t<stream_t> streams(nstreams);
    for(auto& itr : streams)
    {
        CUDA_CHECK_CALL(cudaStreamCreateWithFlags(&itr, flags));
    }
    return streams;
}

//--------------------------------------------------------------------------------------//
/// destroy a cuda stream
inline void
stream_destroy(array_t<stream_t>& streams)
{
    for(auto& itr : streams)
        CUDA_CHECK_CALL(cudaStreamDestroy(itr));
    streams.clear();
}

//--------------------------------------------------------------------------------------//
/// sync the cuda stream
inline void
stream_sync(stream_t stream)
{
    CUDA_CHECK_CALL(cudaStreamSynchronize(stream));
}

//--------------------------------------------------------------------------------------//
//
//      functions dealing with cuda allocations
//
//--------------------------------------------------------------------------------------//

/// cuda malloc
template <typename _Tp>
inline _Tp*
malloc(size_t n)
{
    _Tp* arr;
    CUDA_CHECK_CALL(cudaMalloc(&arr, n * sizeof(_Tp)));
    return arr;
}

/// cuda malloc
template <typename _Tp>
inline void
free(_Tp*& arr)
{
    CUDA_CHECK_CALL(cudaFree(arr));
    arr = nullptr;
}

/// cuda memcpy
template <typename _Tp>
inline void
memcpy(_Tp* dst, const _Tp* src, size_t n, memcpy_t from_to, stream_t stream)
{
    CUDA_CHECK_CALL(cudaMemcpyAsync(dst, src, n * sizeof(_Tp), from_to, stream));
}

/// cuda memset
template <typename _Tp>
inline void
memset(_Tp* dst, const int& value, size_t n, stream_t stream)
{
    CUDA_CHECK_CALL(cudaMemsetAsync(dst, value, n * sizeof(_Tp), stream));
}

//--------------------------------------------------------------------------------------//

}  // namespace cuda

//--------------------------------------------------------------------------------------//
