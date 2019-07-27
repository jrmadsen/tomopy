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
#include "macros.hh"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>

//--------------------------------------------------------------------------------------//

namespace opencv
{
//--------------------------------------------------------------------------------------//
//
// define some types for when CUDA is enabled
using stream_t      = int;
using event_t       = int;
using error_t       = int;
using memcpy_t      = int;
using device_prop_t = int;
// define some values for when CUDA is enabled
static const int success_v          = 0;
static const int err_not_ready_v    = 1;
static const int host_to_host_v     = 0;
static const int host_to_device_v   = 1;
static const int device_to_host_v   = 2;
static const int device_to_device_v = 3;

template <bool B, typename T = char>
using enable_if_t = impl::enable_if_t<B, T>;

template <typename _Tp>
using array_t = std::vector<_Tp>;

struct interpolation
{
    static int nn() { return CV_INTER_NN; }
    static int linear() { return CV_INTER_LINEAR; }
    static int cubic() { return CV_INTER_CUBIC; }

    static int mode(const std::string& preferred)
    {
        EnvChoiceList<int> choices =
            { EnvChoice<int>(nn(), "NN", "nearest neighbor interpolation"),
              EnvChoice<int>(linear(), "LINEAR", "bilinear interpolation"),
              EnvChoice<int>(cubic(), "CUBIC", "bicubic interpolation") };
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
    static uint32_t _instance = 0;
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
    return 0;
}

//--------------------------------------------------------------------------------------//
/// get last error and reset to cudaSuccess
inline error_t
get_last_error()
{
    return 0;
}

//--------------------------------------------------------------------------------------//
/// get the error string
inline const char*
get_error_string(error_t err)
{
    return "";
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
    return 1;
}

//--------------------------------------------------------------------------------------//

inline device_prop_t
device_properties(int = 0)
{
    return device_prop_t(0);
}

/// print info about devices available (only does this once per process)
inline void
device_query()
{}

//--------------------------------------------------------------------------------------//
/// get the number of devices available
inline int
device_sm_count(int)
{
    return 0;
}

//--------------------------------------------------------------------------------------//
/// sets the thread to a specific device
inline void
set_device(int)
{}

//--------------------------------------------------------------------------------------//
/// sync the device
inline void
device_sync()
{}

//--------------------------------------------------------------------------------------//
/// reset the device
inline void
device_reset()
{}

//--------------------------------------------------------------------------------------//
//
//      functions dealing with CPU allocations
//
//--------------------------------------------------------------------------------------//

/// cpu malloc
template <typename _Tp>
inline _Tp*
malloc(size_t n)
{
    return new _Tp[n];
}

/// cpu malloc
template <typename _Tp>
inline void
free(_Tp*& arr)
{
    delete[] arr;
    arr = nullptr;
}

/// cpu memcpy
template <typename _Tp>
inline void
memcpy(_Tp* dst, const _Tp* src, size_t n, memcpy_t = 0, stream_t = 0)
{
    if(src != dst)
        std::memcpy(dst, src, n * sizeof(_Tp));
}

/// cpu memset
template <typename _Tp>
inline void
memset(_Tp* dst, const int& value, size_t n, stream_t = 0)
{
    std::memset(dst, value, n * sizeof(_Tp));
}

}  // namespace opencv
