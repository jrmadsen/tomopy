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
//  ---------------------------------------------------------------
//   TOMOPY header

/** \file data.hh
 * \headerfile data.hh "include/data.hh"
 * C++ class for storing thread-specific data when doing rotation-based reconstructions
 * CpuData == rotation-based reconstruction with OpenCV
 * GpuData == rotation-based reconstruction with NPP
 */

#pragma once

#include "backend/opencv.hh"
#include "common.hh"
#include "constants.hh"
#include "macros.hh"
#include "typedefs.hh"
#include "utils.hh"

#if defined(TOMOPY_USE_CUDA)
#    include "backend/cuda.hh"
#endif

#include <array>
#include <atomic>
#include <functional>

//======================================================================================//

struct RuntimeOptions
{
    num_threads_t      pool_size     = HW_CONCURRENCY;
    int                interpolation = -1;
    DeviceOption       device;
    std::array<int, 3> block_size = { { 32, 1, 1 } };
    std::array<int, 3> grid_size  = { { 0, 0, 0 } };

    RuntimeOptions(int _pool_size, const char* _interp, const char* _device,
                   int* _grid_size, int* _block_size)
    : pool_size(static_cast<num_threads_t>(_pool_size))
    , device(GetDevice(_device))
    {
        memcpy(grid_size.data(), _grid_size, 3 * sizeof(int));
        memcpy(block_size.data(), _block_size, 3 * sizeof(int));

        if(device.key == "gpu")
        {
#if defined(TOMOPY_USE_CUDA)
            interpolation = cuda::interpolation::mode(_interp);
#else
            interpolation = opencv::interpolation::mode(_interp);
#endif
        }
        else
        {
            interpolation = opencv::interpolation::mode(_interp);
        }
    }

    ~RuntimeOptions() {}

    // disable copying and copy assignment
    RuntimeOptions(const RuntimeOptions&) = delete;
    RuntimeOptions& operator=(const RuntimeOptions&) = delete;

    // create the thread pool -- don't have this in the constructor
    // because you don't want to arbitrarily create thread-pools
    void init()
    {
        if(pool_size == 0)
        {
            pool_size = HW_CONCURRENCY / get_env<size_t>("TOMOPY_PYTHON_THREADS", 1);
        }
    }

    // invoke the generic printer defined in common.hh
    template <typename... _Descriptions, typename... _Objects>
    void print(std::tuple<_Descriptions...>&& _descripts, std::tuple<_Objects...>&& _objs,
               std::ostream& os, intmax_t _prefix_width, intmax_t _obj_width,
               std::ios_base::fmtflags format_flags, bool endline) const
    {
        // tuple of descriptions
        using DescriptType = std::tuple<_Descriptions...>;
        // tuple of objects to print
        using ObjectType = std::tuple<_Objects...>;
        // the operator that does the printing (see end of
        using UnrollType = std::tuple<internal::GenericPrinter<_Objects>...>;

        internal::apply::unroll<UnrollType>(std::forward<DescriptType>(_descripts),
                                            std::forward<ObjectType>(_objs), std::ref(os),
                                            _prefix_width, _obj_width, format_flags,
                                            endline);
    }

    // overload the output operator for the class
    friend std::ostream& operator<<(std::ostream& os, const RuntimeOptions& opts)
    {
        std::stringstream ss;
        opts.print(std::make_tuple("Thread-pool size", "Interpolation mode", "Device",
                                   "Grid size", "Block size"),
                   std::make_tuple(opts.pool_size, opts.interpolation, opts.device,
                                   opts.block_size, opts.grid_size),
                   ss, 30, 20, std::ios::boolalpha, true);
        os << ss.str();
        return os;
    }
};

//======================================================================================//

struct Registration
{
    Registration() {}

    int initialize()
    {
        // make sure this thread has a registered thread-id
        this_thread_id();
        // count the active threads
        return active()++;
    }

    void cleanup(RuntimeOptions* opts)
    {
        auto tid    = this_thread_id();
        auto remain = --active();

        if(remain == 0)
        {
            std::stringstream ss;
            ss << *opts << std::endl;
#if defined(TOMOPY_USE_CUDA)
            for(int i = 0; i < cuda::device_count(); ++i)
            {
                // set the device
                cudaSetDevice(i);
                // sync the device
                cudaDeviceSynchronize();
                // reset the device
                cudaDeviceReset();
            }
#endif
        }
        else
        {
            printf("[%lu] Threads remaining: %i...\n", tid, remain);
        }
    }

    static std::atomic<int>& active()
    {
        static std::atomic<int> _active;
        return _active;
    }
};

//======================================================================================//

#if defined(TOMOPY_USE_PTL)

//--------------------------------------------------------------------------------------//
// when PTL thread-pool is available
//
template <typename DataArray, typename Func, typename... Args>
void
execute(RuntimeOptions* ops, int dt, DataArray& data, Func&& func, Args&&... args)
{
    // get the thread pool
    auto& tp   = ops->thread_pool;
    auto  join = [&]() { stream_sync(0); };
    assert(tp != nullptr);

    try
    {
        tomopy::TaskGroup<void> tg(join, tp.get());
        for(int p = 0; p < dt; ++p)
        {
            auto _func = std::bind(std::forward<Func>(func), std::ref(data),
                                   std::forward<int>(p), std::forward<Args>(args)...);
            tg.run(_func);
        }
        tg.join();
    }
    catch(const std::exception& e)
    {
        std::stringstream ss;
        ss << "\n\nError executing :: " << e.what() << "\n\n";
        {
            AutoLock l(TypeMutex<decltype(std::cout)>());
            std::cerr << e.what() << std::endl;
        }
        throw std::runtime_error(ss.str().c_str());
    }
}

#else

//--------------------------------------------------------------------------------------//
// when PTL thread-pool is not available
//
template <typename DataArray, typename Func, typename... Args>
void
execute(RuntimeOptions* ops, int dt, DataArray& data, Func&& func, Args&&... args)
{
    // sync streams
    auto join = [&]() { /*cuda::stream_sync(0);*/ };

    try
    {
        for(int p = 0; p < dt; ++p)
        {
            auto _func = std::bind(std::forward<Func>(func), std::ref(data),
                                   std::forward<int>(p), std::forward<Args>(args)...);
            _func();
        }
        join();
    }
    catch(const std::exception& e)
    {
        std::stringstream ss;
        ss << "\n\nError executing :: " << e.what() << "\n\n";
        {
            AutoLock l(TypeMutex<decltype(std::cout)>());
            std::cerr << e.what() << std::endl;
        }
        throw std::runtime_error(ss.str().c_str());
    }
}

#endif

//======================================================================================//
