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
//   TOMOPY implementation

#include "backend/opencv/algorithm.hh"
#include "backend/opencv/functional.hh"
#include "backend/ranges.hh"
#include "common.hh"
#include "data.hh"
#include "options.hh"
#include "utils.hh"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <numeric>

//======================================================================================//

// directly call the CPU version
void
mlem_cpu(const float* data, int dy, int dt, int dx, const float* center,
         const float* theta, float* recon, int ngridx, int ngridy, int num_iter,
         RuntimeOptions*);

// directly call the GPU version
void
mlem_cuda(const float* data, int dy, int dt, int dx, const float* center,
          const float* theta, float* recon, int ngridx, int ngridy, int num_iter,
          RuntimeOptions*);

//======================================================================================//

int
cxx_mlem(const float* data, int dy, int dt, int dx, const float* center,
         const float* theta, float* recon, int ngridx, int ngridy, int num_iter,
         int pool_size, const char* interp, const char* device, int* grid_size,
         int* block_size)
{
    auto tid = this_thread_id();
    // registration
    static Registration registration;
    // local count for the thread
    int count = registration.initialize();
    // number of threads started at Python level
    auto tcount = env::get("TOMOPY_PYTHON_THREADS", HW_CONCURRENCY);

    // configured runtime options
    RuntimeOptions opts(pool_size, interp, device, grid_size, block_size);

    // create the thread-pool
    opts.init();
    std::cout << "Options:\n" << opts << std::endl;

    START_TIMER(cxx_timer);
    TIMEMORY_AUTO_TIMER("");

    printf("[%lu]> %s : nitr = %i, dy = %i, dt = %i, dx = %i, nx = %i, ny = %i\n", tid,
           __FUNCTION__, num_iter, dy, dt, dx, ngridx, ngridy);

    if(opts.device.key == "gpu")
        mlem_cuda(data, dy, dt, dx, center, theta, recon, ngridx, ngridy, num_iter,
                  &opts);
    else
        mlem_cpu(data, dy, dt, dx, center, theta, recon, ngridx, ngridy, num_iter, &opts);

    registration.cleanup(&opts);
    REPORT_TIMER(cxx_timer, __FUNCTION__, count, tcount);

    return EXIT_SUCCESS;
}

//======================================================================================//

namespace
{
void
pixels_kernel(int p, int dx, int nx, int ny, float* recon, const float* data, float* sum)
{
    // dt == number of angles
    // dy == number of slices
    // dx == number of pixels
    // nx == ngridx
    // ny == ngridy
    // compute simdata

    auto dx_range = grid_strided_range<device::cpu, 1>(dx);
    auto nx_range = grid_strided_range<device::cpu, 0>(nx);

    for(int i = nx_range.begin(); i < nx_range.end(); i += nx_range.stride())
    {
        auto* _recon = recon + i * nx;
        for(int d = dx_range.begin(); d < dx_range.end(); d += dx_range.stride())
            sum[d] += _recon[d];
    }

    for(int i = nx_range.begin(); i < nx_range.end(); i += nx_range.stride())
    {
        auto* _recon = recon + i * nx;
        auto* _data  = data + p * dx;
        for(int d = dx_range.begin(); d < dx_range.end(); d += dx_range.stride())
            _recon[d] += _data[d] / sum[d];
    }
}

//======================================================================================//

template <typename _Tp, typename _Up>
void
update_kernel(_Tp* recon, const _Up* update, const int32_t* sum_dist, int dx, int size)
{
    if(dx == 0)
        return;
    auto  range = grid_strided_range<device::cpu, 0>(size);
    float fdx   = static_cast<float>(dx);
    for(int i = range.begin(); i < range.end(); i += range.stride())
    {
        auto sum = sum_dist[i];
        _Tp  upd = update[i];
        if(sum != 0)
            recon[i] *= upd / static_cast<float>(sum) / fdx;
    }
}

}  // namespace

//======================================================================================//

void
mlem_cpu(const float* data, int dy, int dt, int dx, const float* center,
         const float* theta, float* recon, int nx, int ny, int num_iter,
         RuntimeOptions* opts)
{
    cv::setNumThreads(4);
    printf("[%lu]> %s : nitr = %i, dy = %i, dt = %i, dx = %i, nx = %i, ny = %i\n",
           this_thread_id(), __FUNCTION__, num_iter, dy, dt, dx, nx, ny);

    TIMEMORY_AUTO_TIMER("");

    auto  interp       = opts->interpolation;
    auto  recon_pixels = static_cast<uintmax_t>(dy * nx * ny);
    auto* update       = opencv::malloc_shared<float>(recon_pixels);
    auto* plus_rot     = opencv::malloc<float>(recon_pixels);
    auto* back_rot     = opencv::malloc<float>(recon_pixels);
    auto* sum_rot      = opencv::malloc<float>(dx);
    auto* sum_dist     = opencv::compute_sum_dist(dy, dt, dx, nx, ny, theta, center[0]);

    using atomic_f4_t = std::atomic<float>;
    auto get_slice    = [&](float* arr, const int& s) { return arr + s * nx * ny; };
    auto get_slice_a  = [&](atomic_f4_t* arr, const int& s) { return arr + s * nx * ny; };

    opencv::atomic_add(recon, opencv::fepsilon, recon_pixels);

    for(int i = 0; i < num_iter; i++)
    {
        // timing and profiling
        TIMEMORY_AUTO_TIMER("");
        START_TIMER(t_start);

        // reset global update and sum_dist
        opencv::memset(update, 0, recon_pixels);

        // loop over independent projection angles
        for(int p = 0; p < dt; ++p)
        {
            float theta_p = fmodf(theta[p], twopi);
            // loop over independent slices
            for(int s = 0; s < dy; ++s)
            {
                opencv::memset(plus_rot, 0, recon_pixels);
                opencv::memset(back_rot, 0, recon_pixels);
                opencv::memset(sum_rot, 0, dx);

                auto* s_recon    = get_slice(recon, s);
                auto* s_update   = get_slice_a(update, s);
                auto* s_data     = data + s * dt * dx;
                auto* s_plus_rot = get_slice(plus_rot, s);
                auto* s_back_rot = get_slice(back_rot, s);

                opencv::rotate(s_plus_rot, s_recon, -theta_p, center[s], nx, ny, interp);
                pixels_kernel(p, dx, nx, ny, s_plus_rot, s_data, sum_rot);
                opencv::rotate(s_back_rot, s_plus_rot, theta_p, center[s], nx, ny,
                               interp);
                opencv::atomic_add(s_update, s_back_rot, nx * ny);
            }
        }

        // update the global recon with global update and sum_dist
        update_kernel(recon, update, sum_dist, dx, recon_pixels);

        // stop profile range and report timing
        REPORT_TIMER(t_start, "iteration", i, num_iter);
    }

    // cleanup
    opencv::free(update);
    opencv::free(sum_dist);
    opencv::free(plus_rot);
    opencv::free(back_rot);

    // sync the device
    opencv::device_sync();
}

//======================================================================================//
#if !defined(TOMOPY_USE_CUDA)
void
mlem_cuda(const float* data, int dy, int dt, int dx, const float* center,
          const float* theta, float* recon, int ngridx, int ngridy, int num_iter,
          RuntimeOptions* opts)
{
    mlem_cpu(data, dy, dt, dx, center, theta, recon, ngridx, ngridy, num_iter, opts);
}
#endif
//======================================================================================//
