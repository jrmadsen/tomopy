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
//   TOMOPY CUDA implementation

#include "backend/cuda/algorithm.hh"
#include "backend/cuda/functional.hh"
#include "backend/ranges.hh"
#include "common.hh"
#include "constants.hh"
#include "data.hh"
#include "utils.hh"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <numeric>

//======================================================================================//

#if defined(TOMOPY_USE_NVTX)
extern nvtxEventAttributes_t nvtx_total;
extern nvtxEventAttributes_t nvtx_iteration;
extern nvtxEventAttributes_t nvtx_slice;
extern nvtxEventAttributes_t nvtx_projection;
extern nvtxEventAttributes_t nvtx_update;
extern nvtxEventAttributes_t nvtx_rotate;
#endif

//======================================================================================//

namespace
{
GLOBAL_CALLABLE void
pixels_kernel(int p, int nx, int dx, float* recon, const float* data, float* sum)
{
    auto nx_range = grid_strided_range<device::gpu, 0>(nx);
    auto dx_range = grid_strided_range<device::gpu, 1>(dx);

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
            _recon[d] += _data[d] - sum[d];
    }
}

//======================================================================================//

GLOBAL_CALLABLE void
update_kernel(float* recon, const float* update, const uint32_t* sum_dist, int dx,
              int size)
{
    if(dx == 0)
        return;
    auto  range = grid_strided_range<device::gpu, 0>(size);
    float fdx   = static_cast<float>(dx);
    for(int i = range.begin(); i < range.end(); i += range.stride())
    {
        uint32_t sum = sum_dist[i];
        float    upd = update[i];
        if(sum != 0)
            recon[i] += upd / static_cast<float>(sum) / fdx;
    }
}

//======================================================================================//

void
compute_projection(int p, int dy, int dt, int dx, int nx, int ny, const float* center,
                   const float* theta, const float* data, const float* recon,
                   float* update, float* rot, float* tmp, float* sum,
                   cuda::kernel_params& params, cuda::stream_t stream,
                   int eInterp = cuda::interpolation::nn())
{
    // calculate some values
    float theta_p = fmodf(theta[p], twopi);
    int   block   = params.block;
    int   grid    = params.compute(nx, block);

    // reset destination arrays (NECESSARY! or will cause NaNs)
    // only do once bc for same theta, same pixels get overwritten
    cuda::memset(rot, 0, dy * nx * ny, stream);
    cuda::memset(tmp, 0, dy * nx * ny, stream);
    cuda::memset(sum, 0, dy * dx, stream);

    int smem = 0;
    for(int s = 0; s < dy; ++s)
    {
        const float* _recon  = recon + s * nx * ny;
        const float* _data   = data + s * dt * dx;
        float*       _update = update + s * nx * ny;
        float*       _rot    = rot + s * nx * ny;
        float*       _tmp    = tmp + s * nx * ny;
        float*       _sum    = sum + s * dx;

        // forward-rotate
        cuda::rotate(_rot, _recon, -theta_p, center[s], nx, ny, stream, eInterp);

        // compute simdata
        pixels_kernel<<<grid, block, smem, stream>>>(p, nx, dx, _rot, _data, _sum);
        CUDA_CHECK_LAST_ERROR(stream);

        // back-rotate
        cuda::rotate(_tmp, _rot, theta_p, center[s], nx, ny, stream, eInterp);

        // update shared update array
        cuda::atomic_add<<<grid, block, 0, stream>>>(_update, _tmp, nx * ny);
        CUDA_CHECK_LAST_ERROR(stream);
    }
}
}

//======================================================================================//

void
sirt_cuda(const float* cpu_data, int dy, int dt, int dx, const float* center,
          const float* theta, float* cpu_recon, int nx, int ny, int num_iter,
          RuntimeOptions* opts)
{
    printf("[%lu]> %s : nitr = %i, dy = %i, dt = %i, dx = %i, nx = %i, ny = %i\n",
           this_thread_id(), __FUNCTION__, num_iter, dy, dt, dx, nx, ny);

    // thread counter for device assignment
    static std::atomic<int> ntid;

    // compute some properties (expected python threads, max threads, device assignment)
    int pythread_num = ntid++;
    int device       = pythread_num % cuda::device_count();  // assign to device

    TIMEMORY_AUTO_TIMER("");

    // GPU allocated copies
    cuda::set_device(device);
    printf("[%lu] Running on device %i...\n", this_thread_id(), device);

    auto recon_pixels = static_cast<uintmax_t>(dy * nx * ny);
    auto params       = cuda::kernel_params(opts->block_size[0], opts->grid_size[0]);
    auto streams      = cuda::stream_create(opts->pool_size);

    auto get_proj   = [&](float* arr, const int& p) { return arr + p * recon_pixels; };
    auto get_slice  = [&](float* arr, const int& s) { return arr + s * nx * ny; };
    auto get_offset = [&](const size_t& offset) { return offset % streams.size(); };
    auto get_stream = [&](const size_t& offset) {
        return streams.at(get_offset(offset));
    };
    auto sync_stream = [&](const size_t& end, const size_t& beg = 0,
                           bool sync_implicit = true) {
        for(size_t i = beg; i < (end % streams.size()); ++i)
            cuda::stream_sync(streams.at(i));
        if(sync_implicit)
            cuda::stream_sync(0);
    };

    // used by compute_sum_dist
    auto* sum_dist_rot = cuda::malloc<int32_t>(nx * ny);
    auto* sum_dist_tmp = cuda::malloc<int32_t>(nx * ny);
    auto* sum_dist     = cuda::malloc<uint32_t>(dy * nx * ny);

    auto* plus_rot = cuda::malloc<float>(streams.size() * recon_pixels);
    auto* back_rot = cuda::malloc<float>(streams.size() * recon_pixels);
    auto* sum_rot  = cuda::malloc<float>(streams.size() * dx);
    auto* data     = cuda::malloc<float>(dy * dt * dx);
    auto* recon    = cuda::malloc<float>(recon_pixels);
    auto* update   = cuda::malloc<float>(recon_pixels);
    cuda::memset(sum_dist_tmp, 1, nx * ny, get_stream(0));
    cuda::memset(sum_dist, 0, dy * nx * ny, get_stream(0));
    cuda::compute_sum_dist(sum_dist, sum_dist_rot, sum_dist_tmp, dy, dt, dx, nx, ny,
                           theta, center[0], params, get_stream(0));
    cuda::memcpy(recon, cpu_recon, recon_pixels, cuda::host_to_device_v, get_stream(1));
    cuda::memcpy(data, cpu_data, dy * dt * dx, cuda::host_to_device_v, get_stream(2));
    cuda::memset(update, 0, recon_pixels, get_stream(3));
    cuda::launch(params, recon_pixels, get_stream(4), cuda::atomic_add<float, float>,
                 recon, cuda::fepsilon, recon_pixels);

    TOMOPY_NVXT_RANGE_PUSH(&nvtx_total);

    // used streams 0, 1, 2, 3, and 4 above
    sync_stream(4);

    // no longer needed
    cuda::free(sum_dist_rot);
    cuda::free(sum_dist_tmp);

    for(int i = 0; i < num_iter; i++)
    {
        // timing and profiling
        TIMEMORY_AUTO_TIMER("");
        TOMOPY_NVXT_RANGE_PUSH(&nvtx_iteration);
        START_TIMER(t_start);

        // reset global update and sum_dist
        cuda::memset(update, 0, recon_pixels, 0);
        cuda::stream_sync(0);

        // execute the loop over slices and projection angles
        // loop over independent projection angles
        for(int p = 0; p < dt; ++p)
        {
            auto offset = p % streams.size();
            // calculate offset for the streams
            float* off_rot = plus_rot + (offset * recon_pixels);
            float* off_tmp = back_rot + (offset * recon_pixels);
            float* off_sum = sum_rot + (offset * dx);
            compute_projection(p, dy, dt, dx, nx, ny, center, theta, data, recon, update,
                               off_rot, off_tmp, off_sum, params, streams.at(offset));
        }

        // sync the thread streams and implicit stream
        sync_stream(streams.size());

        // update the global recon with global update and sum_dist
        cuda::launch(params, recon_pixels, 0, update_kernel, recon, update, sum_dist, dx, recon_pixels);

        // stop profile range and report timing
        TOMOPY_NVXT_RANGE_POP(0);
        REPORT_TIMER(t_start, "iteration", i, num_iter);
    }

    // copy to cpu
    cuda::memcpy(cpu_recon, recon, recon_pixels, cuda::device_to_host_v, 0);

    // sync and destroy main stream
    cuda::stream_destroy(streams);

    // cleanup
    cuda::free(sum_dist);
    cuda::free(recon);
    cuda::free(data);
    cuda::free(update);
    cuda::free(sum_rot);
    cuda::free(back_rot);
    cuda::free(plus_rot);

    TOMOPY_NVXT_RANGE_POP(0);

    // sync the device
    cuda::device_sync();
}

//======================================================================================//
