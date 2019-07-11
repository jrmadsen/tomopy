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

#include "backend/cuda.hh"
#include "backend/cuda_algorithms.hh"
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
pixels_kernel(int p, int dy, int dx, int nx, int ny, float* glob_recon, const float* data)
{
    // dt == number of angles
    // dy == number of slices
    // dx == number of pixels
    // nx == ngridx
    // ny == ngridy

    extern __shared__ float sum[];  // size dy * nx

    // set shared to zero
    {
        auto range = grid_strided_range<device::gpu, 0>(dx);
        for(int i = range.begin(); i < range.end(); i += range.stride())
        {
            sum[i] = 0.0f;
        }
    }
    __syncthreads();

    for(int k = 0; k < dy; ++k)
    {
        float* _sum   = sum + k * dy;
        float* _recon = glob_recon + k * nx * ny;
        auto   range  = grid_strided_range<device::gpu, 0>(dx * nx);
        for(int i = range.begin(); i < range.end(); i += range.stride())
        {
            atomicAdd(&_sum[i % dx], _recon[i]);
        }
    }
    __syncthreads();

    for(int k = 0; k < dy; ++k)
    {
        const float* _data  = data + p * dx;
        float*       _sum   = sum + k * dy;
        float*       _recon = glob_recon + k * nx * ny;
        auto         range  = grid_strided_range<device::gpu, 0>(dx * nx);
        for(int i = range.begin(); i < range.end(); i += range.stride())
        {
            _recon[i] += _data[i] / _sum[i % dx];
        }
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
        if(sum != 0 && update[i] == update[i])
            recon[i] *= upd / static_cast<float>(sum) / fdx;
    }
}

} // namespace

//======================================================================================//

void
mlem_cuda(const float* cpu_data, int dy, int dt, int dx, const float*, const float* theta,
          float* cpu_recon, int nx, int ny, int num_iter, RuntimeOptions* opts)
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

    auto      interp       = opts->interpolation;
    uintmax_t recon_pixels = static_cast<uintmax_t>(dy * nx * ny);
    auto      params       = cuda::kernel_params(opts->block_size[0], opts->grid_size[0]);
    auto      streams      = cuda::stream_create(opts->pool_size);
    float*    update       = cuda::malloc<float>(recon_pixels);
    float*    plus_rot     = cuda::malloc<float>(dt * recon_pixels);
    float*    back_rot     = cuda::malloc<float>(dt * recon_pixels);
    float*    recon        = cuda::malloc<float>(recon_pixels);
    float*    data         = cuda::malloc<float>(dy * dt * dx);
    cuda::memcpy(recon, cpu_recon, recon_pixels, cuda::host_to_device_v, 0);
    cuda::memcpy(data, cpu_data, dy * dt * dx, cuda::host_to_device_v, 0);
    uint32_t* sum_dist = cuda::compute_sum_dist(dy, dt, dx, nx, ny, theta, params);

    TOMOPY_NVXT_RANGE_PUSH(&nvtx_total);

    auto get_stream = [&](const size_t& offset) {
        return streams.at(offset % streams.size());
    };

    auto sync_stream = [&](const size_t& beg, const size_t& end) {
        for(size_t i = beg; i < end; ++i)
            cuda::stream_sync(get_stream(i));
    };

    auto get_proj  = [&](float* arr, const int& p) { return arr + p * recon_pixels; };
    auto get_slice = [&](float* arr, const int& s) { return arr + s * nx * ny; };

    for(int i = 0; i < num_iter; i++)
    {
        // timing and profiling
        TIMEMORY_AUTO_TIMER("");
        TOMOPY_NVXT_RANGE_PUSH(&nvtx_iteration);
        START_TIMER(t_start);

        // reset global update and sum_dist
        cuda::memset(update, 0, recon_pixels, get_stream(0));
        cuda::memset(plus_rot, 0, dt * recon_pixels, get_stream(1));
        cuda::memset(back_rot, 0, dt * recon_pixels, get_stream(2));

        // sync
        sync_stream(0, 3);

        // execute the loop over slices and projection angles
        // loop over independent projection angles
        for(int p = 0; p < dt; ++p)
        {
            cuda::stream_t stream  = get_stream(p);
            float          theta_p = fmodf(theta[p], twopi);

            float* p_plus_rot = get_proj(plus_rot, p);
            for(int s = 0; s < dy; ++s)
            {
                const float* s_recon     = recon + s * nx * ny;
                float*       sp_plus_rot = p_plus_rot + s * nx * ny;
                cuda::rotate(sp_plus_rot, s_recon, -theta_p, nx, ny, stream, interp);
            }

            {
                int smem  = dy * dx * sizeof(float);
                int block = params.block;
                int grid  = params.compute(dx * nx);
                pixels_kernel<<<grid, block, smem, stream>>>(p, dy, dx, nx, ny, plus_rot,
                                                             data);
            }

            // calculate offset for the streams
            auto* p_back_rot = get_proj(back_rot, p);
            for(int s = 0; s < dy; ++s)
            {
                float* sp_plus_rot = get_slice(p_plus_rot, s);
                float* sp_back_rot = get_slice(p_back_rot, s);
                cuda::rotate(sp_back_rot, sp_plus_rot, theta_p, nx, ny, stream, interp);
            }

            CUDA_CHECK_LAST_ERROR(stream);
        }

        // update array
        for(int p = 0; p < dt; ++p)
        {
            cuda::stream_t stream = get_stream(p);
            int            block  = params.block;
            int            grid   = params.compute(recon_pixels);
            cuda::atomic_sum<<<grid, block, 0, stream>>>(update, get_proj(back_rot, p),
                                                         recon_pixels);
        }

        // update the global recon with global update and sum_dist
        int block = params.block;
        int grid  = params.compute(recon_pixels);
        update_kernel<<<grid, block>>>(recon, update, sum_dist, dx, recon_pixels);

        // stop profile range and report timing
        TOMOPY_NVXT_RANGE_POP(0);
        REPORT_TIMER(t_start, "iteration", i, num_iter);
    }

    for(auto& itr : streams)
        cuda::stream_sync(itr);
    cuda::stream_sync(0);

    // copy to cpu
    cuda::memcpy(cpu_recon, recon, recon_pixels, cuda::device_to_host_v, 0);

    // sync and destroy main stream
    cuda::stream_destroy(streams);

    // cleanup
    cuda::free(recon);
    cuda::free(data);
    cuda::free(update);
    cuda::free(sum_dist);

    TOMOPY_NVXT_RANGE_POP(0);

    // sync the device
    cuda::device_sync();
}

//======================================================================================//
