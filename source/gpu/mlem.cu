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
    // dt == number of angles
    // dy == number of slices
    // dx == number of pixels
    // nx == ngridx
    // ny == ngridy

    auto nx_range = grid_strided_range<device::gpu, 0>(nx);

    for(int i = nx_range.begin(); i < nx_range.end(); i += nx_range.stride())
    {
        auto* _recon = recon + i * nx;
#pragma unroll
        for(int d = 0; d < dx; ++d)
            atomicAdd(&sum[d], _recon[d]);
    }

    __syncthreads();

    for(int i = nx_range.begin(); i < nx_range.end(); i += nx_range.stride())
    {
        auto* _recon = recon + i * nx;
        auto* _data  = data + p * dx;
#pragma unroll
        for(int d = 0; d < dx; ++d)
            _recon[d] += _data[d] / sum[d];
    }
}

//======================================================================================//

GLOBAL_CALLABLE void
update_kernel(float* recon, float* update, const uint32_t* sum_dist, int dx, int size)
{
    if(dx == 0)
        return;
    auto  range = grid_strided_range<device::gpu, 0>(size);
    float fdx   = static_cast<float>(dx);
    //#pragma unroll
    for(int i = range.begin(); i < range.end(); i += range.stride())
    {
        uint32_t sum = sum_dist[i];
        float    upd = update[i];
        if(sum != 0 && upd == upd)
        {
            // atomicAdd(&recon[i], upd / static_cast<float>(sum) / fdx);
            recon[i] *= upd / static_cast<float>(sum) / fdx;
        }
        // reset for next iteration
        update[i] = 0.0f;
    }
}

}  // namespace

//======================================================================================//

void
mlem_cuda(const float* cpu_data, int dy, int dt, int dx, const float* center,
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
    auto interp       = opts->interpolation;
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

    cuda::memcpy(data, cpu_data, dy * dt * dx, cuda::host_to_device_v, get_stream(1));
    cuda::memset(update, 0, recon_pixels, get_stream(2));
    cuda::memcpy(recon, cpu_recon, recon_pixels, cuda::host_to_device_v, get_stream(3));

    cuda::launch(params, recon_pixels, get_stream(3), cuda::atomic_add<float, float>,
                 recon, cuda::fepsilon, recon_pixels);

    // used streams 0, 1, 2, 3 above
    sync_stream(4);

    // no longer needed
    cuda::free(sum_dist_rot);
    cuda::free(sum_dist_tmp);

    TOMOPY_NVXT_RANGE_PUSH(&nvtx_total);

    for(int i = 0; i < num_iter; i++)
    {
        // timing and profiling
        TIMEMORY_AUTO_TIMER("");
        TOMOPY_NVXT_RANGE_PUSH(&nvtx_iteration);
        START_TIMER(t_start);

        // loop over independent projection angles
        for(int p = 0; p < dt; ++p)
        {
            auto stream  = get_stream(p);
            auto offset  = get_offset(p);
            auto theta_p = fmodf(theta[p], twopi);

            auto* p_plus = plus_rot + (offset * recon_pixels);
            auto* p_back = back_rot + (offset * recon_pixels);
            auto  p_sum  = sum_rot + (offset * dy * dx);
            // auto* p_update = update + (offset * recon_pixels);

            // CUDA_CHECK_LAST_ERROR(stream);
            cuda::memset(p_plus, 0, recon_pixels, stream);
            // CUDA_CHECK_LAST_ERROR(stream);
            cuda::memset(p_back, 0, recon_pixels, stream);
            // CUDA_CHECK_LAST_ERROR(stream);
            cuda::memset(p_sum, 0, dy * dx, stream);
            // CUDA_CHECK_LAST_ERROR(stream);

            // loop over independent slices
            for(int s = 0; s < dy; ++s)
            {
                auto* s_data  = data + (s * dt * dx);
                auto* s_recon = recon + (s * nx * ny);
                auto* s_plus  = p_plus + (s * nx * ny);
                auto* s_back  = p_back + (s * nx * ny);
                auto* s_sum   = p_sum + (s * dx);

                cuda::rotate(s_plus, s_recon, -theta_p, center[s], nx, ny, interp,
                             stream);
                // CUDA_CHECK_LAST_ERROR(stream);

                cuda::launch(params, dx * nx, stream, pixels_kernel, p, nx, dx, s_plus,
                             s_data, s_sum);
                // CUDA_CHECK_LAST_ERROR(stream);

                cuda::rotate(s_back, s_plus, theta_p, center[s], nx, ny, interp, stream);
                // CUDA_CHECK_LAST_ERROR(stream);

                cuda::launch(params, nx * ny, stream, cuda::atomic_add<float, float*>,
                             update, s_back, nx * ny);
                // CUDA_CHECK_LAST_ERROR(stream);
            }
        }

        /*
        sync_stream(0, streams.size(), false);
        for(int offset = 0; offset < streams.size(); ++offset)
        {
            auto  stream   = get_stream(offset);
            auto* p_update = update + (offset * recon_pixels);
            cuda::launch(params, recon_pixels, stream, merge_update_kernel, update,
                         p_update, recon_pixels);
            // update the global recon with global update and sum_dist
            //cuda::launch(params, recon_pixels, stream, update_kernel, recon, p_update,
            //             sum_dist, dx, recon_pixels);
        }*/

        sync_stream(0, streams.size(), false);
        // update the global recon with global update and sum_dist
        cuda::launch(params, recon_pixels, 0, update_kernel, recon, update, sum_dist, dx,
                     recon_pixels);
        sync_stream(0, 0);

        // stop profile range and report timing
        TOMOPY_NVXT_RANGE_POP(0);
        REPORT_TIMER(t_start, "iteration", i, num_iter);
    }

    sync_stream(0, streams.size());

    // copy to cpu
    cuda::memcpy(cpu_recon, recon, recon_pixels, cuda::device_to_host_v, 0);

    // sync and destroy main stream
    cuda::stream_sync(0);
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
