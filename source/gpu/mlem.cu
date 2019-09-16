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
pixels_kernel(int p, int nx, int dx, float* recon, const float* data)
{
    auto range = grid_strided_range<device::gpu, 0>(dx);
    for(int d = range.begin(); d < range.end(); d += range.stride())
    {
        float sum = 0.0f;
        for(int i = 0; i < nx; ++i)
            sum += recon[i * nx + d];
        if(sum != 0.0f)
        {
            float upd = data[p * dx + d] / sum;
            if(upd == upd)
                for(int i = 0; i < nx; ++i)
                    recon[i * nx + d] += upd;
        }
    }
}

//======================================================================================//

GLOBAL_CALLABLE void
update_kernel(float* recon, float* update, const uint32_t* sum_dist, int dx, int size)
{
    if(dx == 0)
        return;

    float fdx   = static_cast<float>(dx);
    auto  range = grid_strided_range<device::gpu, 0>(size);

    for(int i = range.begin(); i < range.end(); i += range.stride())
    {
        uint32_t sum = sum_dist[i];
        float&   upd = update[i];
        if(sum != 0 && update[i] == update[i])
            recon[i] *= upd / static_cast<float>(sum) / fdx;
        upd = 0.0f;
    }
}

//======================================================================================//

void
compute_projection(int p, int dy, int dt, int dx, int nx, int ny, const float* center,
                   const float* theta, const float* data, const float* recon,
                   float* update, float* rot, float* tmp, cuda::kernel_params& params,
                   cuda::stream_t stream, int eInterp = cuda::interpolation::nn())
{
    // calculate some values
    float theta_p = fmodf(theta[p], twopi);

    // reset destination arrays (NECESSARY! or will cause NaNs)
    // only do once bc for same theta, same pixels get overwritten
    cuda::memset(rot, 0, dy * nx * ny, stream);
    cuda::memset(tmp, 0, dy * nx * ny, stream);

    // use locking to reduce stream changes in NPP
    static mutex_t _mutex;
    auto_lock      lk(_mutex);

    // execute the loop over slices
    for(int s = 0; s < dy; ++s)
    {
        const float* _recon  = recon + s * nx * ny;
        float*       _rot    = rot + s * nx * ny;
        const float* _data   = data + s * dt * dx;
        float*       _update = update + s * nx * ny;
        float*       _tmp    = tmp + s * nx * ny;

        // forward-rotate
        cuda::rotate(_rot, _recon, -theta_p, center[s], nx, ny, eInterp, stream);

        // compute simdata
        cuda::launch(params, nx, stream, pixels_kernel, p, nx, dx, _rot, _data);
        CUDA_CHECK_LAST_ERROR(stream);

        // back-rotate
        cuda::rotate(_tmp, _rot, theta_p, center[s], nx, ny, eInterp, stream);

        // update shared update array
        cuda::launch(params, nx, stream, cuda::atomic_add<float, float*>, _update, _tmp,
                     nx * ny);
        CUDA_CHECK_LAST_ERROR(stream);
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

    uintmax_t recon_pixels = static_cast<uintmax_t>(dy * nx * ny);
    auto      interp       = opts->interpolation;
    auto      params       = cuda::kernel_params(opts->block_size[0], opts->grid_size[0]);
    auto      streams      = cuda::stream_create(opts->pool_size);

    auto get_stream = [&](const size_t& offset) {
        return streams.at(offset % streams.size());
    };
    auto sync_stream = [&](const size_t& end, const size_t& beg = 0,
                           bool sync_implicit = false) {
        for(size_t i = beg; i < (end % (streams.size() + 1)); ++i)
            cuda::stream_sync(streams.at(i));
        if(sync_implicit)
            cuda::stream_sync(0);
    };

    // used by compute_sum_dist
    auto* sum_dist_rot = cuda::malloc<int32_t>(nx * ny);
    auto* sum_dist_tmp = cuda::malloc<int32_t>(nx * ny);
    auto* tmp_rot      = cuda::malloc<float>(streams.size() * recon_pixels);
    auto* tmp_tmp      = cuda::malloc<float>(streams.size() * recon_pixels);

    auto* sum_dist = cuda::malloc<uint32_t>(dy * nx * ny);
    auto* update   = cuda::malloc<float>(recon_pixels);
    auto* recon    = cuda::malloc<float>(recon_pixels);
    auto* data     = cuda::malloc<float>(dy * dt * dx);

    cuda::memset(sum_dist_tmp, 1, nx * ny, get_stream(0));
    cuda::memset(sum_dist, 0, dy * nx * ny, get_stream(0));
    cuda::compute_sum_dist(sum_dist, sum_dist_rot, sum_dist_tmp, dy, dt, dx, nx, ny,
                           theta, center[0], params, get_stream(0));
    // no longer needed
    cuda::free(sum_dist_rot);
    cuda::free(sum_dist_tmp);

    cuda::memcpy(recon, cpu_recon, recon_pixels, cuda::host_to_device_v, get_stream(0));
    cuda::memcpy(data, cpu_data, dy * dt * dx, cuda::host_to_device_v, get_stream(1));
    cuda::memset(update, 0, recon_pixels, get_stream(2));

    TOMOPY_NVXT_RANGE_PUSH(&nvtx_total);

    for(int i = 0; i < num_iter; i++)
    {
        // timing and profiling
        TIMEMORY_AUTO_TIMER("");
        TOMOPY_NVXT_RANGE_PUSH(&nvtx_iteration);
        START_TIMER(t_start);

        // sync
        sync_stream(streams.size());

        // loop over independent projection angles
        for(int p = 0; p < dt; ++p)
        {
            // compute which stream to use
            auto offset = p % streams.size();
            // compute stream-specific data offset
            float* off_rot = tmp_rot + (offset * recon_pixels);
            float* off_tmp = tmp_tmp + (offset * recon_pixels);
            compute_projection(p, dy, dt, dx, nx, ny, center, theta, data, recon, update,
                               off_rot, off_tmp, params, streams.at(offset), interp);
        }

        // sync the thread streams
        sync_stream(streams.size());

        // update the global recon with global update and sum_dist
        cuda::launch(params, recon_pixels, get_stream(0), update_kernel, recon, update,
                     sum_dist, dx, recon_pixels);

        // sync
        sync_stream(1);

        // stop profile range and report timing
        TOMOPY_NVXT_RANGE_POP(0);
        REPORT_TIMER(t_start, "iteration", i, num_iter);
    }

    // sync the thread streams
    sync_stream(streams.size());

    // copy to cpu
    cuda::memcpy(cpu_recon, recon, recon_pixels, cuda::device_to_host_v, get_stream(0));

    // sync the thread streams
    sync_stream(streams.size());

    // sync and destroy main stream
    cuda::stream_destroy(streams);

    // cleanup
    cuda::free(recon);
    cuda::free(data);
    cuda::free(update);
    cuda::free(sum_dist);
    cuda::free(tmp_tmp);
    cuda::free(tmp_rot);

    TOMOPY_NVXT_RANGE_POP(0);

    // sync the device
    cuda::device_sync();
}

//======================================================================================//
