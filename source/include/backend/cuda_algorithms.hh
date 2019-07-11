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

#include "backend/cuda.hh"
#include "backend/device.hh"
#include "backend/ranges.hh"
#include "constants.hh"

#include <cstdint>

namespace cuda
{
//--------------------------------------------------------------------------------------//
//
template <typename _Tp, typename _Up = _Tp>
GLOBAL_CALLABLE void
atomic_sum(_Tp* dst, const _Up* src, uintmax_t size, const _Tp factor)
{
    auto range = grid_strided_range<device::gpu, 0>(size);
    for(auto i = range.begin(); i < range.end(); i += range.stride())
        atomicAdd(&dst[i], static_cast<_Tp>(factor * src[i]));
}

namespace impl
{
//--------------------------------------------------------------------------------------//
//
template <typename _Tp, typename _Func>
inline void
rotate(_Tp* dst, const _Tp* src, const float theta_rad, const int nx, const int ny,
       _Func&& nppi_func, int eInterp = interpolation::nn(), stream_t stream = 0)
{
    nppSetStream(stream);
    TOMOPY_NVXT_RANGE_PUSH(&nvtx_rotate);
    CUDA_CHECK_LAST_ERROR(stream);

    auto get_rotation_matrix_2D = [&](double m[2][3], double scale) {
        double alpha    = scale * cos(theta_rad);
        double beta     = scale * sin(theta_rad);
        double center_x = (0.5 * nx) - 0.5;
        double center_y = (0.5 * ny) - 0.5;

        m[0][0] = alpha;
        m[0][1] = beta;
        m[0][2] = (1.0 - alpha) * center_x - beta * center_y;
        m[1][0] = -beta;
        m[1][1] = alpha;
        m[1][2] = beta * center_x + (1.0 - alpha) * center_y;
    };

    NppiSize siz;
    siz.width  = nx;
    siz.height = ny;

    NppiRect roi;
    roi.x      = 0;
    roi.y      = 0;
    roi.width  = nx;
    roi.height = ny;

    int    step = nx * sizeof(_Tp);
    double rot[2][3];
    get_rotation_matrix_2D(rot, 1.0);

    NppStatus ret = nppi_func(src, siz, step, roi, dst, step, roi, rot, eInterp);
    if(ret != NPP_SUCCESS)
        fprintf(stderr, "[%lu] %s returned non-zero NPP status: %i @ %s:%i. src = %p\n",
                this_thread_id(), __FUNCTION__, ret, __FILE__, __LINE__, (void*) src);

    TOMOPY_NVXT_RANGE_POP(stream);
}

//--------------------------------------------------------------------------------------//

inline GLOBAL_CALLABLE void
compute_sum_dist(int dy, int dx, int nx, int ny, const int32_t* ones, uint32_t* sum_dist,
                 int p)
{
    extern __shared__ int _ones_shared[];
    auto                  range = grid_strided_range<device::gpu, 0>(nx);
    for(int n = range.begin(); n < range.end(); n += range.stride())
    {
        for(int d = 0; d < dx; ++d)
        {
            const int32_t* _ones = ones + (d * nx);
            _ones_shared[n]      = _ones[n];
            __syncthreads();
            for(int s = 0; s < dy; ++s)
            {
                uint32_t* _sum_dist = sum_dist + (s * nx * ny) + (d * nx);
                atomicAdd(&_sum_dist[n], (_ones_shared[n] > 0) ? 1 : 0);
            }
        }
    }
}

}  // namespace impl

//--------------------------------------------------------------------------------------//

inline void
rotate(float* dst, const float* src, const float theta_rad, const int nx, const int ny,
       stream_t stream, const int eInterp)
{
    impl::rotate<float>(dst, src, theta_rad, nx, ny, &nppiWarpAffine_32f_C1R, eInterp,
                        stream);
}

//--------------------------------------------------------------------------------------//
//
//
inline void
rotate(int32_t* dst, const int32_t* src, const float theta_rad, const int nx,
       const int ny, stream_t stream, const int eInterp)
{
    impl::rotate<int32_t>(dst, src, theta_rad, nx, ny, &nppiWarpAffine_32s_C1R, eInterp,
                          stream);
}

//======================================================================================//

inline uint32_t*
compute_sum_dist(int dy, int dt, int dx, int nx, int ny, const float* theta,
                 kernel_params& params)
{
    // due to some really strange issue with streams, we use the default stream here
    // because after this has been executed more than once (i.e. we do SIRT and then
    // MLEM or MLEM and then SIRT), NPP returns error code -1000.
    // it has nothing to do with algorithm strangely... and only occurs here
    // where we rotate integers. This does not affect floats...

    auto block = params.block;
    auto grid  = params.compute(nx, block);
    auto smem  = nx * sizeof(int32_t);

    int32_t*  rot = cuda::malloc<int32_t>(nx * ny);
    int32_t*  tmp = cuda::malloc<int32_t>(nx * ny);
    uint32_t* sum = cuda::malloc<uint32_t>(dy * nx * ny);

    cuda::memset(tmp, 1, nx * ny, 0);
    cuda::memset(sum, 0, dy * nx * ny, 0);

    assert(rot != nullptr);
    assert(tmp != nullptr);
    assert(sum != nullptr);

    for(int p = 0; p < dt; ++p)
    {
        float theta_p_rad = fmodf(theta[p], twopi);
        cuda::memset<int32_t>(rot, 0, nx * nx, 0);
        rotate(rot, tmp, -theta_p_rad, nx, ny, 0, interpolation::nn());
        impl::compute_sum_dist<<<grid, block, smem>>>(dy, dx, nx, ny, rot, sum, p);
    }

    CUDA_CHECK_LAST_ERROR(0);  // debug mode only
    cuda::device_sync();

    // destroy
    cuda::free(tmp);
    cuda::free(rot);

    return sum;
}

}  // namespace cuda
