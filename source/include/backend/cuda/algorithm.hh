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

#include "backend/cuda/functional.hh"
#include "backend/device.hh"
#include "backend/ranges.hh"
#include "constants.hh"

#include <cstdint>

namespace cuda
{
//--------------------------------------------------------------------------------------//
//
template <typename _Tp, typename _Up, enable_if_t<(is_pointer_v<_Up>::value), int> = 0>
GLOBAL_CALLABLE void
add(_Tp* dst, const _Up src, uintmax_t size)
{
    auto range = grid_strided_range<device::gpu, 0>(size);
#pragma unroll
    for(auto i = range.begin(); i < range.end(); i += range.stride())
    {
        // printf("[ptr] %s :: %lld of %lld\n", __FUNCTION__, i, size);
        dst[i] += static_cast<_Tp>(src[i]);
    }
}

//--------------------------------------------------------------------------------------//
//
template <typename _Tp, typename _Up, enable_if_t<(!is_pointer_v<_Up>::value), int> = 0>
GLOBAL_CALLABLE void
add(_Tp* dst, const _Up factor, uintmax_t size)
{
    auto range = grid_strided_range<device::gpu, 0>(size);
#pragma unroll
    for(auto i = range.begin(); i < range.end(); i += range.stride())
    {
        // printf("[pod] %s :: %lld of %lld\n", __FUNCTION__, i, size);
        dst[i] += static_cast<_Tp>(factor);
    }
}

//--------------------------------------------------------------------------------------//
//
template <typename _Tp, typename _Up, enable_if_t<(is_pointer_v<_Up>::value), int> = 0>
GLOBAL_CALLABLE void
atomic_add(_Tp* dst, const _Up src, uintmax_t size)
{
    auto range = grid_strided_range<device::gpu, 0>(size);
#pragma unroll
    for(auto i = range.begin(); i < range.end(); i += range.stride())
    {
        // printf("[ptr] %s :: %lld of %lld\n", __FUNCTION__, i, size);
        atomicAdd(&dst[i], static_cast<_Tp>(src[i]));
    }
}

//--------------------------------------------------------------------------------------//
//
template <typename _Tp, typename _Up, enable_if_t<(!is_pointer_v<_Up>::value), int> = 0>
GLOBAL_CALLABLE void
atomic_add(_Tp* dst, const _Up factor, uintmax_t size)
{
    auto range = grid_strided_range<device::gpu, 0>(size);
#pragma unroll
    for(auto i = range.begin(); i < range.end(); i += range.stride())
    {
        // printf("[pod] %s :: %lld of %lld\n", __FUNCTION__, i, size);
        atomicAdd(&dst[i], static_cast<_Tp>(factor));
    }
}

namespace impl
{
//--------------------------------------------------------------------------------------//
//
template <typename _Tp, typename _Func>
inline void
rotate(_Tp* dst, const _Tp* src, const float theta_rad, const float center, const int nx,
       const int ny, _Func&& nppi_func, int eInterp = interpolation::nn(),
       stream_t stream = 0)
{
    CUDA_CHECK_LAST_ERROR(stream);

    if(stream != nppGetStream())
        nppSetStream(stream);

    TOMOPY_NVXT_RANGE_PUSH(&nvtx_rotate);
    CUDA_CHECK_LAST_ERROR(stream);

    auto get_rotation_matrix_2D = [&](double m[2][3], double scale) {
        double alpha    = scale * cos(theta_rad);
        double beta     = scale * sin(theta_rad);
        double center_x = center;
        double center_y = center;

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
    // extern __shared__ int _ones_shared[];
    auto range = grid_strided_range<device::gpu, 0>(nx);
    for(int n = range.begin(); n < range.end(); n += range.stride())
    {
        for(int d = 0; d < dx; ++d)
        {
            const int32_t* _ones = ones + (d * nx);
            //_ones_shared[n]      = _ones[n];
            //__syncthreads();
            for(int s = 0; s < dy; ++s)
            {
                uint32_t* _sum_dist = sum_dist + (s * nx * ny) + (d * nx);
                atomicAdd(&_sum_dist[n], (_ones[n] > 0) ? 1 : 0);
            }
        }
    }
}

}  // namespace impl

//--------------------------------------------------------------------------------------//

inline void
rotate(float* dst, const float* src, const float& theta_rad, const float& center,
       const int& nx, const int& ny, const int& eInterp, stream_t stream)
{
    impl::rotate<float>(dst, src, theta_rad, center, nx, ny, &nppiWarpAffine_32f_C1R,
                        eInterp, stream);
}

//--------------------------------------------------------------------------------------//
//
//
inline void
rotate(int32_t* dst, const int32_t* src, const float& theta_rad, const float& center,
       const int& nx, const int& ny, const int& eInterp, stream_t stream)
{
    impl::rotate<int32_t>(dst, src, theta_rad, center, nx, ny, &nppiWarpAffine_32s_C1R,
                          eInterp, stream);
}

//======================================================================================//

template <typename _Sum = uint32_t, typename _Tmp = int32_t>
inline void
compute_sum_dist(_Sum* sum, _Tmp* rot, _Tmp* tmp, int dy, int dt, int dx, int nx, int ny,
                 const float* theta, const float& center, kernel_params& params,
                 stream_t stream = 0)
{
    // due to some really strange issue with streams, we use the default stream here
    // because after this has been executed more than once (i.e. we do SIRT and then
    // MLEM or MLEM and then SIRT), NPP returns error code -1000.
    // it has nothing to do with algorithm strangely... and only occurs here
    // where we rotate integers. This does not affect floats...

    for(int p = 0; p < dt; ++p)
    {
        float theta_p_rad = fmodf(theta[p], twopi);
        cuda::memset(rot, 0, nx * nx, stream);
        CUDA_CHECK_LAST_ERROR(stream);

        rotate(rot, tmp, -theta_p_rad, center, nx, ny, interpolation::nn(), stream);
        CUDA_CHECK_LAST_ERROR(stream);

        launch(params, nx, stream, impl::compute_sum_dist, dy, dx, nx, ny, rot, sum, p);
        CUDA_CHECK_LAST_ERROR(stream);
    }
}

namespace impl
{
template <size_t block_size, typename _Tp>
DEVICE_CALLABLE void
wrap_reduce(volatile _Tp* sdata, size_t tid)
{
    if(block_size >= 64)
        sdata[tid] += sdata[tid + 32];
    if(block_size >= 32)
        sdata[tid] += sdata[tid + 16];
    if(block_size >= 16)
        sdata[tid] += sdata[tid + 8];
    if(block_size >= 8)
        sdata[tid] += sdata[tid + 4];
    if(block_size >= 4)
        sdata[tid] += sdata[tid + 2];
    if(block_size >= 2)
        sdata[tid] += sdata[tid + 1];
}

template <size_t block_size, typename _Tp>
GLOBAL_CALLABLE void
reduce(_Tp* g_idata, _Tp* g_odata, size_t n)
{
    __shared__ _Tp sdata[block_size];

    size_t tid = threadIdx.x;
    // size_t i = blockIdx.x*(block_size*2) + tid;
    // size_t gridSize = block_size*2*gridDim.x;
    size_t i        = blockIdx.x * (block_size) + tid;
    size_t gridSize = block_size * gridDim.x;
    sdata[tid]      = 0;

    while(i < n)
    {
        sdata[tid] += g_idata[i];
        i += gridSize;
    }
    // while (i < n) { sdata[tid] += g_idata[i] + g_idata[i+block_size]; i += gridSize; }
    __syncthreads();

    if(block_size >= 1024)
    {
        if(tid < 512)
        {
            sdata[tid] += sdata[tid + 512];
        }
        __syncthreads();
    }
    if(block_size >= 512)
    {
        if(tid < 256)
        {
            sdata[tid] += sdata[tid + 256];
        }
        __syncthreads();
    }
    if(block_size >= 256)
    {
        if(tid < 128)
        {
            sdata[tid] += sdata[tid + 128];
        }
        __syncthreads();
    }
    if(block_size >= 128)
    {
        if(tid < 64)
        {
            sdata[tid] += sdata[tid + 64];
        }
        __syncthreads();
    }

    if(tid < 32)
        wrap_reduce<block_size>(sdata, tid);
    if(tid == 0)
        g_odata[blockIdx.x] = sdata[0];
}
}  // namespace impl

template <size_t block_size, typename _Tp>
void
reduce(_Tp* src, _Tp* dst, size_t N, _Tp* tmp, stream_t stream = 0,
       bool atomic_dst_update = true)
{
    size_t n         = N;
    size_t grid_size = std::ceil(static_cast<float>(n) / block_size);
    _Tp*   from      = src;
    cuda::memset(tmp, 0, grid_size, stream);

    do
    {
        grid_size = std::ceil(static_cast<float>(n) / block_size);
        impl::reduce<block_size><<<grid_size, block_size, 0, stream>>>(from, tmp, n);
        CUDA_CHECK_LAST_ERROR(stream);
        from = tmp;
        n    = grid_size;
    } while(n > block_size);

    if(n > 1)
    {
        impl::reduce<block_size><<<1, block_size, 0, stream>>>(tmp, tmp, n);
        CUDA_CHECK_LAST_ERROR(stream);
    }

    if(atomic_dst_update)
    {
        atomic_add<_Tp, _Tp*><<<1, 1, 0, stream>>>(dst, tmp, 1);
    }
    else
    {
        add<_Tp, _Tp*><<<1, 1, 0, stream>>>(dst, tmp, 1);
    }
    CUDA_CHECK_LAST_ERROR(stream);
}

}  // namespace cuda
