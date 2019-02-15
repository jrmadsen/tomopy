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
//   TOMOPY header file
//
//  Description:
//
//      Here we declare the GPU interface
//

#pragma once

#include "common.hh"
#include "gpu.hh"
#include "utils.hh"

//======================================================================================//

#if !defined(TOMOPY_USE_CUDA)
#    if !defined(__global__)
#        define __global__
#    endif
#    if !defined(__device__)
#        define __device__
#    endif
#endif

//======================================================================================//

#if defined(__NVCC__) && defined(TOMOPY_USE_CUDA)

//======================================================================================//

inline int
GetBlockSize()
{
    static thread_local int _instance = GetEnv<int>("CUDA_BLOCK_SIZE", 32);
    return _instance;
}

//======================================================================================//

inline int
GetGridSize()
{
    // default value of zero == calculated according to block and loop size
    static thread_local int _instance = GetEnv<int>("CUDA_GRID_SIZE", 0);
    return _instance;
}

//======================================================================================//

inline int
ComputeGridSize(int size, int block_size = GetBlockSize())
{
    return (size + block_size - 1) / block_size;
}

//======================================================================================//
// interpolation types
#    define GPU_NN NPPI_INTER_NN
#    define GPU_LINEAR NPPI_INTER_LINEAR
#    define GPU_CUBIC NPPI_INTER_CUBIC

//======================================================================================//

inline int
GetNppInterpolationMode()
{
    static int eInterp =
        GetEnv<int>("TOMOPY_NPP_INTER", GetEnv<int>("TOMOPY_INTER", GPU_CUBIC));
    return eInterp;
}

//======================================================================================//

class gpu_data
{
public:
    // typedefs
    typedef gpu_data                                     this_type;
    typedef int32_t                                      int_type;
    typedef std::shared_ptr<gpu_data>                    gpu_data_ptr_t;
    typedef std::vector<gpu_data_ptr_t>                  gpu_data_array_t;
    typedef std::tuple<gpu_data_array_t, float*, float*> init_data_t;

public:
    // ctors, dtors, assignment
    gpu_data(int device, int dy, int dt, int dx, int nx, int ny, const float* data,
             float* recon)
    : m_device(device)
    , m_grid(GetGridSize())
    , m_block(GetBlockSize())
    , m_dy(dy)
    , m_dt(dt)
    , m_dx(dx)
    , m_nx(nx)
    , m_ny(ny)
    , m_use_rot(nullptr)
    , m_use_tmp(nullptr)
    , m_rot(nullptr)
    , m_tmp(nullptr)
    , m_update(nullptr)
    , m_sum_dist(nullptr)
    , m_recon(recon)
    , m_data(data)
    {
        cuda_set_device(m_device);
        m_streams = create_streams(m_num_streams, cudaStreamNonBlocking);
        m_use_rot = gpu_malloc<int_type>(m_nx * m_ny);
        m_use_tmp = gpu_malloc<int_type>(m_nx * m_ny);
        m_rot     = gpu_malloc<float>(m_nx * m_ny);
        m_tmp     = gpu_malloc<float>(m_nx * m_ny);
        m_update  = gpu_malloc<float>(m_dy * m_nx * m_ny);
        gpu_memset<int_type>(m_use_tmp, 1, nx * ny, *m_streams);
    }

    ~gpu_data()
    {
        cudaFree(m_use_rot);
        cudaFree(m_use_tmp);
        cudaFree(m_rot);
        cudaFree(m_tmp);
        cudaFree(m_update);
        cudaFree(m_sum_dist);
        destroy_streams(m_streams, m_num_streams);
    }

    gpu_data(const this_type&) = delete;
    gpu_data(this_type&&)      = default;

    this_type& operator=(const this_type&) = delete;
    this_type& operator=(this_type&&) = default;

public:
    // access functions
    int          device() const { return m_device; }
    int          grid() const { return compute_grid(m_dx); }
    int          block() const { return m_block; }
    int_type*    use_rot() const { return m_use_rot; }
    int_type*    use_tmp() const { return m_use_tmp; }
    float*       rot() const { return m_rot; }
    float*       tmp() const { return m_tmp; }
    float*       update() const { return m_update; }
    float*       sum_dist() const { return m_sum_dist; }
    float*       recon() { return m_recon; }
    const float* recon() const { return m_recon; }
    const float* data() const { return m_data; }
    cudaStream_t stream(int n = 0) { return m_streams[n % m_num_streams]; }

public:
    // assistant functions
    int compute_grid(int size) const
    {
        return (m_grid < 1) ? ((size + m_block - 1) / m_block) : m_grid;
    }

    void sync(int stream_id = -1)
    {
        auto _sync = [&](cudaStream_t _stream) { stream_sync(_stream); };

        if(stream_id >= 0)
            _sync(m_streams[stream_id % m_num_streams]);
        else
            for(int i = 0; i < m_num_streams; ++i)
                _sync(m_streams[i]);
    }

    void alloc_sum_dist()
    {
        cuda_set_device(m_device);
        m_sum_dist = gpu_malloc<float>(m_dy * m_nx * m_ny);
    }

    void reset()
    {
        gpu_memset<float>(m_update, 0, m_dy * m_nx * m_ny, *m_streams);
        if(m_sum_dist)
            gpu_memset<float>(m_sum_dist, 0, m_dy * m_nx * m_ny, *m_streams);
    }

public:
    // static functions
    static init_data_t initialize(int device, int nthreads, int dy, int dt, int dx,
                                  int ngridx, int ngridy, float* cpu_recon,
                                  const float* cpu_data)
    {
        float* recon   = gpu_malloc<float>(dy * ngridx * ngridy);
        float* data    = gpu_malloc<float>(dy * dt * dx);
        auto   streams = create_streams(2, cudaStreamNonBlocking);
        cpu2gpu_memcpy<float>(recon, cpu_recon, dy * ngridx * ngridy, streams[0]);
        cpu2gpu_memcpy<float>(data, cpu_data, dy * dt * dx, streams[1]);
        gpu_data_array_t _gpu_data(nthreads);
        for(int ii = 0; ii < nthreads; ++ii)
            _gpu_data[ii] = gpu_data_ptr_t(
                new gpu_data(device, dy, dt, dx, ngridx, ngridy, data, recon));
        stream_sync(streams[0]);
        stream_sync(streams[1]);
        destroy_streams(streams, 2);
        return init_data_t(_gpu_data, recon, data);
    }

    static void reset(gpu_data_array_t& data)
    {
        // reset "update" to zero
        for(auto& itr : data)
            itr->reset();
    }

    static void sync(gpu_data_array_t& data)
    {
        // sync all the streams
        for(auto& itr : data)
            itr->sync();
    }

protected:
    // data
    int           m_device;
    int           m_grid;
    int           m_block;
    int           m_dy;
    int           m_dt;
    int           m_dx;
    int           m_nx;
    int           m_ny;
    int_type*     m_use_rot;
    int_type*     m_use_tmp;
    float*        m_rot;
    float*        m_tmp;
    float*        m_update;
    float*        m_sum_dist;
    float*        m_recon;
    const float*  m_data;
    int           m_num_streams = 2;
    cudaStream_t* m_streams     = nullptr;
};

//======================================================================================//
//  reduction
//======================================================================================//

__global__ void
deviceReduceKernel(const float* in, float* out, int N);

//--------------------------------------------------------------------------------------//

__global__ void
sum_kernel_block(float* sum, const float* input, int n);

//--------------------------------------------------------------------------------------//

DLL float
deviceReduce(const float* in, float* out, int N);

//--------------------------------------------------------------------------------------//

DLL float
reduce(float* _in, float* _out, int size);

//======================================================================================//
//  rotate
//======================================================================================//

DLL gpu_data::int_type*
    cuda_rotate(const gpu_data::int_type* src, const float theta_rad, const float theta_deg,
                const int nx, const int ny, cudaStream_t stream = 0,
                const int eInterp = GetNppInterpolationMode());

//--------------------------------------------------------------------------------------//

DLL float*
cuda_rotate(const float* src, const float theta_rad, const float theta_deg, const int nx,
            const int ny, cudaStream_t stream = 0,
            const int eInterp = GetNppInterpolationMode());

//--------------------------------------------------------------------------------------//

DLL void
cuda_rotate_ip(gpu_data::int_type* dst, const gpu_data::int_type* src,
               const float theta_rad, const float theta_deg, const int nx, const int ny,
               cudaStream_t stream = 0, const int eInterp = GetNppInterpolationMode());

//--------------------------------------------------------------------------------------//

DLL void
cuda_rotate_ip(float* dst, const float* src, const float theta_rad, const float theta_deg,
               const int nx, const int ny, cudaStream_t stream = 0,
               const int eInterp = GetNppInterpolationMode());

//======================================================================================//
// mult kernels
//======================================================================================//

template <typename _Tp>
__global__ void
cuda_mult_kernel(_Tp* dst, uintmax_t size, const _Tp factor)
{
    auto i0      = blockIdx.x * blockDim.x + threadIdx.x;
    auto istride = blockDim.x * gridDim.x;
    for(auto i = i0; i < size; i += istride)
        dst[i] = factor * dst[i];
}

//======================================================================================//
// sum kernels
//======================================================================================//

template <typename _Tp>
__global__ void
cuda_sum_kernel(_Tp* dst, const _Tp* src, uintmax_t size, const _Tp factor)
{
    auto i0      = blockIdx.x * blockDim.x + threadIdx.x;
    auto istride = blockDim.x * gridDim.x;
    for(auto i = i0; i < size; i += istride)
        dst[i] += factor * src[i];
}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
__global__ void
cuda_atomic_sum_kernel(_Tp* dst, const _Tp* src, uintmax_t size, const _Tp factor)
{
    auto i0      = blockIdx.x * blockDim.x + threadIdx.x;
    auto istride = blockDim.x * gridDim.x;
    for(auto i = i0; i < size; i += istride)
        atomicAdd(&dst[i], factor * src[i]);
}
//--------------------------------------------------------------------------------------//

#endif  // __CUDACC__

//--------------------------------------------------------------------------------------//
