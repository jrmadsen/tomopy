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
#include <type_traits>

namespace impl
{
//--------------------------------------------------------------------------------------//
//
template <bool B, class T = char>
using enable_if_t = typename std::enable_if<B, T>::type;

//--------------------------------------------------------------------------------------//
//
template <typename _Intp>
struct range
{
    using this_type = range<_Intp>;

    HOST_DEVICE_CALLABLE range(_Intp _begin, _Intp _end, _Intp _stride)
    : m_begin(_begin)
    , m_end(_end)
    , m_stride(_stride)
    {}

    HOST_DEVICE_CALLABLE _Intp& begin() { return m_begin; }
    HOST_DEVICE_CALLABLE _Intp begin() const { return m_begin; }
    HOST_DEVICE_CALLABLE _Intp& end() { return m_end; }
    HOST_DEVICE_CALLABLE _Intp end() const { return m_end; }
    HOST_DEVICE_CALLABLE _Intp& stride() { return m_stride; }
    HOST_DEVICE_CALLABLE _Intp stride() const { return m_stride; }

protected:
    _Intp m_begin;
    _Intp m_end;
    _Intp m_stride;
};
}  // namespace impl

//======================================================================================//
//
//  These provide loop parameters for grid-strided loops on GPU or traditional loops on
//  CPU
//
//======================================================================================//

template <typename _Device, size_t DIM = 0, typename _Intp = int32_t>
struct grid_strided_range : impl::range<_Intp>
{
    using base_type = impl::range<_Intp>;
    static_assert(DIM < 0 || DIM > 2,
                  "Error DIM parameter must be 0 (x), 1 (y), or 2 (z)");
};

//--------------------------------------------------------------------------------------//
// overload for 0/x
//
template <typename _Device, typename _Intp>
struct grid_strided_range<_Device, 0, _Intp> : impl::range<_Intp>
{
    using base_type = impl::range<_Intp>;
    template <bool B, typename T = char>
    using enable_if_t = impl::enable_if_t<B, T>;

#if defined(TOMOPY_USE_CUDA) && defined(__NVCC__)
    template <typename _Dev                                       = _Device,
              enable_if_t<std::is_same<_Dev, device::gpu>::value> = 0>
    DEVICE_CALLABLE explicit grid_strided_range(_Intp max_iter)
    : base_type(blockIdx.x * blockDim.x + threadIdx.x, max_iter, blockDim.x * gridDim.x)
    {}

    template <typename _Dev                                       = _Device,
              enable_if_t<std::is_same<_Dev, device::cpu>::value> = 0>
#endif
    explicit grid_strided_range(_Intp max_iter)
    : base_type(0, max_iter, 1)
    {}

    using base_type::begin;
    using base_type::end;
    using base_type::stride;
};

//--------------------------------------------------------------------------------------//
// overload for 1/y
//
template <typename _Device, typename _Intp>
struct grid_strided_range<_Device, 1, _Intp> : impl::range<_Intp>
{
    using base_type = impl::range<_Intp>;
    template <bool B, typename T = char>
    using enable_if_t = impl::enable_if_t<B, T>;

#if defined(TOMOPY_USE_CUDA) && defined(__NVCC__)
    template <typename _Dev                                       = _Device,
              enable_if_t<std::is_same<_Dev, device::gpu>::value> = 0>
    DEVICE_CALLABLE explicit grid_strided_range(_Intp max_iter)
    : base_type(blockIdx.y * blockDim.y + threadIdx.y, max_iter, blockDim.y * gridDim.y)
    {}

    template <typename _Dev                                       = _Device,
              enable_if_t<std::is_same<_Dev, device::cpu>::value> = 0>
#endif
    explicit grid_strided_range(_Intp max_iter)
    : base_type(0, max_iter, 1)
    {}

    using base_type::begin;
    using base_type::end;
    using base_type::stride;
};

//--------------------------------------------------------------------------------------//
// overload for 2/z
//
template <typename _Device, typename _Intp>
struct grid_strided_range<_Device, 2, _Intp> : impl::range<_Intp>
{
    using base_type = impl::range<_Intp>;
    template <bool B, typename T = char>
    using enable_if_t = impl::enable_if_t<B, T>;

#if defined(TOMOPY_USE_CUDA) && defined(__NVCC__)
    template <typename _Dev                                       = _Device,
              enable_if_t<std::is_same<_Dev, device::gpu>::value> = 0>
    DEVICE_CALLABLE explicit grid_strided_range(_Intp max_iter)
    : base_type(blockIdx.z * blockDim.z + threadIdx.z, max_iter, blockDim.z * gridDim.z)
    {}

    template <typename _Dev                                       = _Device,
              enable_if_t<std::is_same<_Dev, device::cpu>::value> = 0>
#endif
    explicit grid_strided_range(_Intp max_iter)
    : base_type(0, max_iter, 1)
    {}

    using base_type::begin;
    using base_type::end;
    using base_type::stride;
};
