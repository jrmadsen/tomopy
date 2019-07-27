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
#include "backend/opencv/functional.hh"
#include "backend/ranges.hh"
#include "constants.hh"
#include "typedefs.hh"

#include <cstdint>

//--------------------------------------------------------------------------------------//

#define DEFINE_OPENCV_DATA_TYPE(pod_type, opencv_type)                                   \
    template <>                                                                          \
    struct data_type<pod_type>                                                           \
    {                                                                                    \
        template <typename _Up = pod_type>                                               \
        static constexpr int value()                                                     \
        {                                                                                \
            return opencv_type;                                                          \
        }                                                                                \
    };

//--------------------------------------------------------------------------------------//

namespace opencv
{
//--------------------------------------------------------------------------------------//
// non-atomic types
//
template <typename _Tp, typename _Up>
void
atomic_add(_Tp& dst, const _Up& val)
{
    dst += static_cast<_Tp>(val);
}

//--------------------------------------------------------------------------------------//
// atomic types
//
template <typename _Tp, typename _Up>
void
atomic_add(std::atomic<_Tp>& dst, const _Up& val)
{
    auto _dst = dst.load(std::memory_order_relaxed);
    while(!dst.compare_exchange_strong(_dst, _dst + val, std::memory_order_relaxed))
        _dst = dst.load(std::memory_order_relaxed);
}

//--------------------------------------------------------------------------------------//
//
template <typename _Tp, typename _Up, enable_if_t<(is_pointer_v<_Up>::value), int> = 0>
void
atomic_add(_Tp* dst, const _Up& src, uintmax_t size)
{
    auto range = grid_strided_range<device::cpu, 0>(size);
    for(auto i = range.begin(); i < range.end(); i += range.stride())
        atomic_add(dst[i], src[i]);
}

//--------------------------------------------------------------------------------------//
//
template <typename _Tp, typename _Up, enable_if_t<(!is_pointer_v<_Up>::value), int> = 0>
void
atomic_add(_Tp* dst, const _Up& factor, uintmax_t size)
{
    auto range = grid_strided_range<device::cpu, 0>(size);
    for(auto i = range.begin(); i < range.end(); i += range.stride())
        atomic_add(dst[i], factor);
}

//--------------------------------------------------------------------------------------//

namespace impl
{
//--------------------------------------------------------------------------------------//

template <typename _Tp>
struct data_type
{
    template <typename _Up = _Tp>
    static constexpr int value()
    {
        static_assert(std::is_same<_Up, _Tp>::value, "OpenCV data type not overloaded");
        return -1;
    }
};

//--------------------------------------------------------------------------------------//

// floating point types
DEFINE_OPENCV_DATA_TYPE(float, CV_32F)
DEFINE_OPENCV_DATA_TYPE(double, CV_64F)
// signed integer type
DEFINE_OPENCV_DATA_TYPE(int8_t, CV_8S)
DEFINE_OPENCV_DATA_TYPE(int16_t, CV_16S)
DEFINE_OPENCV_DATA_TYPE(int32_t, CV_32S)
// unsigned integer types
DEFINE_OPENCV_DATA_TYPE(uint8_t, CV_8U)
DEFINE_OPENCV_DATA_TYPE(uint16_t, CV_16U)

//--------------------------------------------------------------------------------------//
//
inline cv::Mat
affine_transform(const cv::Mat& warp_src, const float& theta, const float& _center,
                 const int& nx, const int& ny, const int& eInterp, const float& scale)
{
    cv::Mat   warp_dst = cv::Mat::zeros(nx, ny, warp_src.type());
    cv::Point center   = cv::Point(_center, _center);
    cv::Mat   rot      = cv::getRotationMatrix2D(center, theta, scale);
    cv::warpAffine(warp_src, warp_dst, rot, warp_src.size(), eInterp);
    return warp_dst;
}

//--------------------------------------------------------------------------------------//

}  // namespace impl

//--------------------------------------------------------------------------------------//

template <typename _Tp>
void
rotate(_Tp* dst, const _Tp* src, const float& theta, const float& center, const int& nx,
       const int& ny, const int& eInterp, const float& scale = 1.0f)
{
    cv::Mat warp_src = cv::Mat::zeros(nx, ny, impl::data_type<_Tp>::value());
    std::memcpy(warp_src.ptr(), src, nx * ny * sizeof(float));
    cv::Mat warp_rot =
        impl::affine_transform(warp_src, theta * degrees, center, nx, ny, eInterp, scale);
    std::memcpy(dst, warp_rot.ptr(), nx * ny * sizeof(float));
}

//--------------------------------------------------------------------------------------//

inline int32_t*
compute_sum_dist(int dy, int dt, int dx, int nx, int ny, const float* theta,
                 const float center)
{
    auto compute = [&](const array_t<int32_t>& ones, int32_t* sum_dist, int p) {
        for(int s = 0; s < dy; ++s)
        {
            for(int d = 0; d < dx; ++d)
            {
                int32_t*       _sum_dist = sum_dist + (s * nx * ny) + (d * nx);
                const int32_t* _ones     = ones.data() + (d * nx);
                for(int n = 0; n < nx; ++n)
                {
                    _sum_dist[n] += (_ones[n] > 0) ? 1 : 0;
                }
            }
        }
    };

    array_t<int32_t> rot(nx * ny, 0);
    array_t<int32_t> tmp(nx * ny, 1);
    int32_t*         sum_dist = new int32_t[dy * nx * ny];
    opencv::memset(sum_dist, 0, dy * nx * ny);

    for(int p = 0; p < dt; ++p)
    {
        float theta_p = fmodf(theta[p], twopi);
        rotate(rot.data(), tmp.data(), -theta_p, center, nx, ny, interpolation::nn());
        compute(rot, sum_dist, p);
    }

    return sum_dist;
}

//--------------------------------------------------------------------------------------//

}  // namespace opencv

#undef DEFINE_OPENCV_DATA_TYPE  // don't pollute
