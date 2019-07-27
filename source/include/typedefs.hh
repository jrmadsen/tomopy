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
//   TOMOPY header

/** \file typedefs.hh
 * \headerfile typedefs.hh "include/typedefs.hh"
 * typedef (shorthand type definitions) used by C++ code
 */

#pragma once

#include "macros.hh"

#include <cstdint>
#include <memory>
#include <thread>
#include <unordered_map>
#include <vector>

//======================================================================================//

template <typename _Tp>
using array_t = std::vector<_Tp>;
template <typename _Tp>
using cuda_device_info = std::unordered_map<int, _Tp>;

using uarray_t      = array_t<uint32_t>;
using iarray_t      = array_t<int32_t>;
using farray_t      = array_t<float>;
using darray_t      = array_t<double>;
using num_threads_t = decltype(std::thread::hardware_concurrency());
using mutex_t       = std::mutex;

#if !defined(TOMOPY_USE_PTL)
using AutoLock = std::unique_lock<mutex_t>;

template <typename _Tp>
mutex_t&
TypeMutex()
{
    static mutex_t _instance;
    return _instance;
}
#else
#    include "PTL/AutoLock.hh"
#endif

//======================================================================================//

namespace impl
{
/// Alias template for enable_if
template <bool B, typename T>
using enable_if_t = typename std::enable_if<B, T>::type;

/// Alias template for decay
template <class T>
using decay_t = typename std::decay<T>::type;

}  // namespace impl

template <bool B, typename T = char>
using enable_if_t = impl::enable_if_t<B, T>;

template <typename _Tp>
struct is_pointer_v
{
    using value_type                  = decltype(std::is_pointer<_Tp>::value);
    constexpr static value_type value = std::is_pointer<_Tp>::value;
};

//======================================================================================//
