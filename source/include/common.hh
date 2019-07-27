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

#pragma once

#include "constants.hh"
#include "environment.hh"
#include "macros.hh"
#include "typedefs.hh"

#include "backend/opencv.hh"
#if defined(TOMOPY_USE_CUDA)
#    include "backend/cuda.hh"
#endif

BEGIN_EXTERN_C
#include "cxx_extern.h"
END_EXTERN_C

#include <algorithm>
#include <deque>

#if defined(__NVCC__)
#    define TOMOPY_LAMBDA __host__ __device__
#else
#    define TOMOPY_LAMBDA
#endif

//======================================================================================//
// function for printing an array
//
template <typename _Tp, std::size_t _N>
std::ostream&
operator<<(std::ostream& os, const std::array<_Tp, _N>& arr)
{
    std::stringstream ss;
    ss.setf(os.flags());
    for(std::size_t i = 0; i < _N; ++i)
    {
        ss << arr[i];
        if(i + 1 < _N)
        {
            ss << ", ";
        }
    }
    os << ss.str();
    return os;
}

//======================================================================================//
// for generic printing operations in a clean fashion
//
namespace impl
{
//
//----------------------------------------------------------------------------------//
/// Alias template for enable_if
template <bool B, typename T>
using enable_if_t = typename std::enable_if<B, T>::type;

/// Alias template for decay
template <class T>
using decay_t = typename std::decay<T>::type;

struct apply_impl
{
    //----------------------------------------------------------------------------------//
    //  end of recursive expansion
    //
    template <std::size_t _N, std::size_t _Nt, typename _Operator, typename _TupleA,
              typename _TupleB, typename... _Args, enable_if_t<(_N == _Nt), int> = 0>
    static void unroll(_TupleA&& _tupA, _TupleB&& _tupB, _Args&&... _args)
    {
        // call constructor
        using TypeA        = decltype(std::get<_N>(_tupA));
        using TypeB        = decltype(std::get<_N>(_tupB));
        using OperatorType = typename std::tuple_element<_N, _Operator>::type;
        OperatorType(std::forward<TypeA>(std::get<_N>(_tupA)),
                     std::forward<TypeB>(std::get<_N>(_tupB)),
                     std::forward<_Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//
    //  recursive expansion until _N == _Nt
    //
    template <std::size_t _N, std::size_t _Nt, typename _Operator, typename _TupleA,
              typename _TupleB, typename... _Args, enable_if_t<(_N < _Nt), int> = 0>
    static void unroll(_TupleA&& _tupA, _TupleB&& _tupB, _Args&&... _args)
    {
        // call constructor
        using TypeA        = decltype(std::get<_N>(_tupA));
        using TypeB        = decltype(std::get<_N>(_tupB));
        using OperatorType = typename std::tuple_element<_N, _Operator>::type;
        OperatorType(std::forward<TypeA>(std::get<_N>(_tupA)),
                     std::forward<TypeB>(std::get<_N>(_tupB)),
                     std::forward<_Args>(_args)...);
        // recursive call
        unroll<_N + 1, _Nt, _Operator, _TupleA, _TupleB, _Args...>(
            std::forward<_TupleA>(_tupA), std::forward<_TupleB>(_tupB),
            std::forward<_Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//
    // unroll loop
    template <size_t _N, size_t _Nt, typename _Func, typename... _Args,
              typename std::enable_if<(_N == 1), char>::type = 0>
    TOMOPY_LAMBDA static void loop_unroll(_Func&& __func, _Args&&... __args)
    {
        std::forward<_Func>(__func)(_Nt - _N, std::forward<_Args>(__args)...);
    }

    template <size_t _N, size_t _Nt, typename _Func, typename... _Args,
              typename std::enable_if<(_N > 1), char>::type = 0>
    TOMOPY_LAMBDA static void loop_unroll(_Func&& __func, _Args&&... __args)
    {
        std::forward<_Func>(__func)(_Nt - _N, std::forward<_Args>(__args)...);
        loop_unroll<_N - 1, _Nt, _Func, _Args...>(std::forward<_Func>(__func),
                                                  std::forward<_Args>(__args)...);
    }
};

//======================================================================================//

struct apply
{
    //----------------------------------------------------------------------------------//
    // invoke the recursive expansion
    template <typename _Operator, typename _TupleA, typename _TupleB, typename... _Args,
              std::size_t _N  = std::tuple_size<decay_t<_TupleA>>::value,
              std::size_t _Nb = std::tuple_size<decay_t<_TupleB>>::value>
    static void unroll(_TupleA&& _tupA, _TupleB&& _tupB, _Args&&... _args)
    {
        static_assert(_N == _Nb, "tuple_size A must match tuple_size B");
        apply_impl::template unroll<0, _N - 1, _Operator, _TupleA, _TupleB, _Args...>(
            std::forward<_TupleA>(_tupA), std::forward<_TupleB>(_tupB),
            std::forward<_Args>(_args)...);
    }

    //----------------------------------------------------------------------------------//

    template <size_t _N, typename _Func, typename... _Args>
    TOMOPY_LAMBDA static void loop_unroll(_Func&& __func, _Args&&... __args)
    {
        apply_impl::template loop_unroll<_N, _N, _Func, _Args...>(
            std::forward<_Func>(__func), std::forward<_Args>(__args)...);
    }
};

//--------------------------------------------------------------------------------------//
// generic operator for printing
//
template <typename Type>
struct GenericPrinter
{
    GenericPrinter(const std::string& _prefix, const Type& obj, std::ostream& os,
                   intmax_t _prefix_width, intmax_t _obj_width,
                   std::ios_base::fmtflags format_flags, bool endline)
    {
        std::stringstream ss;
        ss.setf(format_flags);
        ss << std::setw(_prefix_width) << std::right << _prefix << " = "
           << std::setw(_obj_width) << obj;
        if(endline)
            ss << std::endl;
        os << ss.str();
    }
};

}  // end namespace impl
