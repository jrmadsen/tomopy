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

/** \file macros.hh
 * \headerfile macros.hh "include/macros.hh"
 * Include files + some standard macros available to C++
 */

#pragma once

extern "C"
{
#include "macros.h"
}

#include "profiler.hh"

//======================================================================================//
//  C headers

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

//======================================================================================//
//  C++ headers

#include <atomic>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

//======================================================================================//
// determine if C++20 is available
#if __cplusplus > 201703L
#    define CXX_20
#endif

//======================================================================================//
// this function is used by a macro -- returns a unique identifier to the thread
inline uintmax_t
this_thread_id()
{
    static std::atomic<uintmax_t> tcounter;
    static thread_local auto      tid = tcounter++;
    return tid;
}

//======================================================================================//
// get the number of hardware threads
#if !defined(HW_CONCURRENCY)
#    define HW_CONCURRENCY std::thread::hardware_concurrency()
#endif

//======================================================================================//
// debugging
#if !defined(PRINT_HERE)
#    if defined(DEBUG)
#        define PRINT_HERE(extra)                                                        \
            fprintf(stderr, "[%lu]> %s@'%s':%i %s\n", this_thread_id(), __FUNCTION__,    \
                    __FILE__, __LINE__, extra)
#    else
#        define PRINT_HERE(...)
#    endif
#endif

//======================================================================================//
// start a timer
#if !defined(START_TIMER)
#    define START_TIMER(var) auto var = std::chrono::high_resolution_clock::now()
#endif

//======================================================================================//
// report a timer
#if !defined(REPORT_TIMER)
#    define REPORT_TIMER(start_time, note, counter, total_count)                         \
        {                                                                                \
            auto end_time = std::chrono::high_resolution_clock::now();                   \
            std::chrono::duration<double> elapsed_seconds = end_time - start_time;       \
            printf("[%lu]> %-16s :: %3i of %3i... %5.2f seconds\n",                      \
                   static_cast<unsigned long>(this_thread_id()), note, counter,          \
                   total_count, elapsed_seconds.count());                                \
        }
#endif

//======================================================================================//
