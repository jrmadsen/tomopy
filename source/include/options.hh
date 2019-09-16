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

/** \file options.hh
 * \headerfile options.hh "include/options.hh"
 * C++ class for storing thread-specific options when doing rotation-based reconstructions
 * CpuData == rotation-based reconstruction with OpenCV
 * GpuData == rotation-based reconstruction with NPP
 */

#pragma once

#include "backend/device.hh"
#include "common.hh"
#include "constants.hh"
#include "macros.hh"
#include "typedefs.hh"
#include "utils.hh"

#include <array>
#include <atomic>
#include <deque>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

//======================================================================================//
//  This class enables the selection of a device at runtime
//
struct DeviceOption
{
public:
    using string_t       = std::string;
    int      index       = -1;
    string_t key         = "";
    string_t description = "";

    DeviceOption() {}

    DeviceOption(const int& _idx, const string_t& _key, const string_t& _desc)
    : index(_idx)
    , key(_key)
    , description(_desc)
    {}

    static void spacer(std::ostream& os, const char c = '-')
    {
        std::stringstream ss;
        ss.fill(c);
        ss << std::setw(90) << ""
           << "\n";
        os << ss.str();
    }

    friend bool operator==(const DeviceOption& lhs, const DeviceOption& rhs)
    {
        return (lhs.key == rhs.key && lhs.index == rhs.index);
    }

    friend bool operator==(const DeviceOption& itr, const string_t& cmp)
    {
        return (!is_numeric(cmp)) ? (itr.key == tolower(cmp))
                                  : (itr.index == from_string<int>(cmp));
    }

    friend bool operator!=(const DeviceOption& lhs, const DeviceOption& rhs)
    {
        return !(lhs == rhs);
    }

    friend bool operator!=(const DeviceOption& itr, const string_t& cmp)
    {
        return !(itr == cmp);
    }

    static void header(std::ostream& os)
    {
        std::stringstream ss;
        ss << "\n";
        spacer(ss, '=');
        ss << "Available GPU options:\n";
        ss << "\t" << std::left << std::setw(5) << "INDEX"
           << "  \t" << std::left << std::setw(12) << "KEY"
           << "  " << std::left << std::setw(40) << "DESCRIPTION"
           << "\n";
        os << ss.str();
    }

    static void footer(std::ostream& os)
    {
        std::stringstream ss;
        ss << "\nTo select an option for runtime, set 'device' parameter to an "
           << "INDEX or KEY above\n";
        spacer(ss, '=');
        os << ss.str();
    }

    friend std::ostream& operator<<(std::ostream& os, const DeviceOption& opt)
    {
        std::stringstream ss;
        ss << "\t" << std::right << std::setw(5) << opt.index << "  \t" << std::left
           << std::setw(12) << opt.key << "  " << std::left << std::setw(40)
           << opt.description;
        os << ss.str();
        return os;
    }

    // helper function for converting to lower-case
    inline static std::string tolower(std::string val)
    {
        for(auto& itr : val)
            itr = static_cast<char>(::tolower(itr));
        return val;
    }

    // helper function to convert string to another type
    template <typename _Tp>
    static _Tp from_string(const std::string& val)
    {
        std::stringstream ss;
        _Tp               ret;
        ss << val;
        ss >> ret;
        return ret;
    }

    // helper function to determine if numeric represented as string
    inline static bool is_numeric(const std::string& val)
    {
        if(val.length() > 0)
        {
            auto f = val.find_first_of("0123456789");
            if(f == std::string::npos)  // no numbers
                return false;
            auto l = val.find_last_of("0123456789");
            if(val.length() <= 2)  // 1, 2., etc.
                return true;
            else
                return (f != l);  // 1.0, 1e3, 23, etc.
        }
        return false;
    }
};

//======================================================================================//
// this function selects the device to run the reconstruction on
//

inline DeviceOption
GetDevice(const std::string& preferred)
{
    auto pythreads               = env::get("TOMOPY_PYTHON_THREADS", HW_CONCURRENCY);
    using DeviceOptionList       = std::deque<DeviceOption>;
    DeviceOptionList options     = { DeviceOption(0, "cpu", "Run on CPU (OpenCV)") };
    std::string      default_key = "cpu";

    if(device_enabled() && device_count() > 0)
    {
        options.push_back(DeviceOption(1, "gpu", "Run on GPU (CUDA NPP)"));
        default_key = "gpu";
#if defined(TOMOPY_USE_NVTX)
        // initialize nvtx data
        init_nvtx();
#endif
        // print device info
        device_query();
    }
    else
    {
        auto_lock l(type_mutex<decltype(std::cout)>());
        std::cerr << "\n##### No CUDA device(s) available #####\n" << std::endl;
    }

    // find the default entry
    auto default_itr =
        std::find_if(options.begin(), options.end(),
                     [&](const DeviceOption& itr) { return (itr == default_key); });

    //------------------------------------------------------------------------//
    // print the options the first time it is encountered
    auto print_options = [&]() {
        static std::atomic_uint _once;
        auto                    _count = _once++;
        if(_count % pythreads > 0)
        {
            if(_count + 1 == pythreads)
            {
                _once.store(0);
            }
            return;
        }

        std::stringstream ss;
        DeviceOption::header(ss);
        for(const auto& itr : options)
        {
            ss << itr;
            if(itr == *default_itr)
                ss << "\t(default)";
            ss << "\n";
        }
        DeviceOption::footer(ss);

        auto_lock l(type_mutex<decltype(std::cout)>());
        std::cout << "\n" << ss.str() << std::endl;
    };
    //------------------------------------------------------------------------//
    // print the option selection first time it is encountered
    auto print_selection = [&](DeviceOption& selected_opt) {
        static std::atomic_uint _once;
        auto                    _count = _once++;
        if(_count % pythreads > 0)
        {
            if(_count + 1 == pythreads)
            {
                _once.store(0);
            }
            return;
        }

        std::stringstream ss;
        DeviceOption::spacer(ss, '-');
        ss << "Selected device: " << selected_opt << "\n";
        DeviceOption::spacer(ss, '-');

        auto_lock l(type_mutex<decltype(std::cout)>());
        std::cout << ss.str() << std::endl;
    };
    //------------------------------------------------------------------------//

    // print the GPU execution type options
    print_options();

    default_key = default_itr->key;
    auto key    = preferred;

    auto selection = std::find_if(options.begin(), options.end(),
                                  [&](const DeviceOption& itr) { return (itr == key); });

    if(selection == options.end())
        selection =
            std::find_if(options.begin(), options.end(),
                         [&](const DeviceOption& itr) { return itr == default_key; });

    print_selection(*selection);

    return *selection;
}

//======================================================================================//

struct RuntimeOptions
{
    num_threads_t      pool_size     = HW_CONCURRENCY;
    int                interpolation = -1;
    DeviceOption       device;
    std::array<int, 3> block_size = { { 512, 1, 1 } };
    std::array<int, 3> grid_size  = { { 0, 0, 0 } };

    RuntimeOptions(int _pool_size, const char* _interp, const char* _device,
                   int* _grid_size, int* _block_size)
    : pool_size(static_cast<num_threads_t>(_pool_size))
    , device(GetDevice(_device))
    {
        memcpy(grid_size.data(), _grid_size, 1 * sizeof(int));
        memcpy(block_size.data(), _block_size, 1 * sizeof(int));
        grid_size[1] = grid_size[2] = 0;
        block_size[1] = block_size[2] = 1;

        interpolation = interpolation::mode(_interp);
        if(device.key == "gpu" && !device_enabled())
        {
            throw std::runtime_error(
                "Error! Selected device 'gpu' is not available without CUDA support!");
        }
        else if(device.key == "cpu" && device_count() == 0)
        {
            throw std::runtime_error("Error! The device count returned zero. It appears "
                                     "tomopy was not compiled with "
                                     "OpenCV or CUDA support.");
        }
    }

    ~RuntimeOptions() {}

    // disable copying and copy assignment
    RuntimeOptions(const RuntimeOptions&) = delete;
    RuntimeOptions& operator=(const RuntimeOptions&) = delete;

    // create the thread pool -- don't have this in the constructor
    // because you don't want to arbitrarily create thread-pools
    void init()
    {
        if(pool_size == 0)
        {
            pool_size = std::min<num_threads_t>(HW_CONCURRENCY, 12);
        }
    }

    // invoke the generic printer defined in common.hh
    template <typename... _Descriptions, typename... _Objects>
    void print(std::tuple<_Descriptions...>&& _descripts, std::tuple<_Objects...>&& _objs,
               std::ostream& os, intmax_t _prefix_width, intmax_t _obj_width,
               std::ios_base::fmtflags format_flags, bool endline) const
    {
        // tuple of descriptions
        using DescriptType = std::tuple<_Descriptions...>;
        // tuple of objects to print
        using ObjectType = std::tuple<_Objects...>;
        // the operator that does the printing (see end of
        using UnrollType = std::tuple<GenericPrinter<_Objects>...>;

        apply::unroll<UnrollType>(std::forward<DescriptType>(_descripts),
                                  std::forward<ObjectType>(_objs), std::ref(os),
                                  _prefix_width, _obj_width, format_flags, endline);
    }

    // overload the output operator for the class
    friend std::ostream& operator<<(std::ostream& os, const RuntimeOptions& opts)
    {
        std::stringstream ss;
        opts.print(std::make_tuple("Thread-pool size", "Interpolation mode", "Device",
                                   "Grid size", "Block size"),
                   std::make_tuple(opts.pool_size, opts.interpolation, opts.device,
                                   opts.block_size, opts.grid_size),
                   ss, 30, 20, std::ios::boolalpha, true);
        os << ss.str();
        return os;
    }
};

//======================================================================================//
