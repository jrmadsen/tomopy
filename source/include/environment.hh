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

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <tuple>

//--------------------------------------------------------------------------------------//
// use this function to get rid of "unused parameter" warnings
//
template <typename... _Args>
void
consume_parameters(_Args...)
{}

//--------------------------------------------------------------------------------------//
// a non-string environment option with a string identifier
template <typename _Tp>
using EnvChoice = std::tuple<_Tp, std::string, std::string>;

//--------------------------------------------------------------------------------------//
// list of environment choices with non-string and string identifiers
template <typename _Tp>
using EnvChoiceList = std::set<EnvChoice<_Tp>>;

//--------------------------------------------------------------------------------------//

class EnvSettings
{
public:
    typedef std::mutex                        mutex_t;
    typedef std::string                       string_t;
    typedef std::multimap<string_t, string_t> env_map_t;
    typedef std::pair<string_t, string_t>     env_pair_t;

public:
    static EnvSettings* GetInstance()
    {
        static EnvSettings* _instance = new EnvSettings();
        return _instance;
    }

public:
    template <typename _Tp>
    void insert(const std::string& env_id, _Tp val)
    {
        std::stringstream ss;
        ss << std::boolalpha << val;
        m_mutex.lock();
        if(m_env.find(env_id) != m_env.end())
        {
            for(const auto& itr : m_env)
                if(itr.first == env_id && itr.second == ss.str())
                {
                    m_mutex.unlock();
                    return;
                }
        }
        m_env.insert(env_pair_t(env_id, ss.str()));
        m_mutex.unlock();
    }

    template <typename _Tp>
    void insert(const std::string& env_id, EnvChoice<_Tp> choice)
    {
        _Tp&         val      = std::get<0>(choice);
        std::string& str_val  = std::get<1>(choice);
        std::string& descript = std::get<2>(choice);

        std::stringstream ss, ss_long;
        ss << std::boolalpha << val;
        ss_long << std::boolalpha << std::setw(8) << std::left << val << " # (\""
                << str_val << "\") " << descript;
        m_mutex.lock();
        if(m_env.find(env_id) != m_env.end())
        {
            for(const auto& itr : m_env)
                if(itr.first == env_id && itr.second == ss.str())
                {
                    m_mutex.unlock();
                    return;
                }
        }
        m_env.insert(env_pair_t(env_id, ss_long.str()));
        m_mutex.unlock();
    }

    const env_map_t& get() const { return m_env; }
    mutex_t&         mutex() const { return m_mutex; }

    friend std::ostream& operator<<(std::ostream& os, const EnvSettings& env)
    {
        std::stringstream filler;
        filler.fill('#');
        filler << std::setw(90) << "";
        std::stringstream ss;
        ss << filler.str() << "\n# Environment settings:\n";
        env.mutex().lock();
        for(const auto& itr : env.get())
        {
            ss << "# " << std::setw(35) << std::right << itr.first << "\t = \t"
               << std::left << itr.second << "\n";
        }
        env.mutex().unlock();
        ss << filler.str();
        os << ss.str() << std::endl;
        return os;
    }

private:
    env_map_t       m_env;
    mutable mutex_t m_mutex;
};

//--------------------------------------------------------------------------------------//
//  use this function to get an environment variable setting +
//  a default if not defined, e.g.
//      int num_threads =
//          get_env<int>("FORCENUMBEROFTHREADS",
//                          std::thread::hardware_concurrency());
//
template <typename _Tp>
_Tp
get_env(const std::string& env_id, _Tp _default = _Tp())
{
    char* env_var = std::getenv(env_id.c_str());
    if(env_var)
    {
        std::string        str_var = std::string(env_var);
        std::istringstream iss(str_var);
        _Tp                var = _Tp();
        iss >> var;
        // record value defined by environment
        EnvSettings::GetInstance()->insert<_Tp>(env_id, var);
        return var;
    }
    // record default value
    EnvSettings::GetInstance()->insert<_Tp>(env_id, _default);

    // return default if not specified in environment
    return _default;
}

//--------------------------------------------------------------------------------------//
//  overload for boolean
//
template <>
inline bool
get_env(const std::string& env_id, bool _default)
{
    char* env_var = std::getenv(env_id.c_str());
    if(env_var)
    {
        std::string var = std::string(env_var);
        bool        val = true;
        if(var.find_first_not_of("0123456789") == std::string::npos)
            val = (bool) atoi(var.c_str());
        else
        {
            for(auto& itr : var)
                itr = tolower(itr);
            if(var == "off" || var == "false")
                val = false;
        }
        // record value defined by environment
        EnvSettings::GetInstance()->insert<bool>(env_id, val);
        return val;
    }
    // record default value
    EnvSettings::GetInstance()->insert<bool>(env_id, false);

    // return default if not specified in environment
    return _default;
}

//--------------------------------------------------------------------------------------//
//  overload for get_env + message when set
//
template <typename _Tp>
_Tp
get_env(const std::string& env_id, _Tp _default, const std::string& msg)
{
    char* env_var = std::getenv(env_id.c_str());
    if(env_var)
    {
        std::string        str_var = std::string(env_var);
        std::istringstream iss(str_var);
        _Tp                var = _Tp();
        iss >> var;
        std::cout << "Environment variable \"" << env_id << "\" enabled with "
                  << "value == " << var << ". " << msg << std::endl;
        // record value defined by environment
        EnvSettings::GetInstance()->insert<_Tp>(env_id, var);
        return var;
    }
    // record default value
    EnvSettings::GetInstance()->insert<_Tp>(env_id, _default);

    // return default if not specified in environment
    return _default;
}

//--------------------------------------------------------------------------------------//
//  use this function to get an environment variable setting from set of choices
//
//      EnvChoiceList<int> choices =
//              { EnvChoice<int>(NN,     "NN",     "nearest neighbor interpolation"),
//                EnvChoice<int>(LINEAR, "LINEAR", "bilinear interpolation"),
//                EnvChoice<int>(CUBIC,  "CUBIC",  "bicubic interpolation") };
//
//      int eInterp = get_env<int>("INTERPOLATION", choices, CUBIC);
//
template <typename _Tp>
_Tp
get_env(const std::string& env_id, const EnvChoiceList<_Tp>& _choices, _Tp _default)
{
    auto asupper = [](std::string var) {
        for(auto& itr : var)
            itr = toupper(itr);
        return var;
    };

    char* env_var = std::getenv(env_id.c_str());
    if(env_var)
    {
        std::string str_var = std::string(env_var);
        std::string upp_var = asupper(str_var);
        _Tp         var     = _Tp();
        // check to see if string matches a choice
        for(const auto& itr : _choices)
        {
            if(asupper(std::get<1>(itr)) == upp_var)
            {
                // record value defined by environment
                EnvSettings::GetInstance()->insert(env_id, itr);
                return std::get<0>(itr);
            }
        }
        std::istringstream iss(str_var);
        iss >> var;
        // check to see if string matches a choice
        for(const auto& itr : _choices)
        {
            if(var == std::get<0>(itr))
            {
                // record value defined by environment
                EnvSettings::GetInstance()->insert(env_id, itr);
                return var;
            }
        }
        // the value set in env did not match any choices
        std::stringstream ss;
        ss << "\n### Environment setting error @ " << __FUNCTION__ << " (line "
           << __LINE__ << ")! Invalid selection for \"" << env_id
           << "\". Valid choices are:\n";
        for(const auto& itr : _choices)
            ss << "\t\"" << std::get<0>(itr) << "\" or \"" << std::get<1>(itr) << "\" ("
               << std::get<2>(itr) << ")\n";
        std::cerr << ss.str() << std::endl;
        abort();
    }

    std::string _name = "???";
    std::string _desc = "description not provided";
    for(const auto& itr : _choices)
        if(std::get<0>(itr) == _default)
        {
            _name = std::get<1>(itr);
            _desc = std::get<2>(itr);
            break;
        }

    // record default value
    EnvSettings::GetInstance()->insert(env_id, EnvChoice<_Tp>(_default, _name, _desc));

    // return default if not specified in environment
    return _default;
}

//--------------------------------------------------------------------------------------//

template <typename _Tp>
_Tp
GetChoice(const EnvChoiceList<_Tp>& _choices, const std::string str_var)
{
    auto asupper = [](std::string var) {
        for(auto& itr : var)
            itr = toupper(itr);
        return var;
    };

    std::string upp_var = asupper(str_var);
    _Tp         var     = _Tp();
    // check to see if string matches a choice
    for(const auto& itr : _choices)
    {
        if(asupper(std::get<1>(itr)) == upp_var)
        {
            // record value defined by environment
            return std::get<0>(itr);
        }
    }
    std::istringstream iss(str_var);
    iss >> var;
    // check to see if string matches a choice
    for(const auto& itr : _choices)
    {
        if(var == std::get<0>(itr))
        {
            // record value defined by environment
            return var;
        }
    }
    // the value set in env did not match any choices
    std::stringstream ss;
    ss << "\n### Environment setting error @ " << __FUNCTION__ << " (line " << __LINE__
       << ")! Invalid selection \"" << str_var << "\". Valid choices are:\n";
    for(const auto& itr : _choices)
        ss << "\t\"" << std::get<0>(itr) << "\" or \"" << std::get<1>(itr) << "\" ("
           << std::get<2>(itr) << ")\n";
    std::cerr << ss.str() << std::endl;
    abort();
}

//--------------------------------------------------------------------------------------//

inline void
PrintEnv(std::ostream& os = std::cout)
{
    os << (*EnvSettings::GetInstance());
}

//--------------------------------------------------------------------------------------//
