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
//   TOMOPY implementation

//======================================================================================//

#include "common.hh"
#include "gpu.hh"

BEGIN_EXTERN_C
#include "gpu.h"
#include "utils_cuda.h"
END_EXTERN_C

#if defined(TOMOPY_USE_NVTX)
extern nvtxEventAttributes_t nvtx_total;
extern nvtxEventAttributes_t nvtx_iteration;
extern nvtxEventAttributes_t nvtx_slice;
extern nvtxEventAttributes_t nvtx_projection;
extern nvtxEventAttributes_t nvtx_update;
extern nvtxEventAttributes_t nvtx_rotate;
#endif

//======================================================================================//
// interpolation types
#define INTER_NN NPPI_INTER_NN
#define INTER_LINEAR NPPI_INTER_LINEAR
#define INTER_CUBIC NPPI_INTER_CUBIC

//======================================================================================//

//  gridDim:    This variable contains the dimensions of the grid.
//  blockIdx:   This variable contains the block index within the grid.
//  blockDim:   This variable and contains the dimensions of the block.
//  threadIdx:  This variable contains the thread index within the block.

//======================================================================================//
//
//  rotate
//
//======================================================================================//

template <typename _Tp>
void
print_array(const _Tp* data, int nx, int ny, const std::string& desc)
{
    std::stringstream ss;
    ss << desc << "\n\n";
    ss << std::fixed;
    ss.precision(3);
    for(int j = 0; j < ny; ++j)
    {
        ss << "  ";
        for(int i = 0; i < nx; ++i)
        {
            ss << std::setw(8) << data[j * nx + i] << " ";
        }
        ss << std::endl;
    }
    std::cout << ss.str() << std::endl;
}

//======================================================================================//
//
//  rotate
//
//======================================================================================//

void
cuda_rotate_kernel(float* dst, const float* src, const float theta_rad,
                   const float theta_deg, const int nx, const int ny,
                   int eInterp = INTER_CUBIC, cudaStream_t stream = 0)
{
    cudaStreamSynchronize(stream);
    nppSetStream(stream);
    NVTX_RANGE_PUSH(&nvtx_rotate);

    auto getRotationMatrix2D = [&](double m[2][3], double scale) {
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

    int    step = nx * sizeof(float);
    double rot[2][3];
    getRotationMatrix2D(rot, 1.0);

#if defined(DEBUG)
    printf("theta = %5.1f\n", theta_deg);
    print_array((double*) rot, 3, 2, "rot");
#endif

// #define USE_NPPI_ROTATE
#if defined(USE_NPPI_ROTATE)
    NppStatus ret = nppiRotate_32f_C1R(src, siz, step, roi, dst, step, roi, theta_deg,
                                       rot[0][2], rot[1][2], eInterp);
#else
    NppStatus ret =
        nppiWarpAffine_32f_C1R(src, siz, step, roi, dst, step, roi, rot, eInterp);
#endif
    if(ret != NPP_SUCCESS)
        printf("%s returned non-zero NPP status: %i\n", __FUNCTION__, ret);

    cudaStreamSynchronize(stream);
    NVTX_RANGE_POP(&nvtx_rotate);
    CUDA_CHECK_LAST_ERROR();
}

//======================================================================================//

inline int
GetInterpolationMode()
{
    static int eInterp = GetEnv<int>("TOMOPY_INTER", INTER_CUBIC);
    return eInterp;
}

//======================================================================================//

float*
cuda_rotate(const float* src, const float theta_rad, const float theta_deg, const int nx,
            const int ny, cudaStream_t stream)
{
    float* _dst = gpu_malloc<float>(nx * ny);
    cuda_rotate_kernel(_dst, src, theta_rad, theta_deg, nx, ny, GetInterpolationMode(),
                       stream);
    return _dst;
}

//======================================================================================//

void
cuda_rotate_ip(float* dst, const float* src, const float theta_rad, const float theta_deg,
               const int nx, const int ny, cudaStream_t stream)
{
    cuda_rotate_kernel(dst, src, theta_rad, theta_deg, nx, ny, GetInterpolationMode(),
                       stream);
}

//======================================================================================//
