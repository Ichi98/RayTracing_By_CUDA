#ifndef RTWEEKEND_H
#define RTWEEKEND_H
//==============================================================================================
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is
// distributed without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication
// along with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==============================================================================================

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"

// C++ Std Usings

using std::fabs;
using std::sqrt;


__device__ __host__ __forceinline__ double degrees_to_radians(double degrees) {
    return degrees * 3.1415926535897932385 / 180.0;
}

__device__ __forceinline__ double random_double(curandState* globalState, int ind) {
    curandState localState = globalState[ind];
    float RANDOM = curand_uniform(&localState);// uniform distribution
    globalState[ind] = localState;
    return RANDOM;
}

__device__ __forceinline__ double random_double(double min, double max, curandState* globalState, int ind) {

    return min + (max - min) * random_double(globalState, ind);
}

// Common Headers

#include "color.h"
#include "interval.h"
#include "ray.h"
#include "vec3.h"
#include "Vector.h"
#endif
