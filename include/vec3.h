#ifndef VEC3_H
#define VEC3_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//==============================================================================================
// Originally written in 2016 by Peter Shirley <ptrshrl@gmail.com>
//
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is
// distributed without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication
// along with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==============================================================================================

class vec3 {
public:
    double e[3];

    __device__ __host__ vec3() : e{ 0.,0.,0. } {}
    __device__ __host__ vec3(double e0, double e1, double e2) : e{ e0, e1, e2 } {}

    __device__ __host__ double x() const { return e[0]; }
    __device__ __host__ double y() const { return e[1]; }
    __device__ __host__ double z() const { return e[2]; }

    __device__ __host__ vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    __device__ __host__ double operator[](int i) const { return e[i]; }
    __device__ __host__ double& operator[](int i) { return e[i]; }

    __device__ __host__ vec3& operator+=(const vec3& v) {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

    __device__ __host__ vec3& operator*=(double t) {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    __device__ __host__ vec3& operator/=(double t) {
        return *this *= 1 / t;
    }

    __device__ __host__ double length() const {
        return sqrt(length_squared());
    }

    __device__ __host__ double length_squared() const {
        return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
    }

    __device__ bool near_zero() const {
        // Return true if the vector is close to zero in all dimensions.
        auto s = 1e-8;
        return (fabs(e[0]) < s) && (fabs(e[1]) < s) && (fabs(e[2]) < s);
    }

    __device__ static vec3 random(curandState* globalState, int ind) {
        return vec3(random_double(globalState, ind), random_double(globalState, ind), random_double(globalState, ind));
    }

    __device__ static vec3 random(double min, double max, curandState* globalState, int ind) {
        return vec3(random_double(min, max, globalState, ind), random_double(min, max, globalState, ind), random_double(min, max, globalState, ind));
    }
};

// point3 is just an alias for vec3, but useful for geometric clarity in the code.
using point3 = vec3;


// Vector Utility Functions

__device__ __host__ __forceinline__
vec3 operator+(const vec3& u, const vec3& v) {
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

__device__ __host__ __forceinline__
vec3 operator-(const vec3& u, const vec3& v) {
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

__device__ __host__ __forceinline__
vec3 operator*(const vec3& u, const vec3& v) {
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

__device__ __host__ __forceinline__
vec3 operator*(double t, const vec3& v) {
    return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__device__ __host__ __forceinline__
vec3 operator*(const vec3& v, double t) {
    return t * v;
}

__device__ __host__ __forceinline__
vec3 operator/(const vec3& v, double t) {
    return (1 / t) * v;
}

__device__ __host__ __forceinline__
double dot(const vec3& u, const vec3& v) {
    return u.e[0] * v.e[0]
        + u.e[1] * v.e[1]
        + u.e[2] * v.e[2];
}

__device__ __host__ __forceinline__
vec3 cross(const vec3& u, const vec3& v) {
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
        u.e[2] * v.e[0] - u.e[0] * v.e[2],
        u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

__device__ __host__ __forceinline__
vec3 unit_vector(const vec3& v) {
    return v / v.length();
}

//__device__ __forceinline__ vec3 random_in_unit_disk() {
//    while (true) {
//        auto p = vec3(random_double(-1,1), random_double(-1,1), 0);
//        if (p.length_squared() < 1)
//            return p;
//    }
//}
//
__device__ __forceinline__ vec3 random_in_unit_sphere(curandState* globalState, int ind) {
    while (true) {
        auto p = vec3::random(-1, 1, globalState, ind);
        if (p.length_squared() < 1)
            return p;
    }
}

__device__ __forceinline__ vec3 random_unit_vector(curandState* globalState, int ind) {
    return unit_vector(random_in_unit_sphere(globalState, ind));
}

//__device__ __forceinline__ vec3 random_on_hemisphere(const vec3& normal) {
//    vec3 on_unit_sphere = random_unit_vector();
//    if (dot(on_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
//        return on_unit_sphere;
//    else
//        return -on_unit_sphere;
//}

__device__ __forceinline__ vec3 reflect(const vec3& v, const vec3& n) {
    return v - 2 * dot(v, n) * n;
}

__device__ __forceinline__ vec3 refract(const vec3& uv, const vec3& n, double etai_over_etat) {
    auto cos_theta = fmin(dot(-uv, n), 1.0);
    vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    vec3 r_out_parallel = -sqrt(fabs(1.0 - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}


#endif
