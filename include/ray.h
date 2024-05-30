#pragma once
#include "vec3.h"
class Ray
{
public:
	__device__ __host__ Ray() :_origin(point3(0, 0, 0)), _direction(vec3(0, 0, 0)) {};
	__device__ __host__ Ray(const point3& orig, const vec3& dir) :_origin(orig), _direction(dir) {}

	__device__ __host__ point3 origin() const
	{
		return _origin;
	}

	__device__ __host__ vec3 direction() const
	{
		return _direction;
	}

	__device__ __host__ void setOrigin(const point3& orig)
	{
		_origin = orig;
	}

	__device__ __host__ void setDirection(const vec3& dir)
	{
		_direction = dir;
	}

	__device__ __host__ point3 at(double t) const
	{
		return _origin + t * _direction;
	}

private:
	point3 _origin;
	vec3 _direction;
 };