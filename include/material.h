#pragma once
#include"rtweekend.h"
#include "hittable.h"

class material
{
public:
	__device__ virtual ~material() = default;

	__device__ virtual bool scatter(const Ray& r_in, const hit_record& rec, color& attenuation, Ray& scattered, curandState* globalState, int ind) const = 0;
};

class lambertian :public material
{
public:
	__device__ lambertian(const color a) :albedo(a) {};

	__device__ bool scatter(const Ray& r_in, const hit_record& rec, color& attenuation, Ray& scattered, curandState* globalState, int ind) const override
	{
		vec3 direction = rec.normal + random_unit_vector(globalState, ind);
		if (direction.near_zero())
			direction = rec.normal;
		scattered = Ray(rec.p, direction);
		attenuation = albedo;
		return true;
	}

private:
	color albedo;
};

class metal :public material
{
public:
	__device__ metal(const color a, const double f) :albedo(a), fuzz(f < 1 ? f : 1) {};

	__device__ bool scatter(const Ray& r_in, const hit_record& rec, color& attenuation, Ray& scattered, curandState* globalState, int ind) const override
	{
		vec3 direction = reflect(unit_vector(r_in.direction()), rec.normal);
		scattered = Ray(rec.p, direction + fuzz * random_unit_vector(globalState, ind));
		attenuation = albedo;
		return dot(rec.normal, scattered.direction()) > 0;
	}

private:
	color albedo;
	double fuzz;
};

class dielectric : public material {
public:
	__device__ dielectric(double index_of_refraction) : ir(index_of_refraction) {}
	__device__ bool scatter(const Ray& r_in, const hit_record& rec, color& attenuation, Ray& scattered, curandState* globalState, int ind)
		const override {
		attenuation = color(1.0, 1.0, 1.0);
		double refraction_ratio = rec.front_face ? (1.0 / ir) : ir;
		vec3 unit_direction = unit_vector(r_in.direction());
		double cos_theta = fmin(dot(-unit_direction, rec.normal), 1.0);
		double sin_theta = sqrt(1 - cos_theta * cos_theta);
		bool cannot_refract = refraction_ratio * sin_theta > 1.0;
		vec3 direction;
		if (cannot_refract || reflectance(cos_theta, refraction_ratio) > random_double(globalState, ind))//reflect			
			direction = reflect(unit_direction, rec.normal);
		else //refract
			direction = refract(unit_direction, rec.normal, refraction_ratio);

		scattered = Ray(rec.p, direction);
		return true;
	}
private:
	double ir; // Index of Refraction

	__device__ double reflectance(double cosine, double ref_idx)const {
		// Use Schlick's approximation for reflectance.
		auto r0 = (1 - ref_idx) / (1 + ref_idx);
		r0 = r0 * r0;
		return r0 + (1 - r0) * pow((1 - cosine), 5);
	}
};