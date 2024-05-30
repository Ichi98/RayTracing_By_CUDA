#pragma once
#include "vec3.h"
#include "hittable.h"

class Sphere :public hittable {
public:
    __device__ Sphere(const point3 c, const double& r, material* mat) :_center(c), _radius(r), _material(mat) {}

    __device__ bool hit(const Ray& r, interval ray_t, hit_record& rec) const override
    {
        vec3 oc = _center - r.origin();
        auto a = r.direction().length_squared();
        auto h = dot(r.direction(), oc);
        auto c = oc.length_squared() - _radius * _radius;
        auto discriminant = h * h - a * c;
        if (discriminant < 0)
            return false;

        auto sqrtd = sqrt(discriminant);

        // Find the nearest root that lies in the acceptable range.
        auto root = (h - sqrtd) / a;
        if (!ray_t.surrounds(root)) {
            root = (h + sqrtd) / a;
            if (!ray_t.surrounds(root))
                return false;

        }

        rec.t = root;
        rec.p = r.at(rec.t);
        vec3 outward_normal = (rec.p - _center) / _radius;
        rec.set_face_normal(r, outward_normal);
        rec.mat = _material;

        return true;
    }

    __device__ point3 center() const
    {
        return _center;
    }

    __device__ double radius()const
    {
        return _radius;
    }

private:
    double _radius;
    point3 _center;
    material* _material;
};