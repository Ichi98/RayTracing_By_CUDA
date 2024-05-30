#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H
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

#include "hittable.h"

class hittable_list {
public:
    Vector<hittable*> objects;
    __device__ hittable_list() {}
    __device__ hittable_list(Vector<hittable*> objs) { objects = objs; }
    __device__ ~hittable_list() { clear(); }

    __device__ void clear() {
        for (int i = 0; i < objects.size(); ++i) {
            delete objects[i];
        }
    }

    __device__ void add(hittable* object) {
        objects.push_back(object);
    }

    __device__ bool hit(const Ray& r, interval ray_t, hit_record& rec) const {
        hit_record temp_rec;
        bool hit_anything = false;
        auto closest_so_far = ray_t.maxium;
  
        for (int i = 0; i < objects.size(); ++i) {
            hittable* object = objects[i];
            if (object->hit(r, interval(ray_t.minimum, closest_so_far), temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
        }

        return hit_anything;
    }
};


#endif
