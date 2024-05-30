#pragma once
#include "rtweekend.h"
#include "color.h"
#include "hittable.h"
#include "hittable_list.h"
#include "material.h"

class camera
{
public:
    double aspect_ratio = 1.0;
    int image_width = 1024;
    int max_depth = 10;
    double vfov = 90;
    point3 lookfrom = point3(0, 0, -1);
    point3 lookat = point3(0, 0, 0);
    vec3 vup = vec3(0, 1, 0);

    //初始化相机视角
    __host__ __device__ void initialize()
    {
        vfov = 90;
        lookfrom = point3(-2, 2, 1);
        lookat = point3(0, 0, -1);
        vup = vec3(0, 1, 0);

        image_height = static_cast<int>(image_width / aspect_ratio);
        image_height = image_height < 1 ? 1 : image_height;
        camera_center = lookfrom;

        auto focal_length = (lookfrom - lookat).length();
        auto theta = degrees_to_radians(vfov);
        auto h = tan(theta / 2);
        auto view_height = 2.0 * h * focal_length;
        auto view_width = view_height * static_cast<double>(image_width) / image_height;

        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        //计算横纵向边缘向量
        vec3 viewport_u = view_width * u;
        vec3 viewport_v = -view_height * v;

        //计算像素间距
        delta_pixel_u = viewport_u / image_width;
        delta_pixel_v = viewport_v / image_height;

        //计算起始参照点
        point3 viewport_upper_left = camera_center - focal_length * w - 0.5 * (viewport_u + viewport_v);
        viewport_pixel00 = viewport_upper_left + 0.5 * (delta_pixel_u + delta_pixel_v);
    }

    //根据像素位置获取随机的光线
    __device__ Ray get_ray(int i, int j, curandState* globalState, int ind) const {
        // Get a randomly sampled camera ray for the pixel at location i,j.
        auto pixel_center = viewport_pixel00 + (i * delta_pixel_u) + (j * delta_pixel_v);
        auto pixel_sample = pixel_center + pixel_sample_square(globalState, ind);
        //auto pixel_sample = pixel_center;
        auto ray_origin = camera_center;
        auto ray_direction = pixel_sample - ray_origin;
        Ray r(ray_origin, ray_direction);
        return r;
    }

    __device__ __forceinline__
        color ray_color(Ray r, hittable_list* dev_spheres, curandState* state, int ind) {
        hit_record rec;
        vec3 mul(1, 1, 1);
        int depth = max_depth;
        while ((depth > 0) && dev_spheres->hit(r, interval(0.001, INF), rec)) {
            depth--;
            Ray scattered;
            color attenuation;
            if (rec.mat->scatter(r, rec, attenuation, scattered, state, ind)) {
                r = scattered;
                mul = mul * attenuation;
            }
            else {
                return color(0, 0, 0);
            }
        }
        if (depth == 0) return vec3(0, 0, 0);
        vec3 unit_direction = unit_vector(r.direction());
        double t = 0.5 * (unit_direction.y() + 1.0);
        color res = mul * ((1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0));
        return res;
    }

private:
    int image_height;
    point3 camera_center; // Camera center
    point3 viewport_pixel00; // Location of pixel 0, 0
    vec3 delta_pixel_u; // Offset to pixel to the right
    vec3 delta_pixel_v; // Offset to pixel below
    vec3 u, v, w;

    __device__ vec3 pixel_sample_square(curandState* globalState, int ind) const {
        // Returns a random point in the square surrounding a pixel at the origin.
        auto px = -0.5 + random_double(globalState, ind);
        auto py = -0.5 + random_double(globalState, ind);
        return (px * delta_pixel_u) + (py * delta_pixel_v);
    }

};

//class camera {
//public:
//    double aspect_ratio = 1.0;
//    int image_width = 100;
//    int samples_per_pixel = 10;
//    int max_depth = 10;
//    double vfov = 90;
//    point3 lookfrom = point3(0, 0, -1);
//    point3 lookat = point3(0, 0, 0);
//    vec3 vup = vec3(0, 1, 0);
//
//    void render(const hittable& world)
//    {
//        initialize();
//        auto* odata = (unsigned char*)malloc(image_width * image_height * 3);
//        long long idx = 0;
//        for (int j = 0; j < image_height; ++j) {
//            for (int i = 0; i < image_width; ++i)
//            {
//                color pixel_color(0, 0, 0);
//                for (int sample = 0; sample < samples_per_pixel; ++sample) {
//                    Ray r = get_ray(i, j);
//                    pixel_color += ray_color(r, max_depth,world);
//                }
//                pixel_color /= samples_per_pixel;
//                idx = 3 * j * image_width + 3 * i;
//                write_color(odata, idx, pixel_color);                
//            }
//        }
//
//        //输出为png图片
//        /*std::string output = "picture2.png";
//        stbi_write_png(output.c_str(), image_width, image_height, 3, odata, 0);
//        stbi_image_free(odata);
//        std::clog << "\rDone. \n";*/
//    }
//
//private:
//    int image_height;
//    point3 camera_center; // Camera center
//    point3 viewport_pixel00; // Location of pixel 0, 0
//    vec3 delta_pixel_u; // Offset to pixel to the right
//    vec3 delta_pixel_v; // Offset to pixel below
//    vec3 u, v, w;
//
//	void initialize() 
//	{
//        image_height = static_cast<int>(image_width / aspect_ratio);
//        image_height = image_height < 1 ? 1 : image_height;
//        camera_center = lookfrom;
//
//        auto focal_length = (lookfrom-lookat).length();
//        auto theta = degrees_to_radians(vfov);
//        auto h = tan(theta / 2);
//        auto view_height = 2.0 * h * focal_length;
//        auto view_width = view_height * static_cast<double>(image_width) / image_height;
//        
//        w = unit_vector(lookfrom - lookat);
//        u = unit_vector(cross(vup,w));
//        v = cross(w,u);
//
//        //计算横纵向边缘向量
//        vec3 viewport_u = view_width * u;
//        vec3 viewport_v = -view_height * v;
//
//        //计算像素间距
//        delta_pixel_u = viewport_u / image_width;
//        delta_pixel_v = viewport_v / image_height;
//
//        //计算起始参照点
//        point3 viewport_upper_left = camera_center  - focal_length * w - 0.5 * (viewport_u + viewport_v);
//        viewport_pixel00 = viewport_upper_left + 0.5 * (delta_pixel_u + delta_pixel_v);
//	}
//
//    color ray_color(const Ray& r, int depth,const hittable& world) {
//
//        if (depth <= 0)
//            return color(0, 0, 0);
//        hit_record rec;
//        if (world.hit(r, interval(0.001, infinity), rec))
//        {
//            //vec3 reflect_direction = random_on_hemisphere(rec.normal);
//            //vec3 reflect_direction = rec.normal + random_unit_vector();
//            Ray scattered; 
//            color attenutation;
//            if(rec.mat->scatter(r, rec, attenutation, scattered));
//                return attenutation * ray_color(scattered,depth-1,world);
//        }
//            
//
//        vec3 unit_dir = unit_vector(r.direction());
//        double a = 0.5 * (unit_dir.y() + 1.0);
//        return (1.0 - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0);
//    }
//
//    Ray get_ray(int i, int j) const {
//        // Get a randomly sampled camera ray for the pixel at location i,j.
//        auto pixel_center = viewport_pixel00 + (i * delta_pixel_u) + (j * delta_pixel_v);
//        auto pixel_sample = pixel_center + pixel_sample_square();
//        auto ray_origin = camera_center;
//        auto ray_direction = pixel_sample - ray_origin;
//        return Ray(ray_origin, ray_direction);
//    }
//    vec3 pixel_sample_square() const {
//        // Returns a random point in the square surrounding a pixel at the origin.
//        auto px = -0.5 + random_double();
//        auto py = -0.5 + random_double();
//        return (px * delta_pixel_u) + (py * delta_pixel_v);
//    }
//
//};