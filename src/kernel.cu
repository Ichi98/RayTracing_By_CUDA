#include<cmath>
#include <cuda.h>
#include <book.h>
#include <cpu_bitmap.h>
#include <ctime>
#include <random>

#include "rtweekend.h"
#include "sphere.h"
#include "hittable_list.h"
#include "material.h"
#include "camera.h"

#include <stdio.h>

#define INF     2e10f

int W = 1024;
int H = 1024;


__device__ hittable_list* world = nullptr;
__device__ material* mats[4];


// globals needed by the update routine
struct DataBlock {
    unsigned char* dev_bitmap;
};

//设置GPU的随机数
__global__ void setup_rand(curandState* state, unsigned long seed)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = iy * blockDim.x * gridDim.x + ix;
    curand_init(seed, idx, 0, &state[idx]);// initialize the state
}

//主渲染函数
__global__ void render(unsigned char* buffer, camera* cam, curandState* state)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int idx = x + y * blockDim.x * gridDim.x;

    //设置多重光线采样
    int samples_per_pixel = 200;
    color pixel_color(0.0, 0.0, 0.0);
    for (int ss = 0; ss < samples_per_pixel; ++ss)
    {
        Ray r = cam->get_ray(x, y, state, idx);
        pixel_color += cam->ray_color(r, world, state, idx);
    }
    pixel_color /= samples_per_pixel;

    //首先需要根据x轴镜像翻转
    y = gridDim.y * blockDim.y - y - 1;
    idx = x + y * blockDim.x * gridDim.x;
    //而后再进行颜色写入
    buffer[4 * idx] = (int)(pixel_color.x() * 255);
    buffer[4 * idx + 1] = (int)(pixel_color.y() * 255);
    buffer[4 * idx + 2] = (int)(pixel_color.z() * 255);
    buffer[4 * idx + 3] = 255;
}

__global__ void initWorld()
{
    //material* mats[4];
    mats[0] = new lambertian(color(0.8, 0.8, 0.0));
    mats[1] = new lambertian(color(0.1, 0.2, 0.5));
    //mats[2] = new metal(color(0.8, 0.8, 0.8), 0.3);
    mats[2] = new dielectric(1.5);
    mats[3] = new metal(color(0.8, 0.6, 0.2), 0.0);


    world = new hittable_list();
    world->objects.resize(4);
    world->objects[0] = new Sphere(point3(0.0, -100.5, -1.0), 100, mats[0]);
    world->objects[1] = new Sphere(point3(0.0, 0.0, -1.0), 0.5, mats[1]);
    world->objects[2] = new Sphere(point3(-1.0, 0.0, -1.0), 0.5, mats[2]);
    world->objects[3] = new Sphere(point3(1.0, 0.0, -1.0), 0.5, mats[3]);
}

int main()
{   
    // capture the start time
    cudaEvent_t   start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start, 0));

    initWorld << <1, 1 >> > ();

    //Init camera
    camera* cam = new camera;  
    cam->initialize();
   
    camera* dev_cam;
    HANDLE_ERROR(cudaMalloc((void**)&dev_cam, sizeof(camera)));
    HANDLE_ERROR(cudaMemcpy(dev_cam, cam,
        sizeof(camera),
        cudaMemcpyHostToDevice));

    //Init image buffer
    DataBlock  buffer;
    CPUBitmap bitmap(H, W, &buffer);
    unsigned char* dev_bitmap;

    // allocate memory on the GPU for the output bitmap
    HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap,
        bitmap.image_size()));

    //设置窗口分辨率
    dim3  grids(H / 16, W / 16);
    dim3  threads(16, 16);

    //初始化随机函数
    curandState* devStates;
    cudaMalloc(&devStates, H * W * sizeof(curandState));
    srand(time(0));
    int seed = rand();
    setup_rand << <grids, threads >> > (devStates, seed);

    // generate a bitmap from our sphere data
    render << <grids, threads >> > (dev_bitmap, dev_cam, devStates);

    // copy our bitmap back from the GPU for display
    HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap,
        bitmap.image_size(),
        cudaMemcpyDeviceToHost));

    // get stop time, and display the timing results
    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    float   elapsedTime;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime,
        start, stop));
    printf("Time to generate:  %3.1f ms\n", elapsedTime);

    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));

    HANDLE_ERROR(cudaFree(dev_cam));
    HANDLE_ERROR(cudaFree(dev_bitmap));
    delete cam;

    // display
    bitmap.display_and_exit();

    return 0;
}

