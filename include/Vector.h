#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

template <class T>
class Vector
{
public:

	__host__ __device__ ~Vector()
	{
		delete[] arr;
		arr = nullptr;
	}

	__host__ __device__ void push_back(T elem)
	{
		++_size;
		if (arr != nullptr)
		{
			if (_size <= capacity)
			{
				arr[_size] = elem;
			}
			else
			{
				capacity *= 2;
				T* temp = arr;
				arr = new T[capacity];
				for (int i = 0; i < _size - 1; ++i)
				{
					arr[i] = temp[i];
				}
				arr[_size] = elem;
				delete[] temp;
			}
		}
		else
		{
			capacity = 2;
			arr = new T[capacity];
			arr[0] = elem;
		}
	}

	__host__ __device__ T& operator[](int idx)
	{
		return arr[idx];
	}

	__host__ __device__ const T& operator[](int idx) const
	{
		return arr[idx];
	}

	__host__ __device__ int size() const
	{
		return _size;
	}

	//设置vector初始大小
	__host__ __device__ void resize(int size)
	{
		T* temp = arr;
		arr = new T[size];
		capacity = size;
		_size = size;
		if (temp)
			delete[] temp;
	}

	//__host__ __device__ T& operate
private:
	int capacity = 0;
	int _size = 0;
	T* arr = nullptr;
};