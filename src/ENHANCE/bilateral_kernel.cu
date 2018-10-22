/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "helper_cuda.h"
#include "EnhanceImage.h"

#include <string.h>
#include <fstream>
#include <iostream>

#include <cuda_runtime.h>
#include <npp.h>

#include <helper_string.h>
#include <helper_cuda.h>
#include <helper_math.h>
#include <math.h>
//#include <helper_functions.h>

#define clip(minv, maxv, value)  ((value)<(minv)) ? (minv) : (((value)>(maxv)) ? (maxv) : (value))

//Euclidean Distance (x, y, d) = exp((|x - y| / d)^2 / 2)
__device__ float euclideanLen(float3 a, float3 b, float d)
{
    float mod = (b.x - a.x) * (b.x - a.x) +
                (b.y - a.y) * (b.y - a.y) +
                (b.z - a.z) * (b.z - a.z);

    return __expf(-mod / (2.f * d * d));
}

extern "C"
void updateGaussian(float delta, int radius, float *fGaussian)
{
//    float  fGaussian[64];

    for (int i = 0; i < 2*radius + 1; ++i)
    {
        float x = i-radius;
        fGaussian[i] = expf(-(x*x) / (2*delta*delta));
    }

 //   checkCudaErrors(cudaMemcpyToSymbol(cGaussian, fGaussian, sizeof(float)*(2*radius+1)));
 //   checkCudaErrors(cudaMemcpy(cGaussian, fGaussian, sizeof(float)*(2*radius+1), cudaMemcpyHostToDevice));
}

extern "C"
void updateEuclidean( float e_d, float *fEuclidean, int Num)
{
	for(int i=0; i<Num; i++)
	{
		float mod = i*i;
		 fEuclidean[i] =  expf(-mod / (2.f * e_d * e_d));
	}
}

//column pass using coalesced global memory reads

__global__ void
_bilateral_filter_uchar0(uchar *od, uchar* id, int w, int h, int ch, float e_d, float *cGaussian, int r)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int src_step = w*ch;
    int pos = y*src_step+x*ch;

    if (x >= w || y >= h)
    {
        return;
    }
    else if( x<r || x>w-r-1 || y<r || y> h-r-1)
    {
    	od[pos] = id[pos];od[pos+1] = id[pos+1];od[pos+2] = id[pos+2];
    	return;
    }

    float sum = 0.0f;
    float factor;
    float3 t = {0.f, 0.f, 0.f};
    float3 center;

    center.x = id[pos];
    center.y = id[pos+1];
    center.z = id[pos+2];

    for (int i = -r; i <= r; i++)
    {
        for (int j = -r; j <= r; j++)
        {
            float3 curPix;
            int xx =  (x+j);//clip(0, w-1, (x+j));//
            int yy = (y+i);//clip(0, h-1, (y+i));//
            int pos1 = yy*src_step + xx*ch;
            curPix.x = id[pos1]; curPix.y = id[pos1+1]; curPix.z = id[pos1+2];
            factor = cGaussian[i + r] * cGaussian[j + r] *     //domain factor
                     euclideanLen(curPix, center, e_d);             //range factor

            t += factor * curPix;
            sum += factor;
        }
    }
    t = t/sum;

    od[pos] = t.x;od[pos+1] = t.y;od[pos+2] = t.z;
}

__global__ void
_bilateral_filter_uchar(uchar *od, uchar* id, uchar *gray, int w, int h, int ch, float e_d, float *cGaussian, float *cEuclidean, int r)
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int src_step = w*ch;
    int pos = y*src_step+x*ch;

    if (x >= w || y >= h)
    {
        return;
    }
    else if( x<r || x>w-r-1 || y<r || y> h-r-1)
    {
    	od[pos] = id[pos];od[pos+1] = id[pos+1];od[pos+2] = id[pos+2];
    	return;
    }

    float sum = 0.0f;
    float factor;
    float3 t = {0.f, 0.f, 0.f};
    float3 center;
    int centY;

    center.x = id[pos];
    center.y = id[pos+1];
    center.z = id[pos+2];
    centY = gray[y*w+x];

    for (int i = -r; i <= r; i++)
    {
        for (int j = -r; j <= r; j++)
        {
            float3 curPix;
            int curY;
            int xx =  (x+j);//clip(0, w-1, (x+j));//
            int yy = (y+i);//clip(0, h-1, (y+i));//
            int pos1 = yy*src_step + xx*ch;
            curPix.x = id[pos1]; curPix.y = id[pos1+1]; curPix.z = id[pos1+2];
            curY = gray[yy*w+xx];
            factor = cGaussian[i + r] * cGaussian[j + r] *     //domain factor
            		cEuclidean[abs(curY-centY)];             //range factor

            t += factor * curPix;
            sum += factor;
        }
    }
    t = t/sum;

    od[pos] = t.x;od[pos+1] = t.y;od[pos+2] = t.z;
}

// RGB version
extern "C"
int bilateralFilterRGB(uchar *dDest, uchar *dSrc, uchar *dGray,
                           int width, int height, int channels,
                           float e_d, float *cGaussian,  float *cEuclidean, int radius)
{
	dim3 gridSize((width + 16 - 1) / 16, (height + 16 - 1) / 16);
	dim3 blockSize(16, 16);
	_bilateral_filter_uchar<<< gridSize, blockSize>>>(dDest, dSrc, dGray, width, height, channels, e_d, cGaussian, cEuclidean, radius);
	return 0;
}

extern "C"
int bilateralFilterRGB0(uchar *dDest, uchar *dSrc,
                           int width, int height, int channels,
                           float e_d, float *cGaussian, int radius)
{
	dim3 gridSize((width + 16 - 1) / 16, (height + 16 - 1) / 16);
	dim3 blockSize(16, 16);
	_bilateral_filter_uchar0<<< gridSize, blockSize>>>(dDest, dSrc, width, height, channels, e_d, cGaussian, radius);
	return 0;
}

#define TX 16
#define TY 16
#define RAD 1

int divUp(int a, int b) { return (a + b - 1) / b; }

__device__
unsigned char clip_d(int n) { return n > 255 ? 255 : (n < 0 ? 0 : n); }

__device__
int idxClip(int idx, int idxMax) {
  return idx >(idxMax - 1) ? (idxMax - 1) : (idx < 0 ? 0 : idx);
}

__device__
int flatten(int col, int row, int width, int height) {
  return idxClip(col, width) + idxClip(row, height)*width;
}

__global__ void
_bilateral_filter_uchar2(uchar3 *od, uchar3* id, int w, int h, int ch, float e_d, float *cGaussian, int r)
{
	extern __shared__ uchar3 s_in[];
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int src_step = w;
    int pos = row*src_step+col;

    if (col >= w || row >= h)
    {
        return;
    }
    else if( col<r || col>w-r-1 || row<r || row> h-r-1)
    {
    	od[pos] = id[pos];
    	return;
    }

	const int idx = flatten(col, row, w, h);
	// local width and height
	const int s_w = blockDim.x + 2 * r;
	const int s_h = blockDim.y + 2 * r;
	// local indices
	const int s_col = threadIdx.x + r;
	const int s_row = threadIdx.y + r;
	const int s_idx = flatten(s_col, s_row, s_w, s_h);

	// Load regular cells
	  s_in[s_idx] = id[idx];
	  // Load halo cells
	  if (threadIdx.x < r) {
	    s_in[flatten(s_col - r, s_row, s_w, s_h)] = id[flatten(col - r, row, w, h)];
	    s_in[flatten(s_col + blockDim.x, s_row, s_w, s_h)] =  id[flatten(col + blockDim.x, row, w, h)];
	  }
	  if (threadIdx.y < r) {
	    s_in[flatten(s_col, s_row - r, s_w, s_h)] = id[flatten(col, row - r, w, h)];
	    s_in[flatten(s_col, s_row + blockDim.y, s_w, s_h)] = id[flatten(col, row + blockDim.y, w, h)];
	  }

    __syncthreads();

    float sum = 0.0f;
    float factor;
    float3 t = {0.f, 0.f, 0.f};
    float3 center;
    pos = flatten(s_col, s_row, s_w, s_h);
    center.x = s_in[pos].x;
    center.y = s_in[pos].y;
    center.z = s_in[pos].z;

    for (int i = -r; i <= r; i++)
    {
        for (int j = -r; j <= r; j++)
        {
            float3 curPix;
            int xx =  (s_col+j);
            int yy = (s_row+i);
            pos = flatten(xx, yy, s_w, s_h);
            curPix.x = s_in[pos].x;
            curPix.y = s_in[pos].y;
            curPix.z = s_in[pos].z;
            factor = cGaussian[i + r] * cGaussian[j + r] *     //domain factor
                     euclideanLen(curPix, center, e_d);             //range factor

            t += factor * curPix;
            sum += factor;
        }
    }
    t = t/sum;

    od[idx].x = clip_d(t.x);
    od[idx].y = clip_d(t.y);
    od[idx].z = clip_d(t.z);
}

extern "C"
int bilateralFilterRGB2(uchar *dDest, uchar *dSrc,
                           int width, int height, int channels,
                           float e_d, float *cGaussian, int radius)
{
	dim3 gridSize((width + 16 - 1) / 16, (height + 16 - 1) / 16);
	dim3 blockSize(16, 16);
	const size_t smSz = (TX + 2 * radius)*(TY + 2 * radius)*sizeof(uchar)*channels;
	_bilateral_filter_uchar2<<< gridSize, blockSize, smSz>>>((uchar3*)dDest, (uchar3*)dSrc, width, height, channels, e_d, cGaussian, radius);
	return 0;
}
