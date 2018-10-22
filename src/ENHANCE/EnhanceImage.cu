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


#define UMUL(a, b) ( (a) * (b) )
#define UMAD(a, b, c) ( UMUL((a), (b)) + (c) )
#define clip(minv, maxv, value)  ((value)<(minv)) ? (minv) : (((value)>(maxv)) ? (maxv) : (value))

#define		FILL_SAME_TYPE		0

#define THDBLK_SIZE	(32*6)
#if 0
__global__ void kernel_unhazed_rgb32(
		unsigned char *dst,const unsigned char *src,
			const float *guide, const unsigned char *lut,
			unsigned char A,
			int size)
{

#pragma unroll
	for (int i = UMAD(blockIdx.x, blockDim.x, threadIdx.x); i < size; i += UMUL(blockDim.x, gridDim.x))
	{
		float temp = guide[i];
		int offset = i*6;

		unsigned char value_b = (unsigned char)clip(0.0, 255.0, (src[offset] - A)/temp + A);
		unsigned char value_g = (unsigned char)clip(0.0, 255.0, (src[offset+1] - A)/temp + A);
		unsigned char value_r = (unsigned char)clip(0.0, 255.0, (src[offset+2] - A)/temp + A);
		unsigned char value_b2 = (unsigned char)clip(0.0, 255.0, (src[offset+3] - A)/temp + A);
		unsigned char value_g2 = (unsigned char)clip(0.0, 255.0, (src[offset+4] - A)/temp + A);
		unsigned char value_r2 = (unsigned char)clip(0.0, 255.0, (src[offset+5] - A)/temp + A);

		dst[offset] = lut[value_b];
		dst[offset+1] = lut[value_g];
		dst[offset+2] = lut[value_r];
		dst[offset+3] = lut[value_b2];
		dst[offset+4] = lut[value_g2];
		dst[offset+5] = lut[value_r2];
	}

}
#else
__global__ void kernel_unhazed_rgb32(
		unsigned char *dst,const unsigned char *src,
			const float *guide, const unsigned char *lut,
			unsigned char A,
			int width, int height)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if(x >= width || y >= height)
			return;

	int offset_guide = y*width + x;
	int offset = y*width*12 + x*6;

	float temp = guide[offset_guide];

	unsigned char value_b = (unsigned char)clip(0.0, 255.0, (src[offset] - A)/temp + A);
	unsigned char value_g = (unsigned char)clip(0.0, 255.0, (src[offset+1] - A)/temp + A);
	unsigned char value_r = (unsigned char)clip(0.0, 255.0, (src[offset+2] - A)/temp + A);
	unsigned char value_b2 = (unsigned char)clip(0.0, 255.0, (src[offset+3] - A)/temp + A);
	unsigned char value_g2 = (unsigned char)clip(0.0, 255.0, (src[offset+4] - A)/temp + A);
	unsigned char value_r2 = (unsigned char)clip(0.0, 255.0, (src[offset+5] - A)/temp + A);

	dst[offset] = lut[value_b];
	dst[offset+1] = lut[value_g];
	dst[offset+2] = lut[value_r];
	dst[offset+3] = lut[value_b2];
	dst[offset+4] = lut[value_g2];
	dst[offset+5] = lut[value_r2];

	offset += width*6;

	value_b = (unsigned char)clip(0.0, 255.0, (src[offset] - A)/temp + A);
	value_g = (unsigned char)clip(0.0, 255.0, (src[offset+1] - A)/temp + A);
	value_r = (unsigned char)clip(0.0, 255.0, (src[offset+2] - A)/temp + A);
	value_b2 = (unsigned char)clip(0.0, 255.0, (src[offset+3] - A)/temp + A);
	value_g2 = (unsigned char)clip(0.0, 255.0, (src[offset+4] - A)/temp + A);
	value_r2 = (unsigned char)clip(0.0, 255.0, (src[offset+5] - A)/temp + A);

	dst[offset] = lut[value_b];
	dst[offset+1] = lut[value_g];
	dst[offset+2] = lut[value_r];
	dst[offset+3] = lut[value_b2];
	dst[offset+4] = lut[value_g2];
	dst[offset+5] = lut[value_r2];
}
#endif

extern "C" int unhazed_rgb32_(
		unsigned char *dst,const unsigned char *src,
			const float *guide, const unsigned char *lut,
			unsigned char A,
			int width, int height)
{
	dim3 block((width+31)/32,(height+31)/32);
	dim3 thread(32, 32);

	//kernel_unhazed_rgb32<<<width*height/(THDBLK_SIZE)+1, THDBLK_SIZE>>>(dst,  src, guide, lut, A, width*height);
	kernel_unhazed_rgb32<<<block, thread>>>(dst,  src, guide, lut, A, width, height);

	return 0;
}

__global__ void kernel_unhazed_gray(
		unsigned char *dst,const unsigned char *src,
			const float *guide, const unsigned char *lut,
			unsigned char A,
			int width, int height)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if(x >= width || y >= height)
			return;

	int offset_guide = y*width + x;
	int offset = y*width*4 + x*2;

	float temp = guide[offset_guide];

	unsigned char value_y = (unsigned char)clip(0.0, 255.0, (src[offset] - A)/temp + A);
	unsigned char value_y2 = (unsigned char)clip(0.0, 255.0, (src[offset+1] - A)/temp + A);

	dst[offset] = lut[value_y];
	dst[offset+1] = lut[value_y2];

	offset += width*2;

	value_y = (unsigned char)clip(0.0, 255.0, (src[offset] - A)/temp + A);
	value_y2 = (unsigned char)clip(0.0, 255.0, (src[offset+1] - A)/temp + A);

	dst[offset] = lut[value_y];
	dst[offset+1] = lut[value_y2];
}

extern "C" int unhazed_gray_(
		unsigned char *dst,const unsigned char *src,
					const float *guide, const unsigned char *lut,
					unsigned char A,
					int width, int height)
{
	dim3 block((width+31)/32,(height+31)/32);
	dim3 thread(32, 32);

	kernel_unhazed_gray<<<block, thread>>>(dst,  src, guide, lut, A, width, height);

	return 0;
}

static unsigned char *d_DCM = NULL;
static unsigned char *d_DCMH = NULL;
static int blksizeDC = BLOCKSIZE;
static int widthDC = 1920 + blksizeDC;
static int heightDC = 1080 + blksizeDC;

static float *d_boxfilter_N = NULL;

__global__ void kernel_full_u8(
		unsigned char *mem, int size, unsigned char value)
{
	//printf("blockId %d thrId %d: blockDim.x = %d gredDim.x = %d \n", blockIdx.x, threadIdx.x, blockDim.x, gridDim.x);
#pragma unroll
	for (int i = UMAD(blockIdx.x, blockDim.x, threadIdx.x); i < size; i += UMUL(blockDim.x, gridDim.x))
	{
		mem[i] = value;
	}
}

__global__ void kernel_full_u32(
		unsigned int *mem, int size, unsigned int value)
{
	//printf("blockId %d thrId %d: blockDim.x = %d gredDim.x = %d \n", blockIdx.x, threadIdx.x, blockDim.x, gridDim.x);
#pragma unroll
	for (int i = UMAD(blockIdx.x, blockDim.x, threadIdx.x); i < size; i += UMUL(blockDim.x, gridDim.x))
	{
		mem[i] = value;
	}
}

__global__ void kernel_full_f32(
		float *mem, int size, float value)
{
	//printf("blockId %d thrId %d: blockDim.x = %d gredDim.x = %d \n", blockIdx.x, threadIdx.x, blockDim.x, gridDim.x);
#pragma unroll
	for (int i = UMAD(blockIdx.x, blockDim.x, threadIdx.x); i < size; i += UMUL(blockDim.x, gridDim.x))
	{
		mem[i] = value;
	}
}

__global__ void kernel_mul_f32(
		float *out, const float *in1, const float *in2, int size)
{
	//printf("blockId %d thrId %d: blockDim.x = %d gredDim.x = %d \n", blockIdx.x, threadIdx.x, blockDim.x, gridDim.x);
#pragma unroll
	for (int i = UMAD(blockIdx.x, blockDim.x, threadIdx.x); i < size; i += UMUL(blockDim.x, gridDim.x))
	{
		out[i] = in1[i] * in2[i];
	}
}

__global__ void kernel_mulq_f32(
		unsigned char *out, float *in1, float *in2, int size)
{
	//printf("blockId %d thrId %d: blockDim.x = %d gredDim.x = %d \n", blockIdx.x, threadIdx.x, blockDim.x, gridDim.x);
#pragma unroll
	for (int i = UMAD(blockIdx.x, blockDim.x, threadIdx.x); i < size; i += UMUL(blockDim.x, gridDim.x))
	{
		float value = in1[i] * in2[i];
		out[i] = (value > 255) ? 255 : (unsigned char)value;
	}
}

//Internal memory allocation
extern "C" void initDCM(int width, int height, int blockSize)
{
	int dcmSize;
	blksizeDC = blockSize;
	widthDC = (width + blockSize + 31)&(~31);
	heightDC = (height + blockSize + 31)&(~31);
	dcmSize = widthDC * heightDC;

	checkCudaErrors(cudaMalloc((void **)&d_DCM, dcmSize * sizeof(unsigned char)));
    checkCudaErrors(cudaMalloc((void **)&d_DCMH, dcmSize * sizeof(unsigned char)));
    kernel_full_u8<<<heightDC, widthDC>>>( d_DCM, dcmSize, 255 );
    kernel_full_u8<<<heightDC, widthDC>>>( d_DCMH, dcmSize, 255 );

    checkCudaErrors(cudaMalloc((void **)&d_boxfilter_N, width * height * sizeof(float)));
    kernel_full_f32<<<heightDC, widthDC>>>( d_boxfilter_N, width * height, 1.0 );
}

//Internal memory deallocation
extern "C" void unInitDCM(void)
{
    checkCudaErrors(cudaFree(d_DCM)); d_DCM = NULL;
    checkCudaErrors(cudaFree(d_DCMH)); d_DCMH = NULL;
    checkCudaErrors(cudaFree(d_boxfilter_N)); d_boxfilter_N = NULL;
}

__global__ void kernel_darkChannel_rgb32(
		float *dst,float *gray,unsigned char *dark,const unsigned char *src,
			int width, int height, unsigned char *dcm, unsigned char *dcmh, int dcm_skip, int dcmh_skip, int rcLen, float alph)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if(x >= width || y >= height)
			return;

	int halfrclen = (rcLen>>1);
	int offset_dst = y*width+x;
	int offset_src = y*width*12 + x*6;//offset_dst*6;//y*width*3 + 3*x;
	int offset_dcm = (y+halfrclen)*dcm_skip + x + halfrclen;
	unsigned char B = src[offset_src];
	unsigned char G = src[offset_src+1];
	unsigned char R = src[offset_src+2];

	//gray[offset_dst] = (0.299*R + 0.587*G + 0.114*B)*0.003921569;//255.0;
	//__syncthreads();

	dcm[offset_dcm] = MinValue(R, G, B);
	gray[offset_dst] = dcm[offset_dcm]*0.003921569;
	__syncthreads();

	unsigned char minV = 255;

#pragma unroll
	for(int i= -(halfrclen); i<halfrclen; i++){
		minV = (minV < (dcm[offset_dcm + i])) ? minV : (dcm[offset_dcm + i]);
	}
	dcmh[offset_dcm] = minV;
	__syncthreads();

	minV = 255;
#pragma unroll
	for(int j= -(halfrclen); j<halfrclen; j++){
		minV = (minV < (dcmh[offset_dcm + j*dcm_skip])) ? minV : (dcmh[offset_dcm + j*dcm_skip]);
	}
	//for(int j= -(halfrclen*dcm_skip); j<halfrclen*dcm_skip; j+=dcm_skip){
	//	minV = (minV < (dcmh[offset_dcm + j])) ? minV : (dcmh[offset_dcm + j]);
	//}
	dark[offset_dst] = minV;
	float ftmp = (1.0f-(minV*alph));
	dst[offset_dst] = /*(ftmp < 0.000001) ? 0.0 : */ftmp;
}

extern "C" int darkChannel_rgb32_(
		float *dst, float *gray, unsigned char *dark, const unsigned char *src,
			int width, int height, float acl, float weight)
{
	dim3 block((width+31)/32,(height+31)/32);
	dim3 thread(32, 32);

	kernel_darkChannel_rgb32<<<block, thread>>>(dst, gray, dark, src, width, height, d_DCM, d_DCMH, widthDC, heightDC, blksizeDC, weight/acl);

	return 0;
}

__global__ void kernel_darkChannel_gray(
		float *dst,float *gray,unsigned char *dark,const unsigned char *src,
			int width, int height, unsigned char *dcm, unsigned char *dcmh, int dcm_skip, int dcmh_skip, int rcLen, float alph)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if(x >= width || y >= height)
			return;

	int halfrclen = (rcLen>>1);
	int offset_dst = y*width+x;
	int offset_src = y*width*4 + x*2;
	int offset_dcm = (y+halfrclen)*dcm_skip + x + halfrclen;
	unsigned char Y = src[offset_src];

	dcm[offset_dcm] = Y;
	gray[offset_dst] = dcm[offset_dcm]*0.003921569;
	__syncthreads();

	unsigned char minV = 255;

#pragma unroll
	for(int i= -(halfrclen); i<halfrclen; i++){
		minV = (minV < (dcm[offset_dcm + i])) ? minV : (dcm[offset_dcm + i]);
	}
	dcmh[offset_dcm] = minV;
	__syncthreads();

	minV = 255;
#pragma unroll
	for(int j= -(halfrclen); j<halfrclen; j++){
		minV = (minV < (dcmh[offset_dcm + j*dcm_skip])) ? minV : (dcmh[offset_dcm + j*dcm_skip]);
	}
	//for(int j= -(halfrclen*dcm_skip); j<halfrclen*dcm_skip; j+=dcm_skip){
	//	minV = (minV < (dcmh[offset_dcm + j])) ? minV : (dcmh[offset_dcm + j]);
	//}
	dark[offset_dst] = minV;
	float ftmp = (1.0f-(minV*alph));
	dst[offset_dst] = /*(ftmp < 0.000001) ? 0.0 : */ftmp;
}

extern "C" int darkChannel_gray_(
		float *dst, float *gray, unsigned char *dark, const unsigned char *src,
			int width, int height, float acl, float weight)
{
	dim3 block((width+31)/32,(height+31)/32);
	dim3 thread(32, 32);

	kernel_darkChannel_gray<<<block, thread>>>(dst, gray, dark, src, width, height, d_DCM, d_DCMH, widthDC, heightDC, blksizeDC, weight/acl);

	return 0;
}

__global__ void kernel_norgray2rgb32(
		unsigned char *dst, const float *src, int width, int height)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if(x >= width || y >= height)
			return;
	int offset_src = y*width+x;
	int offset_dst = y*width*12+x*6;
	float v = src[offset_src];
	dst[offset_dst] = v*255.0;
	dst[offset_dst+1] = v*255.0;
	dst[offset_dst+2] = v*255.0;
	dst[offset_dst+3] = v*255.0;
	dst[offset_dst+4] = v*255.0;
	dst[offset_dst+5] = v*255.0;
	offset_dst += width*6;
	dst[offset_dst] = v*255.0;
	dst[offset_dst+1] = v*255.0;
	dst[offset_dst+2] = v*255.0;
	dst[offset_dst+3] = v*255.0;
	dst[offset_dst+4] = v*255.0;
	dst[offset_dst+5] = v*255.0;
}

extern "C" int norgray2rgb32_(
		unsigned char *dst, const float *src, int width, int height)
{
	dim3 block((width+31)/32,(height+31)/32);
	dim3 thread(32, 32);

	kernel_norgray2rgb32<<<block, thread>>>(dst, src, width, height);

	return 0;
}

__global__ void kernel_norgray2gray(
		unsigned char *dst, const float *src, int width, int height)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if(x >= width || y >= height)
			return;
	int offset_src = y*width+x;
	int offset_dst = y*width*4+x*2;
	float v = src[offset_src];
	dst[offset_dst] = v*255.0;
	dst[offset_dst+1] = v*255.0;
	offset_dst += width*2;
	dst[offset_dst] = v*255.0;
	dst[offset_dst+1] = v*255.0;
}

extern "C" int norgray2gray_(
		unsigned char *dst, const float *src, int width, int height)
{
	dim3 block((width+31)/32,(height+31)/32);
	dim3 thread(32, 32);

	kernel_norgray2gray<<<block, thread>>>(dst, src, width, height);

	return 0;
}

__global__ void kernel_rgb32_2norgray(
		float *gray, const unsigned char *src,
				int width, int height)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if(x >= width || y >= height)
			return;

	int offset_dst = y*width+x;
	int offset_src = offset_dst*3;//y*width*3 + 3*x;
	unsigned char B = src[offset_src];
	unsigned char G = src[offset_src+1];
	unsigned char R = src[offset_src+2];

	gray[offset_dst] = (0.299*R + 0.587*G + 0.114*B)*0.003921569;//255.0;
}

extern "C" int rgb32_2norgray_(
		float *gray, const unsigned char *src,
				int width, int height)
{
	dim3 block((width+31)/32,(height+31)/32);
	dim3 thread(32, 32);

	kernel_rgb32_2norgray<<<block, thread>>>(gray, src, width, height);

	return 0;
}

// process row
__device__ void
d_boxfilter_x(const float *id, float *od, int w, int h, int r, float scale)
{
    float t;
    // do left edge
#if FILL_SAME_TYPE
    t = id[0] * r;

    for (int x = 0; x < (r + 1); x++)
    {
        t += id[x];
    }

    od[0] = t * scale;

    for (int x = 1; x < (r + 1); x++)
    {
        t += id[x + r];
        t -= id[0];
        od[x] = t * scale;
    }
#else
    t = -id[0];
    for (int x = 0; x < (r + 1); x++)
     {
         t += 2*id[x];
     }

     od[0] = t * scale;

     for (int x = 1; x < (r + 1); x++)
     {
         t += id[x + r];
         t -= id[r+1-x];
         od[x] = t * scale;
     }

#endif

    // main loop
    for (int x = (r + 1); x < w - r; x++)
    {
        t += id[x + r];
        t -= id[x - r - 1];
        od[x] = t * scale;
    }

    // do right edge
#if FILL_SAME_TYPE
    for (int x = w - r; x < w; x++)
    {
        t += id[w - 1];
        t -= id[x - r - 1];
        od[x] = t * scale;
    }
#else
    for (int x = w - r; x < w; x++)
    {
        t += id[w - 1-(x+r-w)];
        t -= id[x - r - 1];
        od[x] = t * scale;
    }
#endif
}

// process column
__device__ void
d_boxfilter_y(const float *id, float *od, int w, int h, int r, float scale)
{
    //float scale = 1.0f / (float)((r << 1) + 1);

    float t;
    // do left edge
#if FILL_SAME_TYPE
    t = id[0] * r;

    for (int y = 0; y < (r + 1); y++)
    {
        t += id[y * w];
    }

    od[0] = t * scale;

    for (int y = 1; y < (r + 1); y++)
    {
        t += id[(y + r) * w];
        t -= id[0];
        od[y * w] = t * scale;
    }
#else
    t = -id[0];

     for (int y = 0; y < (r + 1); y++)
     {
         t += id[y * w]*2;
     }

     od[0] = t * scale;

     for (int y = 1; y < (r + 1); y++)
     {
         t += id[(y + r) * w];
         t -= id[(r+1-y) * w];
         od[y * w] = t * scale;
     }

#endif

    // main loop
    for (int y = (r + 1); y < (h - r); y++)
    {
        t += id[(y + r) * w];
        t -= id[((y - r) * w) - w];
        od[y * w] = t * scale;
    }

    // do right edge
#if FILL_SAME_TYPE
    for (int y = h - r; y < h; y++)
    {
        t += id[(h-1) * w];
        t -= id[((y - r) * w) - w];
        od[y * w] = t * scale;
    }
#else
    for (int y = h - r; y < h; y++)
	{
		t += id[(h-1-(y+r-h)) * w];
		t -= id[((y - r) * w) - w];
		od[y * w] = t * scale;
	}
#endif
}

// process row
__device__ void
d_boxfilter_x_imul(const float *id1, const float *id2, float *od, int w, int h, int r, float scale)
{
    //float scale = 1.0f / (float)((r << 1) + 1);

    float t;
    // do left edge
#if FILL_SAME_TYPE
    t = id1[0]*id2[0] * r;

    for (int x = 0; x < (r + 1); x++)
    {
        t += id1[x]*id2[x];
    }

    od[0] = t * scale;

    for (int x = 1; x < (r + 1); x++)
    {
        t += id1[x + r]*id2[x + r];
        t -= id1[0]*id2[0];
        od[x] = t * scale;
    }
#else
    t = -id1[0]*id2[0];

	for (int x = 0; x < (r + 1); x++)
	{
		t += id1[x]*id2[x]*2;
	}

	od[0] = t * scale;

	for (int x = 1; x < (r + 1); x++)
	{
		t += id1[x + r]*id2[x + r];
		t -= id1[r+1-x]*id2[r+1-x];
		od[x] = t * scale;
	}

#endif

    // main loop
    for (int x = (r + 1); x < w - r; x++)
    {
        t += id1[x + r]*id2[x + r];
        t -= id1[x - r - 1]*id2[x - r - 1];
        od[x] = t * scale;
    }

    // do right edge
#if FILL_SAME_TYPE
    for (int x = w - r; x < w; x++)
    {
        t += id1[w - 1]*id2[w - 1];
        t -= id1[x - r - 1]*id2[x - r - 1];
        od[x] = t * scale;
    }
#else
    for (int x = w - r; x < w; x++)
	{
		t += id1[w - 1-(x+r-w)]*id2[w - 1-(x+r-w)];
		t -= id1[x - r - 1]*id2[x - r - 1];
		od[x] = t * scale;
	}

#endif
}

__global__ void
d_boxfilter_x_global(const float *id, float *od, int w, int h, int r, float scale)
{
    unsigned int y = blockIdx.x*blockDim.x + threadIdx.x;
    d_boxfilter_x(&id[y * w], &od[y * w], w, h, r, scale);
}

__global__ void
d_boxfilter_y_global(const float *id, float *od, int w, int h, int r, float scale)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    d_boxfilter_y(&id[x], &od[x], w, h, r, scale);
}

__global__ void
d_boxfilter_x_imul_global(const float *id1, const float *id2, float *od, int w, int h, int r, float scale)
{
    unsigned int y = blockIdx.x*blockDim.x + threadIdx.x;
    d_boxfilter_x_imul(&id1[y * w], &id2[y * w], &od[y * w], w, h, r, scale);
}

// a = (mean_IP - mean_I.mul(mean_P))/(mean_II - mean_I.mul(mean_I));
// b = mean_P - a.mul(mean_I);
__global__ void kernel_guideFilter_AB(
		const float *mean_I, const float *mean_P, const float *mean_IP, const float *mean_II,
		float *d_A, float *d_B,
				int size)
{
	int i;
#pragma unroll
	for (i = UMAD(blockIdx.x, blockDim.x, threadIdx.x); i < size; i += UMUL(blockDim.x, gridDim.x))
	{
		d_A[i] = (mean_IP[i] - mean_I[i]*mean_P[i])/(mean_II[i] - mean_I[i]*mean_I[i] + EPXITON);
		//d_B[i] = mean_P[i] - d_A[i]*mean_I[i];
	}

#pragma unroll
	for (i = UMAD(blockIdx.x, blockDim.x, threadIdx.x); i < size; i += UMUL(blockDim.x, gridDim.x))
	{
		//d_A[i] = (mean_IP[i] - mean_I[i]*mean_P[i])/(mean_II[i] - mean_I[i]*mean_I[i] + EPXITON);
		d_B[i] = mean_P[i] - d_A[i]*mean_I[i];
	}
}

//guideFiltImg = (unsigned char)255*(mean_a.mul(guide) + mean_b)
__global__ void kernel_guideFilter_guide(
		const float *mean_a, const float *mean_b, const float *norgray,
		float* dst,
				int size)
{
#pragma unroll
	for (int i = UMAD(blockIdx.x, blockDim.x, threadIdx.x); i < size; i += UMUL(blockDim.x, gridDim.x))
	{
		dst[i] = (mean_a[i] * norgray[i] + mean_b[i]);
		//if(dst[i] < TRANSMISSIONRATIO)
		//	dst[i] = TRANSMISSIONRATIO;
	}

#pragma unroll
	for (int i = UMAD(blockIdx.x, blockDim.x, threadIdx.x); i < size; i += UMUL(blockDim.x, gridDim.x))
	{
		if(dst[i] < TRANSMISSIONRATIO)
			dst[i] = TRANSMISSIONRATIO;
	}
}

static void nppBoxFilter(float *pDst, const float *pSrc, int width, int height)
{
		Npp32s nSrcStep, nDstStep;

		nSrcStep = width*sizeof(float);
		nDstStep = width*sizeof(float);
		NppiSize oSrcSize = {(int)width, (int)height};
		NppiSize oSizeROI = {(int)width, (int)height};
		NppiSize oMaskSize = {RADIUS, RADIUS};
		NppiPoint oSrcOffset = {0, 0};
		NppiPoint oAnchor = {oMaskSize.width / 2, oMaskSize.height / 2};

		NppStatus status =
		nppiFilterBoxBorder_32f_C1R((Npp32f *)pSrc, nSrcStep, oSrcSize, oSrcOffset, (Npp32f *)pDst, nDstStep, oSizeROI,
		                             oMaskSize, oAnchor, NPP_BORDER_REPLICATE);
		checkCudaErrors((cudaError_t)status);
}

static void nppBoxFilter_8u(unsigned char *pDst, const unsigned char *pSrc, int width, int height)
{
	Npp32s nSrcStep, nDstStep;

	nSrcStep = width*sizeof(float);
	nDstStep = width*sizeof(float);
	NppiSize oSrcSize = {(int)width, (int)height};
	NppiSize oSizeROI = {(int)width, (int)height};
	NppiSize oMaskSize = {RADIUS, RADIUS};
	NppiPoint oSrcOffset = {0, 0};
	NppiPoint oAnchor = {oMaskSize.width / 2, oMaskSize.height / 2};

	NppStatus status =
			nppiFilterBoxBorder_8u_C1R((Npp8u *)pSrc, nSrcStep, oSrcSize, oSrcOffset, (Npp8u *)pDst, nDstStep, oSizeROI,
	                             oMaskSize, oAnchor, NPP_BORDER_REPLICATE);
	checkCudaErrors((cudaError_t)status);

}

extern "C" int full_f32_(
		float *mem, int size, float value)
{
	kernel_full_f32<<<size/THDBLK_SIZE+1, THDBLK_SIZE>>>( mem, size, value );
	return 0;
}

extern "C" int full_u8_(
		unsigned char *mem, int size, unsigned char value)
{
	kernel_full_u8<<<size/THDBLK_SIZE+1, THDBLK_SIZE>>>( mem, size, value );
	return 0;
}

extern "C" int guideFilter_(
		float *dst,
		const float *tran, const float *norgray,
		int width, int height, cudaStream_t stream[])
{
	static float *m_mean_I, *m_mean_P, *m_mean_IP, *m_mean_II;
	static bool m_bCreate = false;
	static unsigned int count = 0;

	float *temp0, *temp1, *temp2, *temp3, *temp4, *temp5;
	float *d_A,*d_B, *mean_a, *mean_b;

	static int matsize  = 0;
	int r = RADIUS;
	float boxFilterScale = 1.0f / (float)((r << 1)+1);

	if(matsize != (width * height *sizeof(float)))
	{
		count = 0;
		matsize  = width * height *sizeof(float);
		if(m_bCreate){
			checkCudaErrors(cudaFree(m_mean_I));
			checkCudaErrors(cudaFree(m_mean_P));
			checkCudaErrors(cudaFree(m_mean_IP));
			checkCudaErrors(cudaFree(m_mean_II));
			m_bCreate = false;
		}
	}

	if(matsize == 0)
		return 0;

	if(!m_bCreate){
	checkCudaErrors(cudaMalloc((void **)&m_mean_I, matsize));
	checkCudaErrors(cudaMalloc((void **)&m_mean_P, matsize));
	checkCudaErrors(cudaMalloc((void **)&m_mean_IP, matsize));
	checkCudaErrors(cudaMalloc((void **)&m_mean_II, matsize));
	m_bCreate = true;
	}

	checkCudaErrors(cudaMalloc((void **)&temp0, matsize));
	checkCudaErrors(cudaMalloc((void **)&temp1, matsize));
	checkCudaErrors(cudaMalloc((void **)&temp2, matsize));
	checkCudaErrors(cudaMalloc((void **)&temp3, matsize));
	checkCudaErrors(cudaMalloc((void **)&temp4, matsize));
	checkCudaErrors(cudaMalloc((void **)&temp5, matsize));

	if(RADIUS<4)
	{
		//int nthreads_x = 4,  nthreads_y= 4;
		//d_boxfilter_x_global<<<height/nthreads_x, nthreads_x, 1>>>(tran, temp1, width, height, r, boxFilterScale);
		//d_boxfilter_y_global<<<width/nthreads_y, nthreads_y, 1>>>(temp1, dst, width, height, r, boxFilterScale);
		//nppBoxFilter(dst, tran, width, height);

		//nppBoxFilter(mean_I, norgray, width, height);
		//nppBoxFilter(mean_P, tran, width, height);
		//kernel_mul_f32<<<width*height/(32*6)+1, 32*6>>>(temp2, norgray, tran, width*height);
		//nppBoxFilter(mean_IP, temp2, width, height);
		//kernel_mul_f32<<<width*height/(32*6)+1, 32*6>>>(temp3, norgray, norgray, width*height);
		//nppBoxFilter(mean_II, temp3, width, height);

		//float *d_A = temp0;
		//float *d_B = temp1;
		//float *mean_a = temp2;
		//float *mean_b = temp3;

		d_A = temp0;
		d_B = temp1;

		//kernel_guideFilter_AB<<<block, thread>>>(mean_I, mean_P, mean_IP, mean_II, d_A, d_B, width, height);
		kernel_mul_f32<<<width*height/(THDBLK_SIZE)+1, THDBLK_SIZE>>>(m_mean_IP, norgray, tran, width*height);
		kernel_mul_f32<<<width*height/(THDBLK_SIZE)+1, THDBLK_SIZE>>>(m_mean_II, norgray, norgray, width*height);
		kernel_guideFilter_AB<<<width*height/(THDBLK_SIZE)+1, THDBLK_SIZE>>>(norgray, tran, m_mean_IP, m_mean_II, d_A, d_B, width*height);
		kernel_guideFilter_guide<<<width*height/(THDBLK_SIZE)+1, THDBLK_SIZE>>>(d_A, d_B, norgray, dst, width*height);

		//nppBoxFilter(mean_a, d_A, width, height);
		//nppBoxFilter(mean_b, d_B, width, height);

		//checkCudaErrors(cudaDeviceSynchronize());

		//kernel_guideFilter_guide<<<block, thread>>>(mean_a, mean_b, norgray, dst, width, height);
	}
	else
	{
		int nthreads_x = 4,  nthreads_y= 4;

		if(count==0 || (count%2) == 0){

			d_boxfilter_x_global<<<height/nthreads_x, nthreads_x, 0, stream[0]>>>(norgray, temp0, width, height, r, boxFilterScale);
			d_boxfilter_y_global<<<width/nthreads_y, nthreads_y, 0, stream[0]>>>(temp0, m_mean_I, width, height, r, boxFilterScale);

			d_boxfilter_x_global<<<height/nthreads_x, nthreads_x, 0, stream[1]>>>(tran, temp1, width, height, r, boxFilterScale);
			d_boxfilter_y_global<<<width/nthreads_y, nthreads_y, 0, stream[1]>>>(temp1, m_mean_P, width, height, r, boxFilterScale);

			d_boxfilter_x_imul_global<<<height/nthreads_x, nthreads_x, 0, stream[2]>>>(norgray, tran, temp2, width, height, r, boxFilterScale);
			d_boxfilter_y_global<<<width/nthreads_y, nthreads_y, 0, stream[2]>>>(temp2, m_mean_IP, width, height, r, boxFilterScale);

			d_boxfilter_x_imul_global<<<height/nthreads_x, nthreads_x, 0, stream[3]>>>(norgray, norgray, temp3, width, height, r, boxFilterScale);
			d_boxfilter_y_global<<<width/nthreads_y, nthreads_y, 0, stream[3]>>>(temp3, m_mean_II, width, height, r, boxFilterScale);

		}

		if(count==0 || (count%2) == 1){

			d_A = temp0;
			d_B = temp1;
			mean_a = temp2;
			mean_b = temp3;
			checkCudaErrors(cudaMalloc((void **)&mean_a, matsize));
			checkCudaErrors(cudaMalloc((void **)&mean_b, matsize));

			checkCudaErrors(cudaDeviceSynchronize());

			kernel_guideFilter_AB<<<width*height/(THDBLK_SIZE)+1, THDBLK_SIZE>>>(m_mean_I, m_mean_P, m_mean_IP, m_mean_II, d_A, d_B, width*height);

			d_boxfilter_x_global<<<height/nthreads_x, nthreads_x, 0, stream[0]>>>(d_A, temp4, width, height, r, boxFilterScale);
			d_boxfilter_y_global<<<width/nthreads_y, nthreads_y, 0, stream[0]>>>(temp4, mean_a, width, height, r, boxFilterScale);

			d_boxfilter_x_global<<<height/nthreads_x, nthreads_x, 0, stream[1]>>>(d_B, temp5, width, height, r, boxFilterScale);
			d_boxfilter_y_global<<<width/nthreads_y, nthreads_y, 0, stream[1]>>>(temp5, mean_b, width, height, r, boxFilterScale);

			checkCudaErrors(cudaDeviceSynchronize());
			kernel_guideFilter_guide<<<width*height/(THDBLK_SIZE)+1, THDBLK_SIZE>>>(mean_a, mean_b, norgray, dst, width*height);

			checkCudaErrors(cudaFree(mean_a));
			checkCudaErrors(cudaFree(mean_b));

		}

		count ++;
	}

	checkCudaErrors(cudaFree(temp0));
	checkCudaErrors(cudaFree(temp1));
	checkCudaErrors(cudaFree(temp2));
	checkCudaErrors(cudaFree(temp3));
	checkCudaErrors(cudaFree(temp4));
	checkCudaErrors(cudaFree(temp5));

	return 0;
}

__global__ void
kernel_clahe_hist(unsigned int *hist, const unsigned char *dLUT, const unsigned char *data,
		int width, int height, unsigned int uiXSize, unsigned int uiYSize)
{
	//printf("%d x %d  %d x %d  %d , %d   %d , %d\n",
	//		gridDim.x, gridDim.y, blockDim.x, blockDim.y, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);

	unsigned int *histSub = hist + (blockIdx.y * gridDim.x + blockIdx.x)*256;
	int offsetBase = blockIdx.y*uiYSize*width + blockIdx.x*uiXSize;
	for(int j=threadIdx.y; j<uiYSize; j+= 32){
		int offset = offsetBase + j*width;
		for(int i=threadIdx.x; i<uiXSize; i+=32){
			atomicAdd(histSub+dLUT[data[offset+i]], 1);
		}
	}
}

__global__ void
kernel_clahe_hist_roi(unsigned int *hist, const unsigned char *dLUT, const unsigned char *data,
		int width, int height, int startX, int startY, int validW, int validH, unsigned int uiXSize, unsigned int uiYSize)
{
	//printf("%d x %d  %d x %d  %d , %d   %d , %d\n",
	//		gridDim.x, gridDim.y, blockDim.x, blockDim.y, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);

	unsigned int *histSub = hist + (blockIdx.y * gridDim.x + blockIdx.x)*256;
	int offsetBase = (startY+blockIdx.y*uiYSize)*width + blockIdx.x*uiXSize + startX;
	for(int j=threadIdx.y; j<uiYSize; j+= 32){
		int offset = offsetBase + j*width;
		for(int i=threadIdx.x; i<uiXSize; i+=32){
			atomicAdd(histSub+dLUT[data[offset+i]], 1);
		}
	}
}

extern "C" int clahe_hist_(
		unsigned int *hist, const unsigned char *dLUT, const unsigned char *data,
		int width, int height, unsigned int uiNrX, unsigned int uiNrY, cudaStream_t stream[])
{
	unsigned int uiXSize = width/uiNrX, uiYSize = height/uiNrY;
	//unsigned int histLen = uiNrX*uiNrY*256;
	//unsigned int histOnceLen = histLen/8;

	dim3 block(uiNrX, uiNrY);
	dim3 thread(32,32);

	kernel_full_u32<<<uiNrX*uiNrY*256/THDBLK_SIZE, THDBLK_SIZE>>>( hist, uiNrX*uiNrY*256, 0 );

	kernel_clahe_hist<<< block, thread >>>(hist, dLUT, data, width, height, uiXSize, uiYSize);

	return 0;
}

extern "C" int clahe_hist_roi(
		unsigned int *hist, const unsigned char *dLUT, const unsigned char *data,
		int width, int height, int startX, int startY, int validW, int validH,
		unsigned int uiNrX, unsigned int uiNrY, cudaStream_t stream[])
{
	unsigned int uiXSize = validW/uiNrX, uiYSize = validH/uiNrY;
	//unsigned int histLen = uiNrX*uiNrY*256;
	//unsigned int histOnceLen = histLen/8;

	dim3 block(uiNrX, uiNrY);
	dim3 thread(32,32);

	kernel_full_u32<<<uiNrX*uiNrY*256/THDBLK_SIZE, THDBLK_SIZE>>>( hist, uiNrX*uiNrY*256, 0 );

	kernel_clahe_hist_roi<<< block, thread >>>(hist, dLUT, data, width, height, startX, startY, validW, validH, uiXSize, uiYSize);

	return 0;
}

__global__ void
kernel_clahe_interpolate(unsigned char *pDst, unsigned char *pSrc, int uiXRes,
		unsigned int *pulMapLU, unsigned int *pulMapRU, unsigned int *pulMapLB,  unsigned int *pulMapRB,
        unsigned int uiXSize, unsigned int uiYSize, const unsigned char *pLUT, float fNum)
{
	int x = threadIdx.x;
	int y = blockIdx.x;
	unsigned char *curImg = pSrc + y * uiXRes + x;
	unsigned char *curDst = pDst + y * uiXRes + x;
	unsigned int uiXCoef = x, uiYCoef = y , uiXInvCoef = uiXSize - x, uiYInvCoef = uiYSize - y;

	unsigned char GreyValue = (unsigned char)pLUT[*curImg];
    *curDst = (unsigned char) ((uiYInvCoef * (uiXInvCoef*pulMapLU[GreyValue] + uiXCoef * pulMapRU[GreyValue])
							  + uiYCoef * (uiXInvCoef * pulMapLB[GreyValue] + uiXCoef * pulMapRB[GreyValue])) * fNum);

    //*curDst = *curImg;
}

extern "C" int clahe_interpolate_(
		unsigned char *pDst, unsigned char *pSrc, int uiXRes,
		unsigned int *pulMapLU, unsigned int *pulMapRU, unsigned int *pulMapLB,  unsigned int *pulMapRB,
        unsigned int uiXSize, unsigned int uiYSize, const unsigned char *pLUT, cudaStream_t stream)
{
	kernel_clahe_interpolate<<<uiYSize, uiXSize, 0, stream>>>(pDst, pSrc, uiXRes,pulMapLU,pulMapRU,pulMapLB,pulMapRB,
			uiXSize,uiYSize,pLUT, 1.0f/(uiXSize*uiYSize));

	return 0;
}

#define BLCNT	(16*16)

typedef struct _clahe_ctx{
	unsigned int uiSubX[BLCNT], uiSubY[BLCNT];
	unsigned int *pulLU[BLCNT], *pulLB[BLCNT], *pulRU[BLCNT], *pulRB[BLCNT];
	unsigned char *blkDst[BLCNT], *blkSrc[BLCNT];
	float fnum[BLCNT];
}clahe_ctx;

__global__ void
kernel_clahe_interpolate_one(clahe_ctx *dctx, int uiXRes, const unsigned char *pLUT, int blkCnt)
{
	int x = threadIdx.x;
	int y = blockIdx.x;

	for(int i=0; i<blkCnt; i++)
	{
		if(x<dctx->uiSubX[i] && y<dctx->uiSubY[i]){
		unsigned char *curImg = dctx->blkSrc[i] + y * uiXRes + x;
		unsigned char *curDst = dctx->blkDst[i] + y * uiXRes + x;
		//*curDst = *curImg;

		unsigned int uiXCoef = x, uiYCoef = y , uiXInvCoef = dctx->uiSubX[i] - x, uiYInvCoef = dctx->uiSubY[i] - y;
		unsigned int *pulLU = dctx->pulLU[i];
		unsigned int *pulLB = dctx->pulLB[i];
		unsigned int *pulRU = dctx->pulRU[i];
		unsigned int *pulRB = dctx->pulRB[i];

		unsigned char GreyValue = (unsigned char)pLUT[*curImg];
	    *curDst = (unsigned char) ((uiYInvCoef * (uiXInvCoef*pulLU[GreyValue] + uiXCoef * pulRU[GreyValue])
								  + uiYCoef * (uiXInvCoef * pulLB[GreyValue] + uiXCoef * pulRB[GreyValue])) * dctx->fnum[i]);
		}

	}
}

extern "C" int clahe_Interpolate_one_(unsigned char *pDst, unsigned char *pSrc,
		unsigned char *dLUT, unsigned int *dHist,
		unsigned int uiXRes, unsigned int uiYRes, unsigned int uiNrX, unsigned int uiNrY,
		unsigned int uiNrBins, cudaStream_t stream)
{
	unsigned int uiXSize = uiXRes/uiNrX, uiYSize = uiYRes/uiNrY;

	clahe_ctx ctx, *d_ctx;
	unsigned int uiSubYTmp;
	unsigned int uiXL, uiXR, uiYU, uiYB;
	unsigned int uiX, uiY;
	int blkCnt = 0;

	/* Interpolate greylevel mappings to get CLAHE image */
	for (uiY = 0; uiY <= uiNrY; uiY++)
	{
		if (uiY == 0)
		{                     /* special case: top row */
			uiSubYTmp = uiYSize >> 1;  uiYU = 0; uiYB = 0;
		}
		else
		{
			if (uiY == uiNrY) {               /* special case: bottom row */
				uiSubYTmp = uiYSize >> 1;    uiYU = uiNrY-1;  uiYB = uiYU;
			}
			else
			{                     /* default values */
				uiSubYTmp = uiYSize; uiYU = uiY - 1; uiYB = uiYU + 1;
			}
		}
		for (uiX = 0; uiX <= uiNrX; uiX++)
		{
			ctx.uiSubY[blkCnt] = uiSubYTmp;
			if (uiX == 0)
			{                 /* special case: left column */
				ctx.uiSubX[blkCnt] = uiXSize >> 1; uiXL = 0; uiXR = 0;
			}
			else
			{
				if (uiX == uiNrX)
				{             /* special case: right column */
					ctx.uiSubX[blkCnt] = uiXSize >> 1;  uiXL = uiNrX - 1; uiXR = uiXL;
				}
				else
				{                     /* default values */
					ctx.uiSubX[blkCnt] = uiXSize; uiXL = uiX - 1; uiXR = uiXL + 1;
				}
			}

			ctx.pulLU[blkCnt] = dHist + (uiNrBins * (uiYU * uiNrX + uiXL));
			ctx.pulRU[blkCnt] = dHist + (uiNrBins * (uiYU * uiNrX + uiXR));
			ctx.pulLB[blkCnt] = dHist + (uiNrBins * (uiYB * uiNrX + uiXL));
			ctx.pulRB[blkCnt] = dHist + (uiNrBins * (uiYB * uiNrX + uiXR));
			ctx.blkDst[blkCnt] = pDst;
			ctx.blkSrc[blkCnt] = pSrc;
			ctx.fnum[blkCnt] = 1.0f/(ctx.uiSubX[blkCnt]*ctx.uiSubY[blkCnt]);
			//clahe_interpolate_(blkDst[blkCnt], blkSrc[blkCnt],
			//		uiXRes,pulLU[blkCnt],pulRU[blkCnt],pulLB[blkCnt],pulRB[blkCnt],
			//		uiSubX[blkCnt],uiSubY[blkCnt],dLUT,stream);
			//kernel_clahe_interpolate<<<uiYSize, uiXSize, 0, stream>>>(blkDst[blkCnt], blkSrc[blkCnt], uiXRes,
			//		pulLU[blkCnt],pulRU[blkCnt],pulLB[blkCnt],pulRB[blkCnt],
			//		uiSubX,uiSubY,dLUT, 1.0f/(uiSubX*uiSubY));
			pSrc += ctx.uiSubX[blkCnt];
			pDst += ctx.uiSubX[blkCnt];

			blkCnt ++;
		}
		pSrc += (uiSubYTmp - 1) * uiXRes;
		pDst += (uiSubYTmp - 1) * uiXRes;
	}

	checkCudaErrors(cudaMalloc((void **)&d_ctx, sizeof(clahe_ctx)));

	checkCudaErrors(cudaMemcpy(d_ctx, &ctx, sizeof(clahe_ctx), cudaMemcpyHostToDevice));

	kernel_clahe_interpolate_one<<<uiYSize, uiXSize>>>(d_ctx,uiXRes,dLUT,blkCnt);

	checkCudaErrors(cudaFree(d_ctx));

	return 0;
}

extern "C" int clahe_Interpolate_one_roi(unsigned char *pDst, unsigned char *pSrc,
		unsigned char *dLUT, unsigned int *dHist,
		unsigned int uiXRes, unsigned int uiYRes,
		int startX, int startY, int validW, int validH, unsigned int uiNrX, unsigned int uiNrY,
		unsigned int uiNrBins, cudaStream_t stream)
{
	unsigned int uiXSize = validW/uiNrX, uiYSize = validH/uiNrY;

	clahe_ctx ctx, *d_ctx;
	unsigned int uiSubYTmp;
	unsigned int uiXL, uiXR, uiYU, uiYB;
	unsigned int uiX, uiY;
	int blkCnt = 0;

	/* Interpolate greylevel mappings to get CLAHE image */
	pDst += startY*uiXRes+startX;
	pSrc += startY*uiXRes+startX;
	for (uiY = 0; uiY <= uiNrY; uiY++)
	{
		if (uiY == 0)
		{                     /* special case: top row */
			uiSubYTmp = uiYSize >> 1;  uiYU = 0; uiYB = 0;
		}
		else
		{
			if (uiY == uiNrY) {               /* special case: bottom row */
				uiSubYTmp = uiYSize >> 1;    uiYU = uiNrY-1;  uiYB = uiYU;
			}
			else
			{                     /* default values */
				uiSubYTmp = uiYSize; uiYU = uiY - 1; uiYB = uiYU + 1;
			}
		}
		for (uiX = 0; uiX <= uiNrX; uiX++)
		{
			ctx.uiSubY[blkCnt] = uiSubYTmp;
			if (uiX == 0)
			{                 /* special case: left column */
				ctx.uiSubX[blkCnt] = uiXSize >> 1; uiXL = 0; uiXR = 0;
			}
			else
			{
				if (uiX == uiNrX)
				{             /* special case: right column */
					ctx.uiSubX[blkCnt] = uiXSize >> 1;  uiXL = uiNrX - 1; uiXR = uiXL;
				}
				else
				{                     /* default values */
					ctx.uiSubX[blkCnt] = uiXSize; uiXL = uiX - 1; uiXR = uiXL + 1;
				}
			}

			ctx.pulLU[blkCnt] = dHist + (uiNrBins * (uiYU * uiNrX + uiXL));
			ctx.pulRU[blkCnt] = dHist + (uiNrBins * (uiYU * uiNrX + uiXR));
			ctx.pulLB[blkCnt] = dHist + (uiNrBins * (uiYB * uiNrX + uiXL));
			ctx.pulRB[blkCnt] = dHist + (uiNrBins * (uiYB * uiNrX + uiXR));
			ctx.blkDst[blkCnt] = pDst;
			ctx.blkSrc[blkCnt] = pSrc;
			ctx.fnum[blkCnt] = 1.0f/(ctx.uiSubX[blkCnt]*ctx.uiSubY[blkCnt]);
			//clahe_interpolate_(blkDst[blkCnt], blkSrc[blkCnt],
			//		uiXRes,pulLU[blkCnt],pulRU[blkCnt],pulLB[blkCnt],pulRB[blkCnt],
			//		uiSubX[blkCnt],uiSubY[blkCnt],dLUT,stream);
			//kernel_clahe_interpolate<<<uiYSize, uiXSize, 0, stream>>>(blkDst[blkCnt], blkSrc[blkCnt], uiXRes,
			//		pulLU[blkCnt],pulRU[blkCnt],pulLB[blkCnt],pulRB[blkCnt],
			//		uiSubX,uiSubY,dLUT, 1.0f/(uiSubX*uiSubY));
			pSrc += ctx.uiSubX[blkCnt];
			pDst += ctx.uiSubX[blkCnt];

			blkCnt ++;
		}
		pSrc	+= (uiXRes - validW);
		pDst+= (uiXRes - validW);
		pSrc += (uiSubYTmp - 1) * uiXRes;
		pDst += (uiSubYTmp - 1) * uiXRes;
	}

	checkCudaErrors(cudaMalloc((void **)&d_ctx, sizeof(clahe_ctx)));

	checkCudaErrors(cudaMemcpy(d_ctx, &ctx, sizeof(clahe_ctx), cudaMemcpyHostToDevice));

	kernel_clahe_interpolate_one<<<uiYSize, uiXSize>>>(d_ctx,uiXRes,dLUT,blkCnt);

	checkCudaErrors(cudaFree(d_ctx));

	return 0;
}




