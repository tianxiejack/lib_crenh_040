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
#define THDBLK_SIZE	(32*6)


__global__ void kernel_filter(
		unsigned char *dst,
		const unsigned char *src, unsigned char *pRef0, unsigned char *pRef1,unsigned char *pRef2,int IdxNum,
		int width, int height, int channels,
		float tmporal_strength, float Total_Frame_noise, float tmporal_trigger)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if(x >= width || y >= height)
			return;
	int src_step = width*channels;
	int pos = src_step*y+x*channels;
	int i;
	float alph0, alph,m = 0;

	for(i=0; i<channels; i++){
		m+=abs(src[pos+i]-pRef0[pos+i]);
	}
#if 0
	alph0 = (m<Total_Frame_noise)?((Total_Frame_noise-m)/Total_Frame_noise):(0.0);
#else
	alph0 = (Total_Frame_noise<tmporal_trigger)?((tmporal_trigger-Total_Frame_noise)/tmporal_trigger):(0.0);
#endif
	alph = (m < tmporal_strength)?(alph0 + (1-alph0)*m/tmporal_strength):1.0;

	for(i=0; i<channels; i++){
		dst[pos+i] = clip(0,255, (src[pos+i]*alph+(1-alph)*pRef0[pos+i]));
//		dst[pos+i] = clip(0,255, (src[pos+i]*0.25+pRef0[pos+i]*0.25+pRef1[pos+i]*0.25+pRef2[pos+i]*0.25));
	}

}

extern "C" int _tmporal_filter_(
		unsigned char *dst,
		const unsigned char *src, unsigned char **ref, int IdxNum,
		int width, int height, int channels,
		float tmporal_strength, float Total_Frame_noise,  float tmporal_trigger)
{
	dim3 block((width+31)/32,(height+31)/32);
	dim3 thread(32, 32);

	assert(IdxNum>=3);
	kernel_filter<<<block, thread>>>(dst, src, ref[0],ref[1],ref[2], IdxNum, width, height, channels, tmporal_strength, Total_Frame_noise, tmporal_trigger);

	return 0;
}

__global__ void kernel_bgr2y(
		unsigned char *dst,	const unsigned char *src, int width, int height)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if(x >= width || y >= height)
			return;

	unsigned char R, G, B;
	int src_step = width*3, dst_step = width;
	int pos = y*src_step + x*3;

	B = src[pos];	G = src[pos+1];	R = src[pos+2];
	dst[y*dst_step+x] = clip(0, 255, (0.299*R + 0.587*G + 0.114*B));
}

extern "C" int _BGR2Gray_(
		unsigned char *dst,	const unsigned char *src, int width, int height)
{
	dim3 block((width+31)/32,(height+31)/32);
	dim3 thread(32, 32);

	kernel_bgr2y<<<block, thread>>>(dst, src, width, height);

	return 0;
}

__global__ void kernel_noise_lev(
		const unsigned char *src0, const unsigned char *src1,int width, int height, int channels,
		unsigned int uiXSize, unsigned int uiYSize, unsigned int *pFrame_noise)
{
	//printf("%d x %d  %d x %d  %d , %d   %d , %d\n",
		//		gridDim.x, gridDim.y, blockDim.x, blockDim.y, blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
	unsigned int sum;
	unsigned int *noiseSub = pFrame_noise + (blockIdx.y * gridDim.x + blockIdx.x);
	int offsetBase = blockIdx.y*uiYSize*width + blockIdx.x*uiXSize;
	for(int j=threadIdx.y; j<uiYSize; j+= 32){
			int offset = offsetBase + j*width;
			for(int i=threadIdx.x; i<uiXSize; i+=32){
				int pos = (offset+i)*3;
				sum = (unsigned int)(abs(src0[pos]-src1[pos]) + abs(src0[pos+1]-src1[pos+1])  + abs(src0[pos+2]-src1[pos+2])) ;
				atomicAdd(noiseSub, sum);
		}
	}
}

extern __global__ void kernel_full_u32(
		unsigned int *mem, int size, unsigned int value);

extern "C" int _CalNoiseLevel(
		const unsigned char *src0,	const unsigned char *src1, int width, int height, int channels, unsigned int *pFrame_noise)
{
	unsigned int uiXSize = 32, uiYSize = 32;
	unsigned int uiNrX = width/uiXSize, uiNrY = height/uiYSize;

	dim3 block(uiNrX, uiNrY);
	dim3 thread(32,32);

	kernel_full_u32<<<uiNrX*uiNrY/THDBLK_SIZE, THDBLK_SIZE>>>( pFrame_noise, uiNrX*uiNrX, 0 );

	kernel_noise_lev<<<block, thread>>>(src0, src1, width, height, channels, uiXSize, uiYSize, pFrame_noise);

	return 0;
}

