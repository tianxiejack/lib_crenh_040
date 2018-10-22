/*
 * cuProcess.cpp
 *
 *  Created on: May 11, 2017
 *      Author: ubuntu
 */

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv/cv.hpp>
#include <opencv2/opencv.hpp>
#include "cuProcess.hpp"
#include "cuda_mem.cpp"
#include "HIST/histogram_common.h"
#include "ENHANCE/EnhanceImage.h"
#include "math.h"
#include "helper_cuda.h"
#include "clahe.h"

using namespace std;
using namespace cv;

static cudaError_t et;
static CCudaProcess proc;

extern "C"
void updateGaussian(float delta, int radius, float *fGaussian);
extern "C"
void updateEuclidean( float e_d, float *fEuclidean, int Num);

CCudaProcess::CCudaProcess() : m_bUnhazed(false),m_curWidth(0), m_curHeight(0),m_bFltInit(false),m_Index(0)
{
	int i;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	for(i=0; i<CUSTREAM_CNT; i++){
		et = cudaStreamCreate(&m_cuStream[i]);
		OSA_assert(et == cudaSuccess);
	}
	initHistogram256();
	cudaMalloc((void **)&d_Histogram, HISTOGRAM256_BIN_COUNT * sizeof(uint));

	gaussian_delta = 10.f;
	euclidean_delta = 10.f;
	filter_radius = 4;
	 float  fGaussian[64], fEuclidean[768];
	 checkCudaErrors(cudaMalloc((void **)&cGaussian, 64 * sizeof(float)));
	 checkCudaErrors(cudaMalloc((void **)&cEuclidean, 768 * sizeof(float)));
	 updateGaussian(gaussian_delta, filter_radius, fGaussian);
	 updateEuclidean(euclidean_delta, fEuclidean, 768);
	 checkCudaErrors(cudaMemcpy(cGaussian, fGaussian, sizeof(float)*(2*filter_radius+1), cudaMemcpyHostToDevice));
	 checkCudaErrors(cudaMemcpy(cEuclidean, fEuclidean, sizeof(float)*768, cudaMemcpyHostToDevice));

	 noise_IIR_coefficent = 0.5;
	 spatial_strength = 2.0;
	 max_noise = 30.0;
	 Frame_noise_previous = 0.0;
	 temporal_strenght = 50;
	 temporal_trigger_noise = 30;

	 int histSize= uiNR_OF_GREY*32*32*sizeof(int);
	 checkCudaErrors(cudaMalloc((void **)&m_pDeviceBuf, histSize * sizeof(unsigned char)));
}

CCudaProcess::~CCudaProcess()
{
	int i;

	cudaEventDestroy(	start	);
	cudaEventDestroy(	stop	);

	for(i=0; i<CUSTREAM_CNT; i++){
		if(m_cuStream[i] != NULL){
			et = cudaStreamDestroy(m_cuStream[i]);
			OSA_assert(et == cudaSuccess);
			m_cuStream[i] = NULL;
		}
	}
	unhazed_uninit();

	filter_uninit();

	for(i=0; i<SHAREARRAY_CNT; i++)
		cudaFree_share(NULL, i);
	cudaFree(d_Histogram);
	closeHistogram256();
	cudaFree(cGaussian);
	cudaFree(cEuclidean);
	cudaFree(m_pDeviceBuf);
}

unsigned char* CCudaProcess::load(cv::Mat frame, int memChn)
{
	int i;
	unsigned int byteCount = frame.rows * frame.cols * frame.channels() * sizeof(unsigned char);
	unsigned int byteBlock = byteCount/CUSTREAM_CNT;
	unsigned char *d_mem = NULL;

	et = cudaMalloc_share((void**)&d_mem, byteCount, memChn);
	OSA_assert(et == cudaSuccess);

	for(i = 0; i<CUSTREAM_CNT; i++){
		et = cudaMemcpyAsync(d_mem + byteBlock*i, frame.data + byteBlock*i, byteBlock, cudaMemcpyHostToDevice, m_cuStream[i]);
		OSA_assert(et == cudaSuccess);
	}

	et = cudaFree_share(d_mem, memChn);
	OSA_assert(et == cudaSuccess);

	return d_mem;
}

cv::Mat CCudaProcess::dmat_create(int cols, int rows, int type, int memChn)
{
	unsigned char *d_mem = NULL;
	unsigned int byteCount = rows * cols * CV_ELEM_SIZE(type);
	byteCount = (byteCount+3) & (~3);
	et = cudaMalloc_share((void**)&d_mem, byteCount, memChn);
	OSA_assert(et == cudaSuccess);

	cv::Mat rm = cv::Mat(rows, cols, type, d_mem);

	et = cudaFree_share(d_mem, memChn);
	OSA_assert(et == cudaSuccess);

	return rm;
}

void CCudaProcess::dmat_delete(cv::Mat dm, int memChn)
{
	et = cudaFree_share(dm.data, memChn);
	OSA_assert(et == cudaSuccess);
}

#if 1
extern "C" int yuyv2gray_(
	unsigned char *dst, const unsigned char *src,
	int width, int height, cudaStream_t stream);
extern "C" int uyvy2gray_(
	unsigned char *dst, const unsigned char *src,
	int width, int height, cudaStream_t stream);
void CCudaProcess::cutColor(cv::Mat src, cv::Mat &dst, int code)
{
	int i, chId = 0;
	unsigned char *d_src = load(src, chId);
	unsigned char *d_dst = NULL;
	unsigned int byteCount_src = src.rows * src.cols * src.channels() * sizeof(unsigned char);
	unsigned int byteBlock_src = byteCount_src/CUSTREAM_CNT;
	unsigned int byteCount_dst;
	unsigned int byteBlock_dst;

	if(code == CV_YUV2GRAY_YUYV)
	{
		//dst = Mat(src.rows, src.cols, CV_8UC1);
		byteCount_dst = dst.rows * dst.cols * sizeof(unsigned char);
		byteBlock_dst = byteCount_dst/CUSTREAM_CNT;
		cudaMalloc_share((void**)&d_dst, byteCount_dst, chId+1);

		for(i = 0; i<CUSTREAM_CNT; i++){
			yuyv2gray_(d_dst + byteBlock_dst*i,
					d_src + byteBlock_src*i,
					src.cols, (src.rows/CUSTREAM_CNT), m_cuStream[i]);
			cudaMemcpyAsync(dst.data + byteBlock_dst*i, d_dst + byteBlock_dst*i, byteBlock_dst, cudaMemcpyDeviceToHost, m_cuStream[i]);
		}
		cudaFree_share(d_dst, chId+1);
		for(i=0; i<CUSTREAM_CNT; i++)
			cudaStreamSynchronize(m_cuStream[i]);
	}

	if(code == CV_YUV2GRAY_UYVY)
	{
		//dst = Mat(src.rows, src.cols, CV_8UC1);
		byteCount_dst = dst.rows * dst.cols * sizeof(unsigned char);
		byteBlock_dst = byteCount_dst/CUSTREAM_CNT;
		cudaMalloc_share((void**)&d_dst, byteCount_dst, 1);

		for(i = 0; i<CUSTREAM_CNT; i++){
			uyvy2gray_(d_dst + byteBlock_dst*i,
					d_src + byteBlock_src*i,
					src.cols, (src.rows/CUSTREAM_CNT), m_cuStream[i]);
			cudaMemcpyAsync(dst.data + byteBlock_dst*i, d_dst + byteBlock_dst*i, byteBlock_dst, cudaMemcpyDeviceToHost, m_cuStream[i]);
		}
		for(i=0; i<CUSTREAM_CNT; i++)
			cudaStreamSynchronize(m_cuStream[i]);
		cudaFree_share(d_dst, 1);
	}
}

extern "C" int yuyv2bgr_(
	unsigned char *dst, const unsigned char *src,
	int width, int height, cudaStream_t stream);
void CCudaProcess::YUVV2RGB(cv::Mat src, cv::Mat &dst, int code)
{
/*	cudaEventRecord	(	start,	0);*/

	int i,chId = 2;
	unsigned char *d_src = load(src,chId);
	unsigned char *d_dst = NULL;
	unsigned int byteCount_src = src.rows * src.cols * src.channels() * sizeof(unsigned char);;
	unsigned int byteBlock_src = byteCount_src/CUSTREAM_CNT;
	unsigned int byteCount_dst;
	unsigned int byteBlock_dst;

	OSA_assert(code == CV_YUV2BGR_YUYV);
	{
		//dst = Mat(src.rows, src.cols, CV_8UC1);
		byteCount_dst = dst.rows * dst.cols *3* sizeof(unsigned char);
		byteBlock_dst = byteCount_dst/CUSTREAM_CNT;
		cudaMalloc_share((void**)&d_dst, byteCount_dst, chId+1);

		for(i = 0; i<CUSTREAM_CNT; i++){
			yuyv2bgr_(d_dst + byteBlock_dst*i,
					d_src + byteBlock_src*i,
					src.cols, (src.rows/CUSTREAM_CNT), m_cuStream[i]);
		}

//		for(i=0; i<CUSTREAM_CNT; i++){
//			cudaMemcpy(dst.data + byteBlock_dst*i, d_dst + byteBlock_dst*i, byteBlock_dst, cudaMemcpyDeviceToHost);
//		}
		cudaMemcpy(dst.data, d_dst, byteCount_dst, cudaMemcpyDeviceToHost);
		cudaFree_share(d_dst, chId+1);
	}

/*		cudaEventRecord(	stop,	0	);
		cudaEventSynchronize(	stop);
		cudaEventElapsedTime(	&elapsedTime,	start,	stop);
		printf("YUVV2RGB:	%.8f	ms \n", elapsedTime);*/
}

extern "C" int yuyv2yuvplan_(
	unsigned char *dsty, unsigned char *dstu,unsigned char *dstv,const unsigned char *src,
	int width, int height, cudaStream_t stream);
void CCudaProcess::YUVV2YUVPlan(cv::Mat src, cv::Mat &dst)
{
/*	cudaEventRecord	(	start,	0);*/

	int i,chId = 4;
	unsigned char *d_src = load(src,chId);
	unsigned char *d_dst = NULL;
	unsigned int imagesize = src.rows*src.cols;
	unsigned int byteCount_src = src.rows * src.cols * src.channels() * sizeof(unsigned char);
	unsigned int byteBlock_src = byteCount_src/CUSTREAM_CNT;
	unsigned int byteCount_dst;
	unsigned int byteBlock_dst;

	{
		//dst = Mat(src.rows, src.cols, CV_8UC1);
		byteCount_dst = src.rows * src.cols * src.channels()* sizeof(unsigned char);
		byteBlock_dst = byteCount_dst/CUSTREAM_CNT;
		cudaMalloc_share((void**)&d_dst, byteCount_dst, chId+1);

		for(i = 0; i<CUSTREAM_CNT; i++){
			yuyv2yuvplan_(	d_dst + (byteBlock_dst>>1)*i,	d_dst + imagesize + (byteBlock_dst>>2)*i, 	d_dst + imagesize + (imagesize>>1) + (byteBlock_dst>>2)*i,
											d_src + byteBlock_src*i,
											src.cols, (src.rows/CUSTREAM_CNT), m_cuStream[i]);
		}

//		for(i=0; i<CUSTREAM_CNT; i++){
//			cudaMemcpy(dst.data + byteBlock_dst*i, d_dst + byteBlock_dst*i, byteBlock_dst, cudaMemcpyDeviceToHost);
//		}

		dst = cv::Mat(src.rows, src.cols, CV_8UC2, d_dst);

		cudaFree_share(d_dst, chId+1);
	}

/*		cudaEventRecord(	stop,	0	);
		cudaEventSynchronize(	stop);
		cudaEventElapsedTime(	&elapsedTime,	start,	stop);
		printf("YUVV2RGB:	%.8f	ms \n", elapsedTime);*/
}

void cutColor(Mat src, Mat &dst, int code)
{
	proc.cutColor(src, dst, code);
}
void cvtBigVideo(cv::Mat src, cv::Mat &dst, int type)
{
	proc.YUVV2RGB(src, dst, type);
}

void cvtBigVideo_plan(cv::Mat src,cv::Mat &dst)
{
	proc.YUVV2YUVPlan(src, dst);
}

#endif

//Internal memory allocation
extern "C" void initHistogram256(void);

//Internal memory deallocation
extern "C" void closeHistogram256(void);

extern "C" void histogram256(
    uint *d_Histogram,
    void *d_Data,
    uint byteCount
);

extern "C" int rgbHist_(
		unsigned char *dst,const unsigned char *src,uint  *d_Histogram,
	int width, int height);
extern "C" int grayHist_(
		unsigned char *dst,const unsigned char *src,uint  *d_Histogram,
	int width, int height);

void CCudaProcess::cuHistEnh(cv::Mat src, cv::Mat dst)
{
/*	cudaEventRecord	(	start,	0);*/

	histogram256(d_Histogram, (void *)src.data, src.rows*src.cols*src.channels());

	uint Histogram[HISTOGRAM256_BIN_COUNT];

	 et = cudaMemcpy(Histogram, d_Histogram, HISTOGRAM256_BIN_COUNT*sizeof(uint), cudaMemcpyDeviceToHost);
	 OSA_assert(et == cudaSuccess);

	int k, pixelsum = 0, sum = 0;
	for(k=0; k<HISTOGRAM256_BIN_COUNT; k++)
	{
		//Histogram[k] = (uint)(10.0*log((double)Histogram[k]+1));
		pixelsum += Histogram[k] ;
	}
	for(k=0; k<HISTOGRAM256_BIN_COUNT; k++)
	{
		sum += Histogram[k];
		Histogram[k] = (uint)( (255.0 * sum)/(pixelsum - Histogram[k])+0.5f );
	}

	et = cudaMemcpy(d_Histogram, Histogram, HISTOGRAM256_BIN_COUNT*sizeof(uint), cudaMemcpyHostToDevice);
	OSA_assert(et == cudaSuccess);

	if(src.channels() == 3)
		rgbHist_(dst.data, src.data, d_Histogram, src.cols, src.rows);
	else if(src.channels() == 1)
		grayHist_(dst.data,src.data, d_Histogram, src.cols, src.rows);

/*
	cudaEventRecord(	stop,	0	);
	cudaEventSynchronize(	stop);
	cudaEventElapsedTime(	&elapsedTime,	start,	stop);
	printf("cuHistEn:	%.8f	ms \n", elapsedTime);*/
}

static void _calStrech(unsigned char lut[], cv::Point2i lowY, cv::Point2i upY)
{
	int i;
	float	k;
	k = 1.0*lowY.y/lowY.x;
	for(i=0; i<lowY.x; i++){
		lut[i] = (unsigned char)(i*k);
	}
	k = 1.0*(lowY.y-upY.y)/(lowY.x-upY.x);
	for(; i<upY.x; i++){
		lut[i] = (unsigned char)((i-lowY.x+1)*k+lut[lowY.x-1]);
	}
	k = 1.0*(255-upY.y)/(255-upY.x);
	for(; i<256; i++){
		lut[i] = (unsigned char)((i-upY.x+1)*k+lut[upY.x-1]);
	}
}

int CCudaProcess::unhazed_init(int width, int height, int blockSize, float gamma)
{
	if(m_bUnhazed && (m_curWidth != width || m_curHeight != height))
		unhazed_uninit();

	if(!m_bUnhazed){
		unsigned char lut[256];
		unsigned char luts[256];
		cv::Point2i lowY, upY;
		lowY = cv::Point2i(20, 20);
		upY = cv::Point2i(200, 200);

		_calStrech(luts, lowY, upY);

		for(int i=0; i<256; i++)
		{
			lut[i] = luts[(unsigned char)(pow((float)i/255.f,gamma)*255.0f)];
		}

		checkCudaErrors(cudaMalloc((void **)&d_lut, 256 * sizeof(unsigned char)));

		et = cudaMemcpy(d_lut, lut, 256*sizeof(unsigned char), cudaMemcpyHostToDevice);
		OSA_assert(et == cudaSuccess);

		initDCM(width, height, blockSize);

		int dw = (width>>1), dh = (height>>1);
		m_tran = dmat_create(dw, dh, CV_32FC1, SHAREARRAY_CNT-1);
		m_norgray = dmat_create(dw, dh, CV_32FC1, SHAREARRAY_CNT-2);
		m_guideImg = dmat_create(dw, dh, CV_32FC1, SHAREARRAY_CNT-3);

		full_f32_((float*)m_guideImg.data, dw*dh, 0.78);

		m_curWidth = width;  m_curHeight = height;

		m_bUnhazed = true;
	}

	return 0;
}

void CCudaProcess::unhazed_uninit()
{
	if(m_bUnhazed)
	{
		unInitDCM();
		cudaFree(d_lut);

		dmat_delete(m_tran, SHAREARRAY_CNT-1);
		dmat_delete(m_norgray, SHAREARRAY_CNT-2);
		dmat_delete(m_guideImg, SHAREARRAY_CNT-3);

		m_bUnhazed = false;
	}
}

extern "C" int full_u8_(unsigned char *mem, int size, unsigned char value);

void CCudaProcess::filter_init(cv::Mat src)
{
	int width, height;
	unsigned int uiNrX, uiNrY;
	width = src.cols;
	height = src.rows;
	uiNrX= width/32;
	uiNrY = height/32;
	if(m_bFltInit && (m_curWidth != width || m_curHeight != height)){
		filter_uninit();
	}
	if(!m_bFltInit){
		for(int i=0; i<FILTER_REF_NUM; i++){
			m_filterRef[i] = dmat_create(width, height, /*CV_8UC3*/src.type(), SHAREARRAY_CNT-5-i);
			full_u8_((unsigned char*)m_filterRef[i].data, width*height* CV_ELEM_SIZE(src.type()), 0);
		}
		m_YFrame = dmat_create(width, height, CV_8UC1, SHAREARRAY_CNT-5-FILTER_REF_NUM);
		m_NoiseMat = dmat_create(uiNrX, uiNrY, CV_32SC1, SHAREARRAY_CNT-5-FILTER_REF_NUM-1);
		spaceFiltMat = dmat_create(width, height, src.type(), SHAREARRAY_CNT-5-FILTER_REF_NUM-2);

		m_curWidth = width;  m_curHeight = height;
		m_bFltInit = true;
		m_Index = 0;
	}
}

void CCudaProcess::filter_uninit()
{
	if(m_bFltInit){
		for(int i=0; i<FILTER_REF_NUM; i++){
			dmat_delete(m_filterRef[i], SHAREARRAY_CNT-5-i);
		}
		dmat_delete(m_YFrame, SHAREARRAY_CNT-5-FILTER_REF_NUM);
		dmat_delete(m_NoiseMat, SHAREARRAY_CNT-5-FILTER_REF_NUM-1);
		dmat_delete(spaceFiltMat, SHAREARRAY_CNT-5-FILTER_REF_NUM-2);
		m_bFltInit = false;
		m_Index = 0;
	}
}

static unsigned int		frameCount;// = 0;
static float			totalTS;// = 0.f;

void CCudaProcess::cuTmporalFilt(cv::Mat src, cv::Mat dst)
{
	unsigned int byteCount;
	cudaEventRecord	(	start,	0);

	filter_init(src);

	_spaceFilter(src, spaceFiltMat);
	_temporalFilter(spaceFiltMat, dst, m_Index);

	m_Index = (m_Index+1) > (FILTER_REF_NUM-1) ? 0 : (m_Index+1);
	byteCount = src.rows * src.cols * src.channels() * sizeof(unsigned char);
	et = cudaMemcpy(m_filterRef[m_Index].data, dst.data, byteCount, cudaMemcpyDeviceToDevice);
	OSA_assert(et == cudaSuccess);

///*
		cudaEventRecord(	stop,	0	);
		cudaEventSynchronize(	stop);
		cudaEventElapsedTime(	&elapsedTime,	start,	stop);

		frameCount++;
		totalTS += elapsedTime;
		if(frameCount == 100){
			printf("cuTmporalFilt:	%.8f	ms \n", totalTS/100.f);
			frameCount = 0;
			totalTS = 0.f;
		}
//		printf("cuTmporalFilt:	%.8f	ms \n", elapsedTime);
//*/
}

extern "C" int _tmporal_filter_(
		unsigned char *dst,
		const unsigned char *src, unsigned char **ref, int IdxNum,
		int width, int height, int channels,
		float tmporal_strength, float Total_Frame_noise,  float tmporal_trigger);
void CCudaProcess::_temporalFilter(cv::Mat src, cv::Mat dst, int index)
{
	int i, channels;
	unsigned char *psrc, *pdst, *pref[FILTER_REF_NUM];
	psrc = (unsigned char *)src.data;
	pdst = (unsigned char *)dst.data;
	channels = src.channels();
	for(i=0; i<FILTER_REF_NUM; i++){
		pref[i] =  (unsigned char *)m_filterRef[index].data;
		index = (index-1) < 0 ? (FILTER_REF_NUM-1) : (index-1);
	}
	_tmporal_filter_(pdst, psrc, pref, FILTER_REF_NUM, src.cols, src.rows, channels, temporal_strenght, Totla_Frame_noise, temporal_trigger_noise);
}

extern "C"
int bilateralFilterRGB0(uchar *dDest, uchar *dSrc,
                           int width, int height, int channels,
                           float e_d, float *cGaussian, int radius);
extern "C"
int bilateralFilterRGB(uchar *dDest, uchar *dSrc, uchar *dGray,
                           int width, int height, int channels,
                           float e_d, float *cGaussian,  float *cEuclidean, int radius);
extern "C"
int bilateralFilterRGB2(uchar *dDest, uchar *dSrc,
                           int width, int height, int channels,
                           float e_d, float *cGaussian, int radius);
extern "C"
int _BGR2Gray_(
		unsigned char *dst,	const unsigned char *src, int width, int height);
extern "C" int _CalNoiseLevel(
		const unsigned char *src0,	const unsigned char *src1, int width, int height, int channels, unsigned int *pFrame_noise);
void CCudaProcess::_spaceFilter(cv::Mat src, cv::Mat dst)
{
	int i, j, channels, k;
	unsigned char *psrc, *pdst, *pgray, *pref;
	psrc = (unsigned char *)src.data;
	pdst = (unsigned char *)dst.data;
	pgray = (unsigned char *)m_YFrame.data;
	pref = (unsigned char*)m_filterRef[m_Index].data;
	channels = src.channels();
	assert(channels == 3);

	_CalNoiseLevel(psrc, pref, src.cols, src.rows, channels, (unsigned int*)m_NoiseMat.data);
	unsigned int uiNrX, uiNrY;
	float aveNoise, sumNoise;
	uiNrX= src.cols/32;	uiNrY = src.rows/32;
	unsigned int noiseArray[uiNrX*uiNrY];
	et = cudaMemcpy(noiseArray, m_NoiseMat.data, uiNrX*uiNrY*sizeof(unsigned int), cudaMemcpyDeviceToHost);
	OSA_assert(et == cudaSuccess);

	k = 0;
	sumNoise = 0.0;
	for(j=1; j<uiNrY-1; j++){
		for(i=1; i<uiNrX-1; i++){
			aveNoise = (float)noiseArray[j*uiNrX+i]/(32*32);
			if(aveNoise <max_noise){
				sumNoise += aveNoise;
				k++;
			}
		}
	}
	Frame_noise = (k>0)?(sumNoise/k):0.0;

//	printf("%s: Frame_noise=%.2f \n",__func__, Frame_noise);

	Frame_noise_filtered = Frame_noise_previous * noise_IIR_coefficent + Frame_noise *(1 - noise_IIR_coefficent);
	Threshold = spatial_strength * Frame_noise_filtered;
	Totla_Frame_noise = Frame_noise_filtered+0.01;

	Frame_noise_previous = Frame_noise_filtered;

#if 0
	_BGR2Gray_(pgray, psrc, src.cols, src.rows);
	bilateralFilterRGB(pdst, psrc, pgray, src.cols, src.rows, channels, euclidean_delta, cGaussian, cEuclidean, filter_radius);
#else
//	bilateralFilterRGB0(pdst, psrc, src.cols, src.rows, channels, euclidean_delta, cGaussian,  filter_radius);
	bilateralFilterRGB2(pdst, psrc, src.cols, src.rows, channels, euclidean_delta, cGaussian,  filter_radius);
#endif
}

int CCudaProcess::getAtmosphricLight_rgb32(cv::Mat hazeImg, cv::Mat darkChImg)
{
	histogram256(d_Histogram, (void *)darkChImg.data, darkChImg.rows*darkChImg.cols*darkChImg.channels());

	uint Histogram[HISTOGRAM256_BIN_COUNT];

	et = cudaMemcpy(Histogram, d_Histogram, HISTOGRAM256_BIN_COUNT*sizeof(uint), cudaMemcpyDeviceToHost);
	OSA_assert(et == cudaSuccess);

	int graylevel = 0;
	int cols = darkChImg.cols;
	int rows = darkChImg.rows;

	float darkHistogram[GRAYLEVEL];
	float tmpHistogram[GRAYLEVEL];
	memset(darkHistogram, 0, sizeof(darkHistogram));
	memset(tmpHistogram, 0, sizeof(tmpHistogram));

	for(int i=0; i<256; i++){
		darkHistogram[i] = Histogram[i]*1.0/(rows*cols);
	}

	float ligth_ratio = 1 - LIGHTRANGE;         //0.999

	for(int i=0; i<GRAYLEVEL; i++){
		for(int j=0; j<=i; j++){
			tmpHistogram[i] += darkHistogram[j];

			if(tmpHistogram[i] > ligth_ratio)
			{
				graylevel = i;
				i = 256;
				break;
			}
		}
	}

	return graylevel;
}

int CCudaProcess::getAtmosphricLight_gray(cv::Mat hazeImg, cv::Mat darkChImg)
{
	histogram256(d_Histogram, (void *)darkChImg.data, darkChImg.rows*darkChImg.cols*darkChImg.channels());

	uint Histogram[HISTOGRAM256_BIN_COUNT];

	et = cudaMemcpy(Histogram, d_Histogram, HISTOGRAM256_BIN_COUNT*sizeof(uint), cudaMemcpyDeviceToHost);
	OSA_assert(et == cudaSuccess);

	int graylevel = 0;
	int cols = darkChImg.cols;
	int rows = darkChImg.rows;

	float darkHistogram[GRAYLEVEL];
	float tmpHistogram[GRAYLEVEL];
	memset(darkHistogram, 0, sizeof(darkHistogram));
	memset(tmpHistogram, 0, sizeof(tmpHistogram));

	for(int i=0; i<256; i++){
		darkHistogram[i] = Histogram[i]*1.0/(rows*cols);
	}

	float ligth_ratio = 1 - LIGHTRANGE;         //0.999

	for(int i=0; i<GRAYLEVEL; i++){
		for(int j=0; j<=i; j++){
			tmpHistogram[i] += darkHistogram[j];

			if(tmpHistogram[i] > ligth_ratio)
			{
				graylevel = i;
				i = 256;
				break;
			}
		}
	}

	return graylevel;
}

void CCudaProcess::unhazed(cv::Mat inImg, cv::Mat outImg)
{
	static unsigned char A  = TESTAC;
	float gamma = GAMMAVALUE;

	int dw = (inImg.cols>>1), dh = (inImg.rows>>1);
	OSA_assert(inImg.cols == outImg.cols);
	OSA_assert(inImg.rows == outImg.rows);
	OSA_assert(inImg.channels() == outImg.channels());

	unhazed_init(inImg.cols, inImg.rows, BLOCKSIZE, gamma);

	cv::Mat darkImg = dmat_create(dw, dh, CV_8UC1, SHAREARRAY_CNT-4);

	if(inImg.channels() == 3){

		darkChannel_rgb32_((float*)m_tran.data, (float*)m_norgray.data, darkImg.data, inImg.data, dw, dh, (float)A, WEIGHT);
		//norgray2rgb32_(outImg.data, (float*)tran.data, dw, dh);
		//unhazed_rgb32_(outImg.data, inImg.data, (float*)tran.data, d_lut, A, dw, dh);
		guideFilter_((float*)m_guideImg.data, (float*)m_tran.data, (float*)m_norgray.data, dw, dh, m_cuStream);
		//norgray2rgb32_(outImg.data, (float*)m_guideImg.data, dw, dh);

		//int leve = getAtmosphricLight_rgb32(inImg, darkImg);

		unhazed_rgb32_(outImg.data, inImg.data, (float*)m_guideImg.data, d_lut, A, dw, dh);

		//A = min((leve+5), 255);
		//A = (A > TESTAC) ? A : TESTAC;

	}else if(inImg.channels() == 1){

		darkChannel_gray_((float*)m_tran.data, (float*)m_norgray.data, darkImg.data, inImg.data, dw, dh, (float)A, WEIGHT);
		//norgray2gray_(outImg.data, (float*)m_tran.data, dw, dh);

		guideFilter_((float*)m_guideImg.data, (float*)m_tran.data, (float*)m_norgray.data, dw, dh, m_cuStream);
		//norgray2gray_(outImg.data, (float*)m_guideImg.data, dw, dh);

		//int leve = getAtmosphricLight_gray(inImg, darkImg);

		unhazed_gray_(outImg.data, inImg.data, (float*)m_guideImg.data, d_lut, A, dw, dh);

		//A = min((leve+5), 255);
		//A = (A > TESTAC) ? A : TESTAC;
	}

	dmat_delete(darkImg, SHAREARRAY_CNT-4);

}

void CCudaProcess::cuUnhazed(cv::Mat src, cv::Mat dst)
{
	//cudaEventRecord	(	start,	0);

	unhazed(src, dst);

	//cudaEventRecord(	stop,	0	);
	//cudaEventSynchronize(	stop);
	//cudaEventElapsedTime(	&elapsedTime,	start,	stop);
	//printf("cuHistEn:	%.8f	ms \n", elapsedTime);
}

static void MakeLut (kz_pixel_t * pLUT, kz_pixel_t Min, kz_pixel_t Max, unsigned int uiNrBins)
{
    int i;
    const kz_pixel_t BinSize = (kz_pixel_t) (1 + (Max - Min) / uiNrBins);

    for (i = Min; i <= Max; i++)
        pLUT[i] = (i - Min) / BinSize;
}

static void ClipHistogram (unsigned int* pulHistogram, unsigned int
                    uiNrGreylevels, unsigned int ulClipLimit)
{
    unsigned int* pulBinPointer, *pulEndPointer, *pulHisto;
    unsigned int ulNrExcess, ulOldNrExcess, ulUpper, ulBinIncr, ulStepSize, i;
    int lBinExcess;

    ulNrExcess = 0;  pulBinPointer = pulHistogram;
    for (i = 0; i < uiNrGreylevels; i++)
    { /* calculate total number of excess pixels */
        lBinExcess = (int) pulBinPointer[i] - (int) ulClipLimit;
        if (lBinExcess > 0) ulNrExcess += lBinExcess;      /* excess in current bin */
    };

    /* Second part: clip histogram and redistribute excess pixels in each bin */
    ulBinIncr = ulNrExcess / uiNrGreylevels;          /* average binincrement */
    ulUpper =  ulClipLimit - ulBinIncr;  /* Bins larger than ulUpper set to cliplimit */

    for (i = 0; i < uiNrGreylevels; i++)
    {
        if (pulHistogram[i] > ulClipLimit)
            pulHistogram[i] = ulClipLimit; /* clip bin */
        else
        {
            if (pulHistogram[i] > ulUpper)
            {       /* high bin count */
 //               ulNrExcess -= (pulHistogram[i] - ulUpper); pulHistogram[i]=ulClipLimit;
				ulNrExcess -= (ulClipLimit -pulHistogram[i]); pulHistogram[i]=ulClipLimit;
            }
            else
            {                   /* low bin count */
                ulNrExcess -= ulBinIncr; pulHistogram[i] += ulBinIncr;
            }
        }
    }

    do {   /* Redistribute remaining excess  */
        pulEndPointer = &pulHistogram[uiNrGreylevels]; pulHisto = pulHistogram;

        ulOldNrExcess = ulNrExcess;     /* Store number of excess pixels for test later. */

        while (ulNrExcess && pulHisto < pulEndPointer)
        {
            ulStepSize = uiNrGreylevels / ulNrExcess;
            if (ulStepSize < 1)
                ulStepSize = 1;       /* stepsize at least 1 */
            for (pulBinPointer=pulHisto; pulBinPointer < pulEndPointer && ulNrExcess; pulBinPointer += ulStepSize)
            {
                if (*pulBinPointer < ulClipLimit)
                {
                    (*pulBinPointer)++;  ulNrExcess--;    /* reduce excess */
                }
            }
            pulHisto++;       /* restart redistributing on other bin location */
        }
    } while ((ulNrExcess) && (ulNrExcess < ulOldNrExcess));
    /* Finish loop when we have no more pixels or we can't redistribute any more pixels */
}

void MapHistogram (unsigned int* pulHistogram, kz_pixel_t Min, kz_pixel_t Max,
                   unsigned int uiNrGreylevels, unsigned int ulNrOfPixels)
{
    unsigned int i;  unsigned int ulSum = 0;
    const float fScale = ((float)(Max - Min)) / ulNrOfPixels;
    const unsigned int ulMin = (unsigned int) Min;

    for (i = 0; i < uiNrGreylevels; i++)
    {
        ulSum += pulHistogram[i];
        pulHistogram[i]=(unsigned int)(ulMin+ulSum*fScale);
        if (pulHistogram[i] > Max)
            pulHistogram[i] = Max;
    }
}

void CCudaProcess::cuClahe(cv::Mat src, cv::Mat dst, unsigned int uiNrX, unsigned int uiNrY, float fCliplimit, int procType)
{
	//cudaEventRecord	(	start,	0);

	//float fCliplimit = 10.0;
	kz_pixel_t Min = 0;
	kz_pixel_t Max = 255;

	unsigned int uiNrBins = 256;
	//unsigned int uiNrX = 8, uiNrY = 8;
	unsigned int uiXRes = src.cols*src.channels();
	unsigned int uiYRes = src.rows;
	unsigned int uiXSize = uiXRes/uiNrX, uiYSize = uiYRes/uiNrY;

    unsigned char aLUT[uiNrBins];
    MakeLut(aLUT, Min, Max, uiNrBins);
	unsigned char *dLUT;
#if 0
	checkCudaErrors(cudaMalloc((void **)&dLUT, uiNrBins));
#else
	dLUT = m_pDeviceBuf;
#endif
	checkCudaErrors(cudaMemcpy(dLUT, aLUT, uiNrBins, cudaMemcpyHostToDevice));

	unsigned int pulMapArray[uiNrX*uiNrY*uiNrBins];
	unsigned int histSize = uiNrX*uiNrY*uiNrBins*sizeof(unsigned int);
	unsigned int *dHist;
#if 0
	checkCudaErrors(cudaMalloc((void **)&dHist, histSize));
#else
	dHist = (unsigned int*)(m_pDeviceBuf+uiNrBins);
#endif
	clahe_hist_(dHist, dLUT, src.data, src.cols*src.channels(), src.rows, uiNrX, uiNrY, m_cuStream);

	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(pulMapArray, dHist, histSize, cudaMemcpyDeviceToHost));

	unsigned int ulClipLimit;
    if(fCliplimit > 0.0) {
        ulClipLimit = (unsigned int) (fCliplimit * (uiXSize * uiYSize) / uiNrBins);
        ulClipLimit = (ulClipLimit < 1UL) ? 1UL : ulClipLimit;
    }
    else ulClipLimit = 1UL<<14;

    unsigned int ulNrPixels = (unsigned int)uiXSize * (unsigned int)uiYSize;
	unsigned int uiX, uiY;
	unsigned int *pulHist;
    for (uiY = 0; uiY < uiNrY; uiY++)
    {
        for (uiX = 0; uiX < uiNrX; uiX++)
        {
            pulHist = &pulMapArray[uiNrBins * (uiY * uiNrX + uiX)];
            ClipHistogram(pulHist, uiNrBins, ulClipLimit);
            MapHistogram(pulHist, Min, Max, uiNrBins, ulNrPixels);
        }
    }

	checkCudaErrors(cudaMemcpy(dHist, pulMapArray, histSize, cudaMemcpyHostToDevice));
	//clahe_interpolate_(src.data, src.cols*src.channels(), src.rows, dHist, );

	//cudaEventRecord(	stop,	0	);
	//cudaEventSynchronize(	stop);
	//cudaEventElapsedTime(	&elapsedTime,	start,	stop);
	//printf("cuHistEn 0:	%.8f	ms \n", elapsedTime);

	if (procType == 0)
	{

		unsigned int uiSubX, uiSubY;
		unsigned int uiXL, uiXR, uiYU, uiYB;
		unsigned int* pulLU, *pulLB, *pulRU, *pulRB;
		kz_pixel_t* pSrc, *pDst;
		int iStream = 0;

		/* Interpolate greylevel mappings to get CLAHE image */
		for (pSrc = src.data, pDst = dst.data, uiY = 0; uiY <= uiNrY; uiY++)
		{
			if (uiY == 0)
			{                     /* special case: top row */
				uiSubY = uiYSize >> 1;  uiYU = 0; uiYB = 0;
			}
			else
			{
				if (uiY == uiNrY) {               /* special case: bottom row */
					uiSubY = uiYSize >> 1;    uiYU = uiNrY-1;  uiYB = uiYU;
				}
				else
				{                     /* default values */
					uiSubY = uiYSize; uiYU = uiY - 1; uiYB = uiYU + 1;
				}
			}
			for (uiX = 0; uiX <= uiNrX; uiX++)
			{
				if (uiX == 0)
				{                 /* special case: left column */
					uiSubX = uiXSize >> 1; uiXL = 0; uiXR = 0;
				}
				else
				{
					if (uiX == uiNrX)
					{             /* special case: right column */
						uiSubX = uiXSize >> 1;  uiXL = uiNrX - 1; uiXR = uiXL;
					}
					else
					{                     /* default values */
						uiSubX = uiXSize; uiXL = uiX - 1; uiXR = uiXL + 1;
					}
				}

				pulLU = dHist + (uiNrBins * (uiYU * uiNrX + uiXL));
				pulRU = dHist + (uiNrBins * (uiYU * uiNrX + uiXR));
				pulLB = dHist + (uiNrBins * (uiYB * uiNrX + uiXL));
				pulRB = dHist + (uiNrBins * (uiYB * uiNrX + uiXR));
				clahe_interpolate_(pDst, pSrc, uiXRes,pulLU,pulRU,pulLB,pulRB,uiSubX,uiSubY,dLUT,m_cuStream[0]);
				iStream = (iStream+1)%CUSTREAM_CNT;
				pSrc += uiSubX;
				pDst += uiSubX;
			}
			pSrc += (uiSubY - 1) * uiXRes;
			pDst += (uiSubY - 1) * uiXRes;
		}

		checkCudaErrors(cudaDeviceSynchronize());
	}
	else
	{
		clahe_Interpolate_one_(dst.data, src.data, dLUT, dHist, uiXRes, uiYRes, uiNrX, uiNrY, uiNrBins, m_cuStream[0]);
	}

#if 0
	checkCudaErrors(cudaFree(dHist));
	checkCudaErrors(cudaFree(dLUT));
#endif
	//cudaEventRecord(	stop,	0	);
	//cudaEventSynchronize(	stop);
	//cudaEventElapsedTime(	&elapsedTime,	start,	stop);
	//printf("cuHistEn 1:	%.8f	ms \n", elapsedTime);
}

void CCudaProcess::cuClaheROI(cv::Mat src, cv::Mat dst, cv::Rect roi, unsigned int uiNrX, unsigned int uiNrY, float fCliplimit, int procType /*= 0*/)
{
	kz_pixel_t Min = 0;
	kz_pixel_t Max = 255;

	unsigned int uiNrBins = 256;
	//unsigned int uiNrX = 8, uiNrY = 8;
	unsigned int uiXRes = src.cols*src.channels();
	unsigned int uiYRes = src.rows;
	unsigned int uiXSize, uiYSize;

    unsigned char aLUT[uiNrBins];
    MakeLut(aLUT, Min, Max, uiNrBins);
	unsigned char *dLUT;
	checkCudaErrors(cudaMalloc((void **)&dLUT, uiNrBins));
	checkCudaErrors(cudaMemcpy(dLUT, aLUT, uiNrBins, cudaMemcpyHostToDevice));

	unsigned int pulMapArray[uiNrX*uiNrY*uiNrBins];
	unsigned int histSize = uiNrX*uiNrY*uiNrBins*sizeof(unsigned int);
	unsigned int *dHist;
	int startX, startY, validW, validH;
	startX = roi.x;				startY = roi.y;
	validW = roi.width;	validH = roi.height;
	uiXSize = validW/uiNrX, uiYSize = validH/uiNrY;

	checkCudaErrors(cudaMalloc((void **)&dHist, histSize));
	clahe_hist_roi(dHist, dLUT, src.data, src.cols*src.channels(), src.rows, startX, startY, validW, validH, uiNrX, uiNrY, m_cuStream);
	checkCudaErrors(cudaMemcpy(pulMapArray, dHist, histSize, cudaMemcpyDeviceToHost));

	unsigned int ulClipLimit;
    if(fCliplimit > 0.0) {
        ulClipLimit = (unsigned int) (fCliplimit * (uiXSize * uiYSize) / uiNrBins);
        ulClipLimit = (ulClipLimit < 1UL) ? 1UL : ulClipLimit;
    }
    else ulClipLimit = 1UL<<14;

    unsigned int ulNrPixels = (unsigned int)uiXSize * (unsigned int)uiYSize;
	unsigned int uiX, uiY;
	unsigned int *pulHist;
    for (uiY = 0; uiY < uiNrY; uiY++)
    {
        for (uiX = 0; uiX < uiNrX; uiX++)
        {
            pulHist = &pulMapArray[uiNrBins * (uiY * uiNrX + uiX)];
            ClipHistogram(pulHist, uiNrBins, ulClipLimit);
            MapHistogram(pulHist, Min, Max, uiNrBins, ulNrPixels);
        }
    }

	checkCudaErrors(cudaMemcpy(dHist, pulMapArray, histSize, cudaMemcpyHostToDevice));
	//clahe_interpolate_(src.data, src.cols*src.channels(), src.rows, dHist, );

	//cudaEventRecord(	stop,	0	);
	//cudaEventSynchronize(	stop);
	//cudaEventElapsedTime(	&elapsedTime,	start,	stop);
	//printf("cuHistEn 0:	%.8f	ms \n", elapsedTime);

	if (procType == 0)
	{

		unsigned int uiSubX, uiSubY;
		unsigned int uiXL, uiXR, uiYU, uiYB;
		unsigned int* pulLU, *pulLB, *pulRU, *pulRB;
		kz_pixel_t* pSrc, *pDst;
		int iStream = 0;

		/* Interpolate greylevel mappings to get CLAHE image */
		for (pSrc = (src.data+startY*uiXRes+startX), pDst = (dst.data+startY*uiXRes+startX), uiY = 0; uiY <= uiNrY; uiY++)
		{
			if (uiY == 0)
			{                     /* special case: top row */
				uiSubY = uiYSize >> 1;  uiYU = 0; uiYB = 0;
			}
			else
			{
				if (uiY == uiNrY) {               /* special case: bottom row */
					uiSubY = uiYSize >> 1;    uiYU = uiNrY-1;  uiYB = uiYU;
				}
				else
				{                     /* default values */
					uiSubY = uiYSize; uiYU = uiY - 1; uiYB = uiYU + 1;
				}
			}
			for (uiX = 0; uiX <= uiNrX; uiX++)
			{
				if (uiX == 0)
				{                 /* special case: left column */
					uiSubX = uiXSize >> 1; uiXL = 0; uiXR = 0;
				}
				else
				{
					if (uiX == uiNrX)
					{             /* special case: right column */
						uiSubX = uiXSize >> 1;  uiXL = uiNrX - 1; uiXR = uiXL;
					}
					else
					{                     /* default values */
						uiSubX = uiXSize; uiXL = uiX - 1; uiXR = uiXL + 1;
					}
				}

				pulLU = dHist + (uiNrBins * (uiYU * uiNrX + uiXL));
				pulRU = dHist + (uiNrBins * (uiYU * uiNrX + uiXR));
				pulLB = dHist + (uiNrBins * (uiYB * uiNrX + uiXL));
				pulRB = dHist + (uiNrBins * (uiYB * uiNrX + uiXR));
				clahe_interpolate_(pDst, pSrc, uiXRes,pulLU,pulRU,pulLB,pulRB,uiSubX,uiSubY,dLUT,m_cuStream[0]);
				iStream = (iStream+1)%CUSTREAM_CNT;
				pSrc += uiSubX;
				pDst += uiSubX;
			}
			pSrc	+= (uiXRes - validW);
			pDst+= (uiXRes - validW);
			pSrc += (uiSubY - 1) * uiXRes;
			pDst += (uiSubY - 1) * uiXRes;
		}

		checkCudaErrors(cudaDeviceSynchronize());
	}
	else
	{
		clahe_Interpolate_one_roi(dst.data, src.data, dLUT, dHist, uiXRes, uiYRes, startX, startY, validW, validH, uiNrX, uiNrY, uiNrBins, m_cuStream[0]);
	}

	checkCudaErrors(cudaFree(dHist));
	checkCudaErrors(cudaFree(dLUT));
}

void CCudaProcess::cuCvtClaheROI(cv::Mat src, cv::Mat dst, cv::Rect roi, unsigned int uiNrX, unsigned int uiNrY, float fCliplimit, int code,	int procType/* = 0*/)
{
	int i, chId = 0;
	unsigned char *d_src = load(src, chId);
	unsigned char *d_dst = NULL;
	unsigned int byteCount_src = src.rows * src.cols * src.channels() * sizeof(unsigned char);
	unsigned int byteBlock_src = byteCount_src/CUSTREAM_CNT;
	unsigned int byteCount_dst;
	unsigned int byteBlock_dst;
	cv::Mat	enh_src, enh_dst;

	byteCount_dst = dst.rows * dst.cols * sizeof(unsigned char);
	byteBlock_dst = byteCount_dst/CUSTREAM_CNT;
	cudaMalloc_share((void**)&d_dst, byteCount_dst, chId+1);

	if(code == CV_YUV2GRAY_YUYV)
	{
		for(i = 0; i<CUSTREAM_CNT; i++){
			yuyv2gray_(d_dst + byteBlock_dst*i,
					d_src + byteBlock_src*i,
					src.cols, (src.rows/CUSTREAM_CNT), m_cuStream[i]);
		}
	}
	else if(code == CV_YUV2GRAY_UYVY)
	{
		for(i = 0; i<CUSTREAM_CNT; i++){
			uyvy2gray_(d_dst + byteBlock_dst*i,
					d_src + byteBlock_src*i,
					src.cols, (src.rows/CUSTREAM_CNT), m_cuStream[i]);
		}
	}

	enh_src = cv::Mat(dst.rows, dst.cols, CV_8UC1, d_dst);
	enh_dst = cv::Mat(dst.rows, dst.cols, CV_8UC1, d_dst);
	cuClaheROI(enh_src, enh_dst, roi, uiNrX, uiNrY,  fCliplimit, procType);

	for(i = 0; i<CUSTREAM_CNT; i++){
		cudaMemcpyAsync(dst.data + byteBlock_dst*i, d_dst + byteBlock_dst*i, byteBlock_dst, cudaMemcpyDeviceToHost, m_cuStream[i]);
	}
	for(i=0; i<CUSTREAM_CNT; i++)
		cudaStreamSynchronize(m_cuStream[i]);

	cudaFree_share(d_dst, 1);
}

void cuHistEnh(Mat src, Mat dst)
{
	proc.cuHistEnh(src,dst);
}

void cuUnhazed(Mat src, Mat dst)
{
	proc.cuUnhazed(src,dst);
}

void cuTemporalFilter(cv::Mat src, cv::Mat dst)
{
	proc.cuTmporalFilt(src,dst);
}

void cuClahe(cv::Mat src, cv::Mat dst, unsigned int uiNrX, unsigned int uiNrY, float fCliplimit, int procType)
{
	proc.cuClahe(src,dst,uiNrX,uiNrY,fCliplimit, procType);
}

extern void cuClaheRoi(cv::Mat src, cv::Mat dst, cv::Rect roi, unsigned int uiNrX, unsigned int uiNrY, float fCliplimit, int procType)
{
	proc.cuClaheROI(src,dst,roi, uiNrX,uiNrY,fCliplimit, procType);
}

void cuCvtClaheRoi(cv::Mat src, cv::Mat dst, cv::Rect roi, unsigned int uiNrX, unsigned int uiNrY, float fCliplimit, int code, int procType)
{
	proc.cuCvtClaheROI(src, dst , roi, uiNrX, uiNrY, fCliplimit, code, procType);
}


