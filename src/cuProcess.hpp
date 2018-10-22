/*
 * cuProcess.hpp
 *
 *  Created on: May 11, 2017
 *      Author: ubuntu
 */

#ifndef CUPROCESS_HPP_
#define CUPROCESS_HPP_

//#include <glew.h>
//#include <glut.h>
//#include <freeglut_ext.h>
#include <cuda.h>
//#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#define CUSTREAM_CNT		(8)

#define	 FILTER_REF_NUM		(3)

class CCudaProcess
{
	cudaStream_t m_cuStream[CUSTREAM_CNT];
	uint  *d_Histogram;
public:
	CCudaProcess();
	virtual ~CCudaProcess();

	void cutColor(cv::Mat src, cv::Mat &dst, int code);
	void cuHistEnh(cv::Mat src, cv::Mat dst);
	void cuUnhazed(cv::Mat src, cv::Mat dst);
	void cuClahe(cv::Mat src, cv::Mat dst, unsigned int uiNrX, unsigned int uiNrY, float fCliplimit, int procType = 0);
	void cuClaheROI(cv::Mat src, cv::Mat dst, cv::Rect roi, unsigned int uiNrX, unsigned int uiNrY, float fCliplimit, int procType = 0);
	void cuCvtClaheROI(cv::Mat src, cv::Mat dst, cv::Rect roi, unsigned int uiNrX, unsigned int uiNrY, float fCliplimit, int code, int procType = 0);
	void YUVV2RGB(cv::Mat src, cv::Mat &dst, int code);
	void YUVV2YUVPlan(cv::Mat src, cv::Mat &dst);
	void cuTmporalFilt(cv::Mat src, cv::Mat dst);

protected:
	unsigned char* load(cv::Mat frame, int memChn = 0);
	cv::Mat dmat_create(int cols, int rows, int type, int memChn = 0);
	void dmat_delete(cv::Mat dm, int memChn = 0);

	bool m_bUnhazed;
	unsigned char *d_lut;
	//cv::Mat m_darkImg;
	cv::Mat m_tran, m_norgray;
	cv::Mat m_guideImg;
	int m_curWidth, m_curHeight;
	void unhazed(cv::Mat inImg, cv::Mat outImg);
	int unhazed_init(int width, int height, int blockSize, float gamma);
	void unhazed_uninit();
	int getAtmosphricLight_rgb32(cv::Mat hazeImg, cv::Mat darkChImg);
	int getAtmosphricLight_gray(cv::Mat hazeImg, cv::Mat darkChImg);

	cv::Mat	m_filterRef[FILTER_REF_NUM];
	int		m_Index;
	bool	m_bFltInit;
	void filter_init(cv::Mat src);
	void filter_uninit();
	void _temporalFilter(cv::Mat src, cv::Mat dst, int index);
	void _spaceFilter(cv::Mat src, cv::Mat dst);

	float gaussian_delta;// = 4;
	float euclidean_delta;// = 0.1f;
	int filter_radius;// = 5;
	float *cGaussian, *cEuclidean;
	cv::Mat m_YFrame;
	cv::Mat  m_NoiseMat;
	float noise_IIR_coefficent;// = 0.5;
	float temporal_strenght, spatial_strength, Threshold,max_noise, temporal_trigger_noise;
	float Frame_noise_previous, Frame_noise, Frame_noise_filtered, Totla_Frame_noise;
	cv::Mat spaceFiltMat;

private:
	cudaEvent_t	start, stop;
	float elapsedTime;
	unsigned char *m_pDeviceBuf;
};


extern void cutColor(cv::Mat src, cv::Mat &dst, int code);

extern void cuHistEnh(cv::Mat src, cv::Mat dst);

extern void cuClahe(cv::Mat src, cv::Mat dst, unsigned int uiNrX = 8, unsigned int uiNrY = 8, float fCliplimit = 2.5, int procType = 0);

extern void cuClaheRoi(cv::Mat src, cv::Mat dst, cv::Rect roi, unsigned int uiNrX = 8, unsigned int uiNrY = 8, float fCliplimit = 2.5, int procType = 0);

extern void cuCvtClaheRoi(cv::Mat src, cv::Mat dst, cv::Rect roi, unsigned int uiNrX = 2, unsigned int uiNrY = 2, float fCliplimit = 2.5, int code = CV_YUV2GRAY_YUYV, int procType = 0);

extern void cvClahe(cv::Mat frame);

extern void cuUnhazed(cv::Mat src, cv::Mat dst);

extern void cuTemporalFilter(cv::Mat src, cv::Mat dst);

extern void cvtBigVideo(cv::Mat src,cv::Mat &dst,  int type);

extern void cvtBigVideo_plan(cv::Mat src,cv::Mat &dst);

#endif /* CUPROCESS_HPP_ */
