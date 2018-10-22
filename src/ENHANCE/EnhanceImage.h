#ifndef DARKCHANNELPRIOHAZEREMOVE_H_
#define DARKCHANNELPRIOHAZEREMOVE_H_

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <cuda_runtime.h>

#define  MinValue(R,G,B)    ( (R = ( R > G ? G : R)) > B ? B : R )
#define  BLOCKSIZE				 8//15          //×îÐ¡ÖµÂË²š¿éŽóÐ¡
#define  GRAYLEVEL				 256            //256žö»Ò¶Èœ×
#define  LIGHTRANGE				 0.1            //Ñ¡Ôñ°µÍšµÀÁÁ¶ÈÇ°0.1%
#define  LIGHTTHRESH		  	 255//220    //ŽóÆøÁÁ¶ÈãÐÖµ
#define  WEIGHT					 0.95//0.85      //ÍžÉäÂÊÐÞÕýÒò×Ó
#define  TRANSMISSIONRATIO		 0.25//0.1        //ÍžÉäÂÊµ÷œÚÒò×Ó
#define  TRANSMISSIONTHRESH      30//25.5        //ÍžÉäÂÊãÐÖµ
#define  TESTAC					 230             //Ac
#define  RADIUS                  32//30          //ÅŒÊý
#define  EPXITON                 0.04
#define  GAMMAVALUE             0.5 // 0.85
#define  COLORTHREASH            10
#define  MIXCHANNLEDARK          30//25
#define  LOWCUT                  0.005//0.005
#define  HIGHTCUT                0.005//0.005

using namespace cv;
using namespace std;

class EnhanceImage
{
public:
	EnhanceImage(void);
	~EnhanceImage(void);

public:
	//°µÍšµÀ
	void DarkChannelProcess(Mat hazeImg, Mat &unHazeImg);
	void getDarkChannel(Mat hazeImg, Mat &rgbMinImg, Mat &darkChImg);
	void MinFilter(Mat rgbMinImg, Mat &darkChImg, int blockSize);
	unsigned char getAtmosphricLight(Mat hazeImg, Mat darkChImg);
	void getTransmission(Mat src, Mat darkChImg, Mat &transImg, double A);
	void fillBorder(Mat &dst, int radius);
	void MixChannel(Mat rgbMinImg, Mat &darkChImg, Mat &mixChImg, unsigned char &mixThrsh);				//»ìºÏÍšµÀ
	void getAvgMinGray(Mat darkChImg, unsigned char &mixThrsh);					//×ÔÊÊÓŠ»ìºÏÍšµÀµÄ»Ò¶ÈãÐÖµ

	void guideFilter(Mat hazeImg, Mat transImg, Mat &guideFiltImg, int radius, double eps); //opencv boxfilter()
	void guideFilter2(Mat hazeImg, Mat transImg, Mat &guideFiltImg, int radius, double eps); //»Ò¶ÈÍŒÎªµŒÏòÍŒ£¬Œò»¯Áœ²ãfor£¬Ê±Œä³€
	void guideFilter3(Mat hazeImg, Mat transImg, Mat &guideFiltImg, int radius, double eps); //rgbÍŒ×÷ÎªµŒÏòÍŒÏñ
	void guideFilter4(Mat hazeImg, Mat transImg, Mat &guideFiltImg, int radius, double eps); //boxfilter_cpu,gpu cÊµÏÖ£¬œá¹ûÓë2Ò»Ñù

	void normalization(Mat src, Mat &dst);     //¹éÒ»»¯
	void getUnhazedImg(Mat hazeImg, Mat guideImg, Mat &UnhazeImg, unsigned char A);
	void GammaCorrect(Mat &UnhazeImg, float fGamma);

	void BoxFilter(Mat src, Mat &dst, int radius);
	void BoxFilter2(Mat src, Mat &dst, int radius);

	void show(Mat src, Mat dst);

	void computeGold(Mat image, Mat &dst, int w, int h, int r);//gpu_cÐŽ·š£¬(float *image, float *temp, int w, int h, int r);
	void hboxfilter_x(Mat id, Mat &od, int w, int h, int r);//gpu_cÐŽ·š£¬(float *id, float *od, int w, int h, int r);
	void hboxfilter_y(Mat id, Mat &od, int w, int h, int r);//(float *id, float *od, int w, int h, int r);


	//ŸÖ²¿×Ô¶¯É«œ×¡¢×Ô¶¯¶Ô±È¶È
	void AutoColorLevelProcess(Mat srcImg, Mat &dstImg);

	void MinMax_Histgram(Mat src, unsigned char &minR, unsigned char &minG, unsigned char &minB,
		         unsigned char &maxR, unsigned char &maxG, unsigned char &maxB);

	void MinMax_Hist(unsigned char *Hist, int width, int height, unsigned char &min, unsigned char &max);

	void creatMap(unsigned char *MapR, unsigned char *MapG, unsigned char *MapB,
				 unsigned char minR, unsigned char minG, unsigned char minB,
				 unsigned char maxR, unsigned char maxG, unsigned char maxB);

	void setMap(unsigned char *Map, unsigned char min, unsigned char max);

	void MapImage(Mat srcImg, unsigned char *MapR, unsigned char *MapG, unsigned char *MapB, Mat &dstImg);

public:
	unsigned char A_r, A_g, A_b;

};

extern "C" int full_f32_(
		float *mem, int size, float value);

extern "C" void initDCM(int width, int height, int blockSize);

extern "C" void unInitDCM(void);

extern "C" int unhazed_rgb32_(
		unsigned char *dst,const unsigned char *src,
					const float *guide, const unsigned char *lut,
					unsigned char A,
					int width, int height);

extern "C" int unhazed_gray_(
		unsigned char *dst,const unsigned char *src,
					const float *guide, const unsigned char *lut,
					unsigned char A,
					int width, int height);

extern "C" int darkChannel_rgb32_(
		float *dst, float *gray, unsigned char *dark, const unsigned char *src,
			int width, int height, float acl, float weight);

extern "C" int darkChannel_gray_(
		float *dst, float *gray, unsigned char *dark, const unsigned char *src,
			int width, int height, float acl, float weight);

extern "C" int rgb32_2norgray_(
		float *gray, const unsigned char *src,
				int width, int height);

extern "C" int norgray2rgb32_(
		unsigned char *dst, const float *src, int width, int height);

extern "C" int norgray2gray_(
		unsigned char *dst, const float *src, int width, int height);

extern "C" int guideFilter_(
		float *dst,
		const float *tran, const float *norgray,
		int width, int height, cudaStream_t stream[]);

extern "C" int clahe_hist_(
		unsigned int *hist,  const unsigned char *dLUT, const unsigned char *data,
		int width, int height, unsigned int uiNrX, unsigned int uiNrY, cudaStream_t stream[]);

extern "C" int clahe_interpolate_(
		unsigned char *pDst, unsigned char *pSrc, int uiXRes,
		unsigned int *pulMapLU, unsigned int *pulMapRU, unsigned int *pulMapLB,  unsigned int *pulMapRB,
        unsigned int uiXSize, unsigned int uiYSize, const unsigned char *pLUT, cudaStream_t stream);

extern "C" int clahe_Interpolate_one_(unsigned char *pDst, unsigned char *pSrc,
		unsigned char *dLUT, unsigned int *dHist,
		unsigned int uiXRes, unsigned int uiYRes, unsigned int uiNrX, unsigned int uiNrY,
		unsigned int uiNrBins, cudaStream_t stream);

extern "C" int clahe_hist_roi(
		unsigned int *hist, const unsigned char *dLUT, const unsigned char *data,
		int width, int height, int startX, int startY, int validW, int validH,
		unsigned int uiNrX, unsigned int uiNrY, cudaStream_t stream[]);

extern "C" int clahe_Interpolate_one_roi(unsigned char *pDst, unsigned char *pSrc,
		unsigned char *dLUT, unsigned int *dHist,
		unsigned int uiXRes, unsigned int uiYRes,
		int startX, int startY, int validW, int validH, unsigned int uiNrX, unsigned int uiNrY,
		unsigned int uiNrBins, cudaStream_t stream);

extern "C" int full_u8_(unsigned char *mem, int size, unsigned char value);

#endif
