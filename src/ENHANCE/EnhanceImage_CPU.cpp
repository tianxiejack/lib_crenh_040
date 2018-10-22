#include <iostream>
#include "EnhanceImage.h"

using namespace cv;
using namespace std;

#define clip( val, minv, maxv )    (( val = (val < minv ? minv : val ) ) > maxv ? maxv : val )

EnhanceImage::EnhanceImage(void)
{
	A_r = 0;
	A_g = 0;
	A_b = 0;;
}


EnhanceImage::~EnhanceImage(void)
{
}

void EnhanceImage::getDarkChannel(Mat src, Mat &rgbMin, Mat &darkChImg)
{
	rgbMin.create(src.rows, src.cols, CV_8UC1);

	CvScalar pixel;
	vector<Mat> temp;
	unsigned char R, G, B;
	Mat MixChImg;
	unsigned char mixThrsh;

	int cols = src.cols;
	int rows = src.rows;
	double px = 0;
	double time0 = static_cast<double>(getTickCount());

	//Çó3žö·ÖÁ¿×îÐ¡Öµ
	for(int j=0; j<rows; j++){    
		uchar *src_data = src.ptr<uchar>(j);
		uchar *dst_data = rgbMin.ptr<uchar>(j);
		for(int i=0; i<cols; i++){
			B = src_data[i*3];
			G = src_data[i*3+1];
			R = src_data[i*3+2]; 
			px = MinValue(R,G,B);
			dst_data[i] = px;
		}
	}

	time0 = 1000*((double)getTickCount() - time0)/getTickFrequency();
	cout<<"getMinRGB run time: "<<time0<<"msec"<<endl;

	MinFilter(rgbMin, darkChImg, BLOCKSIZE);
// 	getAvgMinGray(darkChImg, mixThrsh);								//add
// 	MixChannel(rgbMin, darkChImg, MixChImg, mixThrsh);				//add
//     darkChImg = MixChImg.clone();								//add

// 	imshow("rgbMin",rgbMin);
// 	waitKey();
}

void EnhanceImage::getAvgMinGray(Mat darkChImg, unsigned char &mixThrsh)
{
	int cols = darkChImg.cols;
	int rows = darkChImg.rows;
	int pixleNum = cols*rows*0.03;
	unsigned char *Hist = NULL;
	mixThrsh = 0;
	int sum = 0;

	Hist = new unsigned char[GRAYLEVEL];
	memset(Hist, 0, GRAYLEVEL*sizeof(unsigned char));

	for(int j=0; j<rows; j++){
		uchar *data = darkChImg.ptr<uchar>(j);
		for(int i=0; i<cols; i++){
			uchar pixel = data[i];
			Hist[pixel]++;
		}
	}

	int sumgray = 0;

	for(int i=0; i<GRAYLEVEL; i++){
		sum += Hist[i];
		sumgray += (i)*Hist[i];  //È¥³ý0µÄÓ°Ïì

		if(sum>pixleNum){
			mixThrsh = sumgray / sum;
			break;
		}
	}

	if(Hist!=NULL){
		delete []Hist;
	}
}

void EnhanceImage::MixChannel(Mat rgbMinImg, Mat &darkChImg, Mat &mixChImg, unsigned char &mixThrsh)
{
	mixChImg.create(rgbMinImg.rows, rgbMinImg.cols, CV_8UC1);

	int cols, rows;
	cols = rgbMinImg.cols;
	rows = rgbMinImg.rows;
	unsigned char tmp = MIXCHANNLEDARK;//mixThrsh;//MIXCHANNLEDARK;

	for(int j=0; j<rows; j++){
		uchar *data_min = rgbMinImg.ptr<uchar>(j);
		uchar *data_dark = darkChImg.ptr<uchar>(j);
		uchar *data_mix = mixChImg.ptr<uchar>(j);

		for(int i=0; i<cols; i++){
			if(data_dark[i]<=tmp){
				data_mix[i] = data_min[i];
			}else{
				data_mix[i] = data_dark[i];
			}
		}
	}
	imshow("mixChImg", mixChImg);
	waitKey();
}

void EnhanceImage::MinFilter(Mat rgbMinImg, Mat &darkImg, int blockSize)
{
	darkImg.create(rgbMinImg.rows, rgbMinImg.cols, CV_8UC1);

	int cols, rows;
	int block_x, block_y;
	Rect rect;
	Mat darkRoi;
	Mat tempRoi;

	double min = 0;
	double max = 0;
	cols = rgbMinImg.cols;
	rows = rgbMinImg.rows;
	block_x = (cols-1)/blockSize;
	block_y = (rows-1)/blockSize;

	double time0 = static_cast<double>(getTickCount());
	uchar tmp = 20;//25;

#if 1
	//×îÐ¡ÖµÂË²š
	for(int j=blockSize/2; j<rows-blockSize/2; j++){

		uchar *data = darkImg.ptr<uchar>(j);
		//uchar *data_min = rgbMinImg.ptr<uchar>(j-blockSize/2);//add

		for(int i=blockSize/2; i<cols-blockSize/2; i++){
			rect = Rect(i-blockSize/2, j-blockSize/2, blockSize, blockSize);
			darkRoi = rgbMinImg(rect);
			minMaxLoc(darkRoi,&min, &max,NULL,NULL);
			data[i] = min;
		}
	}
	
// 	Ìî²¹ÉÏ±ßœç
		for(int j=0; j<blockSize/2; j++){
	
			uchar *tmp = darkImg.ptr<uchar>(j);
			uchar *data = darkImg.ptr<uchar>(blockSize/2);
	
			for(int i=0; i<cols; i++){
	
				if(i>=0 && i<blockSize/2)
				{
					tmp[i] = data[blockSize/2];
				}
				else if(i>=cols-blockSize/2 && i<cols)
				{
					tmp[i] = data[cols-blockSize/2 -1];
				}
				else
				{
					tmp[i] = data[i];
				}
			}	
		}
		//Ìî²¹ÏÂ±ßœç
		for(int j=rows-blockSize/2; j<rows; j++){
	
			uchar *tmp = darkImg.ptr<uchar>(j);
			uchar *data = darkImg.ptr<uchar>(rows-blockSize/2-1);
	
			for(int i=0; i<cols; i++){
	
				if(i>=0 && i<blockSize/2)
				{
					tmp[i] = data[blockSize/2];
				}
				else if(i>=cols-blockSize/2 && i<cols)
				{
					tmp[i] = data[cols-blockSize/2 -1];
				}
				else
				{
					tmp[i] = data[i];
				}
			}	
		}
		//Ìî²¹×óÓÒ±ßœç
		for(int j=blockSize/2; j<rows-blockSize/2; j++){
	
			uchar *tmp = darkImg.ptr<uchar>(j);
			uchar *data = darkImg.ptr<uchar>(j);
	  		
	  		for(int i=0; i<cols; i++){
	
				if(i>=0 && i<blockSize/2)
				{
					tmp[i] = data[blockSize/2];
				}
				if(i>=cols-blockSize/2 && i<cols)
				{
					tmp[i] = data[cols-blockSize/2 -1];
				}
			}
		}
#else
	int r = (blockSize-1)/2;
	Mat dst_expand, dst, Roimg;

	int ex_cols = cols+blockSize;
	int ex_rows = rows+blockSize;

	dst_expand.create(ex_rows, ex_cols, CV_8UC1);
	//dst.create(rows, cols, CV_8UC1);
	copyMakeBorder(rgbMinImg, dst_expand, r, blockSize-r, r, blockSize-r,BORDER_REPLICATE);

	for(int j=r; j<ex_rows-(blockSize-r); j++){
		uchar *data = darkImg.ptr<uchar>(j-r);
		for(int i=r; i<ex_cols-(blockSize-r); i++){

			rect = Rect(i-r, j-r, blockSize, blockSize);
			darkRoi = dst_expand(rect);
			minMaxLoc(darkRoi,&min, &max,NULL,NULL);
			data[i-r] = min;
		}
	}
#endif

	time0 = 1000*((double)getTickCount() - time0)/getTickFrequency();
	cout<<"MinFilter run time: "<<time0<<"msec"<<endl;

	imshow("darkImg",darkImg);
	waitKey();
}

unsigned char EnhanceImage::getAtmosphricLight(Mat hazeImg, Mat darkChImg)
{
	unsigned char A = 0;        //ŽóÆø¹âÇ¿
	Point min_loc, max_loc;
	Rect roi;
	Mat gray, imgROI;

	double min_dark = 0;
	double max_dark = 0;
	double time0 = static_cast<double>(getTickCount());

	//Í³ŒÆ°µÍšµÀ»Ò¶ÈžÅÂÊ·Ö²Œ
	double graylevel = 0; 
	double sum = 0;         //ÏñËØµãÂú×ãÒªÇóµÄAµÄºÍ
	int pointNum = 0;       //Âú×ãÒªÇóµÄÏñËØµãÊý
	int cols = darkChImg.cols;
	int rows = darkChImg.rows;

	float darkHistogram[GRAYLEVEL];
	float tmpHistogram[GRAYLEVEL];
	memset(darkHistogram, 0, sizeof(darkHistogram));
	memset(tmpHistogram, 0, sizeof(tmpHistogram));

	for(int j=0; j<rows; j++){

		uchar *data = darkChImg.ptr<uchar>(j);

		for(int i=0; i<cols; i++){
			uchar gray_level= data[i];
			darkHistogram[gray_level]++;
		}
	}

	for(int i=0; i<256; i++){
		darkHistogram[i] = darkHistogram[i] / (rows*cols);
	}

	//ŽÓ°µÍšµÀÖÐ°ŽÕÕÁÁ¶ÈŽóÐ¡È¡Ç°0.1%µÄÏñËØ
	float ligth_ratio = 1 - LIGHTRANGE;         //0.999

	for(int i=0; i<GRAYLEVEL; i++){
		for(int j=0; j<=i; j++){
			tmpHistogram[i] += darkHistogram[j];

			if(tmpHistogram[i] > ligth_ratio)
			{
				graylevel = (double)i;         //ÕÒµœÀÛŒÆ99.9%µÄÁÙœç»Ò¶Èœ×£¬Ê£ÏÂµÄŸÍÊÇ·ûºÏÒªÇóµÄ0.1%
				i = 256;
				break;
			}
		}
	}
	

	vector<Point> ps;
	Point tmps;
	for(int j=0; j<rows; j++){

		uchar *data_haze = hazeImg.ptr<uchar>(j);
		uchar *data_dark = darkChImg.ptr<uchar>(j);

		for(int i=0; i<cols; i++){
			double temp = data_dark[i];

			if(temp > graylevel)
			{
				sum += data_haze[i*3] + data_haze[i*3 + 1] + data_haze[i*3 + 2];
				pointNum++;
				tmps.x = i;
				tmps.y = j;
				ps.push_back(tmps);

			}
		}
	}

#if 0
	cvtColor(hazeImg, gray, CV_RGB2GRAY);

	for(int i=0; i<ps.size()-1; i++){
		int x = ps[i].x;
		int y = ps[i].y;
		if(maxPixel < gray.at<uchar>(y,x))
		{
			maxPixel = gray.at<uchar>(y,x);             //Ö»È¡»Ò¶ÈÖµ
		}

	}
#else   
	unsigned char maxPixel = 0;
	int posx, posy;

	for(int i=0; i<ps.size()-1; i++){
		int x = ps[i].x;
		int y = ps[i].y;
		uchar temp = (hazeImg.at<Vec3b>(y,x)[0] + hazeImg.at<Vec3b>(y,x)[1] +hazeImg.at<Vec3b>(y,x)[2])/3;   //È¡ÈýžöÍšµÀÆœŸùÖµ
		if(maxPixel < temp)
		{
			maxPixel = temp;
			posx = x;
			posy = y;
		}
	}
#endif

	A = maxPixel;
	A_b = hazeImg.at<Vec3b>(posy, posx)[0];
	A_g = hazeImg.at<Vec3b>(posy, posx)[1];
	A_r = hazeImg.at<Vec3b>(posy, posx)[2];
	//A = sum / (3 * pointNum);

	if(A>=LIGHTTHRESH)
		A = LIGHTTHRESH;

	time0 = 1000*((double)getTickCount() - time0)/getTickFrequency();
 	cout<<"getAtmosphricLight run time: "<<time0<<"msec"<<endl;

	return A;
}

void EnhanceImage::getTransmission(Mat src, Mat darkChImg, Mat &transImg, double A)
{
	if(transImg.rows ==0 && transImg.cols==0){
		transImg.create(darkChImg.rows, darkChImg.cols, CV_32FC1);  
	}
	
	int cols = darkChImg.cols;
	int rows = darkChImg.rows;
	double weight = WEIGHT;
	double time0 = static_cast<double>(getTickCount());
	float temp = 0;
	//ÇóÍžÉäÂÊ
	for(int j=0; j<rows; j++){ 

		uchar *data_dark = darkChImg.ptr<uchar>(j);
		float *data_trans = transImg.ptr<float>(j);    /*uchar*/
		uchar *data_src = src.ptr<uchar>(j);

		for(int i=0; i<cols; i++){
			float temp = 1- weight*data_dark[i]/A;     //TESTAC
			//uchar a = fabs((data_src[i]+data_src[i+1]+data_src[i+2])/3 - A);
			//if(a<COLORTHREASH){
			//	temp = 1- weight*data_dark[i]/A + 0.1;//(COLORTHREASH-a)/COLORTHREASH;
			//}else{
			//	temp = 1- weight*data_dark[i]/A;
			//}
			if(temp<0){
				temp = 0;
			}
			data_trans[i] = temp;
		}
	}

	time0 = 1000*((double)getTickCount() - time0)/getTickFrequency();
	cout<<"getTransmission run time: "<<time0<<"msec"<<endl;

	imshow("transImg",transImg);
	waitKey();
}

void EnhanceImage::guideFilter(Mat frame, Mat transImg, Mat &guideFiltImg, int r, double eps)
{
	Mat hazegray, guide, transImg_;         
	Mat N;//Ã¿žöÁÚÓòµÄŽóÐ¡
	Mat mean_I, mean_P, mean_IP;
	Mat cov_IP, mean_II;
	Mat var_I;//·œ²î
	Mat a, b , q;
	Mat mean_a, mean_b;

	int cols = frame.cols;
	int rows = frame.rows;
	double time0 = static_cast<double>(getTickCount());

	hazegray.create(rows, cols, CV_8UC1);
	guide.create(rows, cols, CV_32FC1);
	guideFiltImg.create(rows, cols, CV_8UC1);

	hazegray = frame;	//cvtColor(frame, hazegray, CV_RGB2GRAY);
	hazegray.convertTo(guide, CV_32FC1, 1.0/255);      //µŒÏòÍŒÏñI
	//transImg.convertTo(transImg_, CV_32FC1);             //ÂË²šÍŒP

	boxFilter(Mat::ones(rows, cols, guide.type()), N, CV_32FC1, Size(r,r));  //N
	boxFilter(guide, mean_I, CV_32FC1, Size(r,r));
	boxFilter(transImg, mean_P, CV_32FC1, Size(r,r));
	boxFilter(guide.mul(transImg), mean_IP, CV_32FC1, Size(r,r));

//	mean_I = mean_I / N; mean_P = mean_P / N; mean_IP = mean_IP / N;

//  	show(mean_I, mean_I);
//  	show(mean_P, mean_P);
//  	show(mean_IP, mean_IP);

	cov_IP = mean_IP - mean_I.mul(mean_P);
	boxFilter(guide.mul(guide), mean_II, CV_32FC1, Size(r,r));
	var_I = mean_II - mean_I.mul(mean_I);

//	    show(cov_IP, cov_IP);

	a = cov_IP / (var_I + eps);
	b = mean_P - a.mul(mean_I);

//  	show(a, a);
//  	show(b, b);

	boxFilter(a, mean_a, CV_32FC1, Size(r,r));
	boxFilter(b, mean_b, CV_32FC1, Size(r,r));

//  	show(mean_a, mean_a);
//  	show(mean_b, mean_b);

	mean_a = mean_a / N;	
	mean_b = mean_b / N;

//  	show(mean_a, mean_a);
//  	show(mean_b, mean_b);

	guideFiltImg = (mean_a.mul(guide) + mean_b) ;

	for(int j=0; j<rows; j++){
		float *data = guideFiltImg.ptr<float>(j);
		for(int i=0; i<cols; i++){
			clip(data[i], 0, 1);
		}
	}

	time0 = 1000*((double)getTickCount() - time0)/getTickFrequency();
	cout<<"guideFilter run time: "<<time0<<"msec"<<endl;

	imshow("guideFiltImg", guideFiltImg);
	waitKey();
}

void EnhanceImage::guideFilter2(Mat frame, Mat transImg, Mat &guideFiltImg, int r, double eps)
{
	Mat hazegray, guide, transImg_;         
	Mat N;//Ã¿žöÁÚÓòµÄŽóÐ¡
	Mat mean_I, mean_P, mean_IP;
	Mat cov_IP, mean_II;
	Mat var_I;//·œ²î
	Mat a, b , q;
	Mat mean_a, mean_b;

	int cols = frame.cols;
	int rows = frame.rows;
	double time0 = static_cast<double>(getTickCount());

	hazegray.create(rows, cols, CV_8UC1);
	guideFiltImg.create(rows, cols, CV_8UC1);
	guide.create(rows, cols, CV_32FC1);

	cvtColor(frame, hazegray, CV_RGB2GRAY);						
	normalization(hazegray, guide);

	BoxFilter(Mat::ones(rows, cols, guide.type()), N, r);
	BoxFilter(guide, mean_I, r);
	BoxFilter(transImg, mean_P, r);
	BoxFilter(guide.mul(transImg), mean_IP, r);

	cov_IP = mean_IP - mean_I.mul(mean_P);									 //mean_IP - mean_I*mean_P
	BoxFilter(guide.mul(guide), mean_II, r);
	var_I = mean_II - mean_I.mul(mean_I);

	a = cov_IP / (var_I + eps);
	b = mean_P - a.mul(mean_I);

	BoxFilter(a, mean_a, r);
	BoxFilter(b, mean_b, r);

	mean_a = mean_a / N;	
	mean_b = mean_b / N;

	guideFiltImg = (mean_a.mul(guide) + mean_b) ;

	for(int j=0; j<rows; j++){
		float *data = guideFiltImg.ptr<float>(j);
		for(int i=0; i<cols; i++){
			clip(data[i], 0, 1);
		}
	}

	time0 = 1000*((double)getTickCount() - time0)/getTickFrequency();
	cout<<"guideFilter run time: "<<time0<<"msec"<<endl;

	imshow("guideFiltImg", guideFiltImg);
	waitKey();
}

void EnhanceImage::guideFilter3(Mat frame, Mat transImg, Mat &guideFiltImg, int r, double eps)
{
	Mat hazegray, transImg_;
	Mat guide_R, guide_G, guide_B;
	Mat N;//Ã¿žöÁÚÓòµÄŽóÐ¡
	Mat mean_Ir, mean_Ig, mean_Ib;
	Mat	mean_P;//r, mean_Pg, mean_Pb;
	Mat mean_IPr, mean_IPg, mean_IPb;
	Mat cov_IPr, cov_IPg, cov_IPb;
	Mat mean_IIr, mean_IIg, mean_IIb;
	Mat var_Ir, var_Ig, var_Ib;//·œ²î
	Mat a_r, a_g, a_b;
	Mat	b_r , b_g, b_b;
	Mat q_r, q_g, q_b;
	Mat mean_ar, mean_ag, mean_ab;
	Mat mean_br, mean_bg, mean_bb;
	vector<Mat> RGBMix;

	int cols = frame.cols;
	int rows = frame.rows;
	double time0 = static_cast<double>(getTickCount());

	hazegray.create(rows, cols, CV_8UC1);
	guide_R.create(rows, cols, CV_32FC1);
	guide_G.create(rows, cols, CV_32FC1);
	guide_B.create(rows, cols, CV_32FC1);
	guideFiltImg.create(rows, cols, CV_8UC1);

	//cvtColor(frame, hazegray, CV_RGB2GRAY);
	//hazegray.convertTo(guide, CV_32FC1, 1.0/255.0);      //µŒÏòÍŒÏñI
	//transImg.convertTo(transImg_, CV_32FC1);             //ÂË²šÍŒP
	split(frame, RGBMix);

	guide_B = RGBMix.at(0);
	guide_G = RGBMix.at(1);
	guide_R = RGBMix.at(2);

	guide_B.convertTo(guide_B, CV_32FC1, 1.0/255.0);
	guide_G.convertTo(guide_G, CV_32FC1, 1.0/255.0);
	guide_R.convertTo(guide_R, CV_32FC1, 1.0/255.0);

	boxFilter(Mat::ones(rows, cols, CV_32FC1), N, CV_32FC1, Size(r,r));  //N

	//mean_I ¡¢mean_P ¡¢ mean_IP
	boxFilter(guide_B, mean_Ib, CV_32FC1, Size(r,r));
	boxFilter(guide_G, mean_Ig, CV_32FC1, Size(r,r));
	boxFilter(guide_R, mean_Ir, CV_32FC1, Size(r,r));

	boxFilter(transImg, mean_P, CV_32FC1, Size(r,r));

	boxFilter(guide_B.mul(transImg), mean_IPb, CV_32FC1, Size(r,r));
	boxFilter(guide_G.mul(transImg), mean_IPg, CV_32FC1, Size(r,r));
	boxFilter(guide_R.mul(transImg), mean_IPr, CV_32FC1, Size(r,r));

	cov_IPb = mean_IPb - mean_Ib.mul(mean_P);
	cov_IPg = mean_IPg - mean_Ig.mul(mean_P);
	cov_IPr = mean_IPr - mean_Ir.mul(mean_P);

	boxFilter(guide_B.mul(guide_B), mean_IIb, CV_32FC1, Size(r,r));
	boxFilter(guide_G.mul(guide_G), mean_IIg, CV_32FC1, Size(r,r));
	boxFilter(guide_R.mul(guide_R), mean_IIr, CV_32FC1, Size(r,r));

	var_Ib = mean_IIb - mean_Ib.mul(mean_Ib);
	var_Ig = mean_IIg - mean_Ig.mul(mean_Ig);
	var_Ir = mean_IIr - mean_Ir.mul(mean_Ir);

	a_b = cov_IPb / (var_Ib + eps);
	a_g = cov_IPg / (var_Ig + eps);
	a_r = cov_IPr / (var_Ir + eps);

	b_b = mean_P - a_b.mul(mean_Ib);
	b_g = mean_P - a_g.mul(mean_Ig);
	b_r = mean_P - a_r.mul(mean_Ir);

	boxFilter(a_b, mean_ab, CV_32FC1, Size(r,r));
	boxFilter(a_g, mean_ag, CV_32FC1, Size(r,r));
	boxFilter(a_r, mean_ar, CV_32FC1, Size(r,r));

	boxFilter(b_b, mean_bb, CV_32FC1, Size(r,r));
	boxFilter(b_g, mean_bg, CV_32FC1, Size(r,r));
	boxFilter(b_r, mean_br, CV_32FC1, Size(r,r));


	mean_ab = mean_ab / N;	mean_ag = mean_ag / N;  mean_ar = mean_ar / N;
	mean_bb = mean_bb / N;  mean_bg = mean_bg / N;  mean_br = mean_br / N;

	vector<Mat> guideMerge;
	Mat tmp1 = (unsigned char)255*(mean_ab.mul(guide_B) + mean_bb.mul(guide_B));
    Mat tmp2 = (unsigned char)255*(mean_ag.mul(guide_G) + mean_bg.mul(guide_G));
	Mat tmp3 = (unsigned char)255*( mean_ar.mul(guide_R) + mean_br.mul(guide_R));

	guideMerge.push_back(tmp1);
	guideMerge.push_back(tmp2);
	guideMerge.push_back(tmp3);

	merge(guideMerge, guideFiltImg);
	show(tmp1, tmp1);show(tmp2, tmp2);show(tmp3, tmp3);
	//guideFiltImg = (unsigned char)255*((mean_ab.mul(guide_B) + mean_ag.mul(guide_G) + mean_ar.mul(guide_R)/3)
	//	                         + (mean_bb.mul(guide_B) + mean_bg.mul(guide_G) + mean_br.mul(guide_R)));
	//q = mean_a.mul(guide) + mean_b;


	//guideFiltImg = (mean_a.mul(guide) + mean_b) ;

	for(int j=0; j<rows; j++){
		float *data = guideFiltImg.ptr<float>(j);
		for(int i=0; i<cols; i++){
			clip(data[i], 0, 1);
		}
	}

	time0 = 1000*((double)getTickCount() - time0)/getTickFrequency();
	cout<<"guideFilter run time: "<<time0<<"msec"<<endl;

	imshow("guideFiltImg", guideFiltImg);
	waitKey();
}

void EnhanceImage::guideFilter4(Mat hazeImg, Mat transImg, Mat &guideFiltImg, int r, double eps)
{
	Mat hazegray, guide, transImg_;
	Mat N;//Ã¿žöÁÚÓòµÄŽóÐ¡
	Mat mean_I, mean_P, mean_IP;
	Mat cov_IP, mean_II;
	Mat var_I;//·œ²î
	Mat a, b , q;
	Mat mean_a, mean_b;

	int cols = hazeImg.cols;
	int rows = hazeImg.rows;
	double time0 = static_cast<double>(getTickCount());

	hazegray.create(rows, cols, CV_8UC1);
	guideFiltImg.create(rows, cols, CV_8UC1);
	guide.create(rows, cols, CV_32FC1);

	cvtColor(hazeImg, hazegray, CV_RGB2GRAY);
	normalization(hazegray, guide);

	computeGold(Mat::ones(rows, cols, guide.type()), N, cols, rows, r);//BoxFilter(Mat::ones(rows, cols, guide.type()), N, r);
	computeGold(guide, mean_I, cols, rows, r);//BoxFilter(guide, mean_I, r);
	computeGold(transImg, mean_P, cols, rows, r);//BoxFilter(transImg, mean_P, r);
	computeGold(guide.mul(transImg), mean_IP, cols, rows, r);//BoxFilter(guide.mul(transImg), mean_IP, r);

	cov_IP = mean_IP - mean_I.mul(mean_P);									 //mean_IP - mean_I*mean_P
	computeGold(guide.mul(guide), mean_II, cols, rows, r);//BoxFilter(guide.mul(guide), mean_II, r);
	var_I = mean_II - mean_I.mul(mean_I);

	a = cov_IP / (var_I + eps);
	b = mean_P - a.mul(mean_I);

	computeGold(a, mean_a, cols, rows, r);//BoxFilter(a, mean_a, r);
	computeGold(b, mean_b, cols, rows, r);//BoxFilter(b, mean_b, r);

	mean_a = mean_a / N;
	mean_b = mean_b / N;

	guideFiltImg = (mean_a.mul(guide) + mean_b) ;
	//q = mean_a.mul(guide) + mean_b;

	for(int j=0; j<rows; j++){
		float *data = guideFiltImg.ptr<float>(j);
		for(int i=0; i<cols; i++){
			clip(data[i], 0, 1);
		}
	}

	time0 = 1000*((double)getTickCount() - time0)/getTickFrequency();
	cout<<"guideFilter run time: "<<time0<<"msec"<<endl;

	imshow("guideFiltImg", guideFiltImg);
	waitKey();
}

void EnhanceImage::BoxFilter(Mat src, Mat &dst, int r)
{
	dst.create(src.rows, src.cols, CV_32FC1);

	int cols, rows;
	int boxWidth, boxHeight;

	cols = src.cols;
	rows = src.rows;
	boxWidth = cols - r;
	boxHeight = rows -r;
	double time0 = static_cast<double>(getTickCount());

	//init
	float *buffer = NULL;
	float *buffer2 = NULL;
	float *sum = NULL;
	float *sum2 = NULL;

	buffer = new float[cols];
	buffer2 = new float[cols];
// 	sum = new float[boxWidth*boxHeight];
// 	sum2 = new float[boxWidth*boxHeight];

	memset(buffer, 0, cols*sizeof(float));
	memset(buffer2, 0, cols*sizeof(float));
// 	memset(sum, 0, boxWidth*boxHeight*sizeof(float));
// 	memset(sum2, 0, boxWidth*boxHeight*sizeof(float));
	
	//boxfilter

	for(int y=0; y<r; y++){
		float *data = src.ptr<float>(y);
		for(int x=0; x<cols; x++){     
			float pixel = data[x];
			buffer[x] += pixel;            //ÁÐÖ®ºÍ
			buffer2[x] += pixel*pixel;      
		}
	}

	for(int y=0; y<rows-r; y++){

		float *data_dst = dst.ptr<float>(y+r/2);
		
		float tmpSum = 0;
		float tmpSum2 = 0;

		for(int j=0; j<r; j++){
			tmpSum += buffer[j];          //ÁÚÓòÖ®ºÍ
			tmpSum2 += buffer2[j];        
		}

		for(int x=0; x<cols-r; x++){   

			if(x!=0){
				tmpSum = tmpSum - buffer[x-1] + buffer[x-1+r];          //ÓÒÒÆ£¬ÁÚÓòÖ®ºÍŒõÈ¥ÒÆ³öµÄÁÐÖ®ºÍ£¬ÔÙŒÓÉÏÒÆœøµÄÁÐÖ®ºÍ
				tmpSum2 = tmpSum2 - buffer2[x-1] + buffer2[x-1+r];		
			}
			//sum[y*(cols-r) + x] = tmpSum;
			//sum2[y*(cols-r) + x] = tmpSum2;
			data_dst[x+r/2] = tmpSum/(r*r);      //tmpSum2/(r*r)  
		}

		float *data = src.ptr<float>(y);
		float *data_next = src.ptr<float>(y+r);

		for(int x=0; x<cols; x++){					  //bufferžüÐÂ,×óÆð£¬ÏÂÒÆ£¬ŒõÈ¥ÒÆ³öµÄ,ŒÓÉÏÒÆœøµÄ
			float pixel = data[x];
			float pixel2 = data_next[x];
					
			buffer[x] = buffer[x] - pixel + pixel2;
			buffer2[x] = buffer2[x] - pixel*pixel + pixel2*pixel2;
		}
	}	

	fillBorder(dst,r);      //Ìî²¹±ßœç

	if(buffer!=NULL){
		delete []buffer;
	}
	if(buffer2!=NULL){
		delete []buffer2;
	}
// 	if(sum!=NULL){
// 		delete []sum;
// 	}
// 	if(sum2!=NULL){
// 		delete []sum2;
// 	}
}

void EnhanceImage::BoxFilter2(Mat src, Mat &dst, int radius)
{
	int cols, rows, r;
	int width_buf, height_buf;
	float sum;
	Mat buffer;

	r = radius;
	cols = src.cols;
	rows = src.rows;
	//width_buf = cols-r;
	//height_buf = rows-r;
	//sum = 0;

	int width = cols-r+1;
	int height = rows-r+1;

	dst.create(rows, cols, CV_32FC1);
	dst = Mat::zeros(rows, cols, CV_32FC1);
	buffer = Mat::zeros(rows, width, CV_32FC1);

	for(long j=0; j<rows*width; j++){
		for(int i=0; i<r; i++){
			buffer.at<float>(j/width, j%width) += src.at<float>(j/width, j%width + i)/(r*r);
		}
	}

	for(long j=0; j<height*width; j++){
		for(int i=0; i<r; i++){
			dst.at<float>(j%height + r/2, j/height + r/2) += buffer.at<float>(j%height+i, j/height);
		}
	}
	fillBorder(dst,r);      //Ìî²¹±ßœç
}

void EnhanceImage::fillBorder(Mat &dst, int radius)
{	
	int r = radius;
	int cols = dst.cols;
	int rows = dst.rows;

	//Ìî²¹ÉÏ±ßœç
	for(int j=0; j<radius/2; j++){

		float *tmp = dst.ptr<float>(j);
		float *data = dst.ptr<float>(r/2);

		for(int i=0; i<cols; i++){

			if(i>=0 && i<r/2)
			{
				tmp[i] = data[r/2];
			}
			else if(i>=cols-r/2 && i<cols)
			{
				tmp[i] = data[cols-r/2 -1];
			}
			else
			{
				tmp[i] = data[i];
			}
		}	
	}
// 	Ìî²¹ÏÂ±ßœç
	for(int j=rows-r/2; j<rows; j++){

		float *tmp = dst.ptr<float>(j);
		float *data = dst.ptr<float>(rows-r/2-1);

		for(int i=0; i<cols; i++){

			if(i>=0 && i<r/2)
			{
				tmp[i] = data[r/2];
			}
			else if(i>=cols-r/2 && i<cols)
			{
				tmp[i] = data[cols-r/2 -1];
			}
			else
			{
				tmp[i] = data[i];
			}
		}	
	}
	//Ìî²¹×óÓÒ±ßœç
	for(int j=r/2; j<rows-r/2; j++){

		float *tmp = dst.ptr<float>(j);
		float *data = dst.ptr<float>(j);

		for(int i=0; i<cols; i++){

			if(i>=0 && i<r/2)
			{
				tmp[i] = data[r/2];
			}
			if(i>=cols-r/2 && i<cols)
			{
				tmp[i] = data[cols-r/2 -1];
			}
		}
	}
}

void EnhanceImage::normalization(Mat src, Mat &dst)
{
	for(int j=0; j<src.rows; j++){
		uchar *data_src = src.ptr<uchar>(j);
		float *data_dst = dst.ptr<float>(j);
		for(int i=0; i<src.cols; i++){
			data_dst[i] = (float)data_src[i]/255.0;
		}
	}
}

void EnhanceImage::getUnhazedImg(Mat hazeImg, Mat guideImg, Mat &UnhazeImg, unsigned char A)
{
// 	if(guideImg.type() != CV_8UC1){
// 		guideImg.convertTo(guideImg, CV_8UC1);
// 	}

	vector<Mat> RGBimg;
	int cols = hazeImg.cols;
	int rows = hazeImg.rows;

	UnhazeImg.create(rows, cols, CV_8UC3);
	double time0 = static_cast<double>(getTickCount());

	//ÈýÍšµÀ·Ö±ðŽŠÀí
	split(hazeImg, RGBimg);

	for(int j =0; j<rows; j++){
#if 1
		uchar *data_b = RGBimg.at(0).ptr<uchar>(j);
		uchar *data_g = RGBimg.at(1).ptr<uchar>(j);
		uchar *data_r = RGBimg.at(2).ptr<uchar>(j);
		float *data_guide = guideImg.ptr<float>(j);

		for(int i=0; i<cols; i++){
			float temp = data_guide[i];

			if(temp/*/255 */< TRANSMISSIONRATIO){
				temp = TRANSMISSIONRATIO/*TRANSMISSIONTHRESH*/;
			}
			int value_b = (int)/*255**/(data_b[i] - A_b/*A*/)/temp + A_b/*A*/;
			int value_g = (int)/*255**/(data_g[i] - A_g/*A*/)/temp + A_g/*A*/;
			int value_r = (int)/*255**/(data_r[i] - A_r/*A*/)/temp + A_r/*A*/;
			data_r[i] = clip(value_r, 0, 255);
			data_b[i] = clip(value_b, 0, 255);
			data_g[i] = clip(value_g, 0, 255);
#else
		Mat guide_r, guide_g, guide_b;
		vector<Mat> guideMerge;

		split(guideImg, guideMerge);

		uchar *data_b = RGBimg.at(0).ptr<uchar>(j);
		uchar *data_g = RGBimg.at(1).ptr<uchar>(j);
		uchar *data_r = RGBimg.at(2).ptr<uchar>(j);

		uchar *data_guide_b = guideMerge.at(0).ptr<uchar>(j);
		uchar *data_guide_g= guideMerge.at(1).ptr<uchar>(j);
		uchar *data_guide_r = guideMerge.at(2).ptr<uchar>(j);

		for(int i=0; i<cols; i++){
			double temp1 = data_guide_b[i];
			double temp2 = data_guide_g[i];
			double temp3 = data_guide_r[i];

			if(temp1/255 < TRANSMISSIONRATIO || temp2/255 < TRANSMISSIONRATIO || temp3/255 < TRANSMISSIONRATIO){
				temp1 = TRANSMISSIONTHRESH;
				temp2 = TRANSMISSIONTHRESH;
				temp3 = TRANSMISSIONTHRESH;
			}
			int value_b = (int)255*(data_b[i] - A)/temp1 + A;
			int value_g = (int)255*(data_g[i] - A)/temp2 + A;
			int value_r = (int)255*(data_r[i] - A)/temp3 + A;
			data_r[i] = clip(value_r, 0, 255);
			data_b[i] = clip(value_b, 0, 255);
			data_g[i] = clip(value_g, 0, 255);
#endif
		}
	}
	merge(RGBimg, UnhazeImg);

	time0 = 1000*((double)getTickCount() - time0)/getTickFrequency();
	cout<<"getUnhazedImg run time: "<<time0<<"msec"<<endl;

// 	imshow("unhazeImg",UnhazeImg);
// 	waitKey();
}

void EnhanceImage::DarkChannelProcess(Mat hazeImg, Mat &unHazeImg)
{
	Mat rgbMinImg, darkChannel, transImg;
	Mat guideFiltImg;

	unsigned char Ac = 0;
	int r = RADIUS;
	double epst = 0.001;
	bool Engamma = false;//false;

	getDarkChannel(hazeImg, rgbMinImg, darkChannel);

	Ac = getAtmosphricLight(hazeImg, darkChannel);

	getTransmission(hazeImg, darkChannel, transImg, Ac);

	guideFilter(/*hazeImg*/rgbMinImg, transImg, guideFiltImg, r, epst);

	getUnhazedImg(hazeImg, guideFiltImg, unHazeImg, Ac);

	if(Engamma){
		GammaCorrect(unHazeImg, GAMMAVALUE);		//ÈôÊµŒÊÍŒÏñŽŠÀíœá¹ûÆ«°µ£¬¿É×ö
	}
}

void EnhanceImage::GammaCorrect(Mat &UnhazeImg, float fGamma)
{
	CV_Assert(UnhazeImg.depth() != sizeof(uchar));

	//dst.create(src.rows, src.cols, src.type());

	unsigned char lut[256];
	double time0 = static_cast<double>(getTickCount());
	int cols = UnhazeImg.cols;
	int rows = UnhazeImg.rows;

	//ÖžÊýÐÍÁÁ¶Èµ÷Õû
	for(int i=0; i<256; i++){
		lut[i] = pow((float)i/255,fGamma)*255.0f;
	}
	
	int channles = UnhazeImg.channels();
	switch(channles)
	{
		case 1:
			{
				for(int j=0; j<rows; j++){
					uchar *data = UnhazeImg.ptr<uchar>(j);
					//uchar *data_dst = dst.ptr<uchar>(j);
					for(int i=0; i<cols; i++){
						uchar tmpixel = data[i];
						data[i] = lut[tmpixel];
					}
				}

			}
		case 3:
			{
				for(int j=0; j<rows; j++){
					uchar *data = UnhazeImg.ptr<uchar>(j);
					//uchar *data_dst = dst.ptr<uchar>(j);
					for(int i=0; i<cols; i++){
						uchar tmpixel_b = data[i*3];
						uchar tmpixel_g = data[i*3+1];
						uchar tmpixel_r = data[i*3+2];

						data[i*3] = lut[tmpixel_b];
						data[i*3+1] = lut[tmpixel_g];
						data[i*3+2] = lut[tmpixel_r];
					}
				}
			}
			break;
	}

	time0 = 1000*((double)getTickCount() - time0)/getTickFrequency();
	cout<<"GammaCorrect run time: "<<time0<<"msec"<<endl;

// 	imshow("gamma",UnhazeImg);
// 	waitKey();
}

void EnhanceImage::hboxfilter_x(Mat id, Mat &od, int w, int h, int r)//(float *id, float *od, int w, int h, int r)
{
	od.create(h, w, CV_32FC1);

	float scale = 1.0f / (2*r+1);

	for (int y = 0; y < h; y++)
	{

		float t;
		// do left edge

		float *id_data = id.ptr<float>(y);
		float *od_data = od.ptr<float>(y);

		//t = id[y*w] * r;
		t = id_data[0] * r;

		for (int x = 0; x < r+1; x++)
		{
			//t += id[y*w+x];                //rÁÐºÍ
			t += id_data[x];
		}

		//od[y*w] = t * scale;              //±£ŽæµÚÒ»žöÖµ
		od_data[0] = t * scale;

		for (int x = 1; x < r+1; x++)
		{
			int c = x;//y*w+x;
			t += id_data[c+r];//id[c+r];
			t -= id_data[0];//id[y*w];
			od_data[c] = t * scale;//od[c] = t * scale;
		}

		// main loop
		for (int x = r+1; x < w-r; x++)
		{
			int c = x;//y*w+x;
			t += id_data[c+r];//id[c+r];
			t -= id_data[c-r-1];//id[c-r-1];
			od_data[c] = t * scale;//od[c] = t * scale;
		}

		// do right edge
		for (int x = w-r; x < w; x++)
		{
			int c = x;//y*w+x;
			t += id_data[w-1];//id[(y*w)+w-1];
			t -= id_data[c-r-1];//id[c-r-1];
			od_data[c] = t * scale;//od[c] = t * scale;
		}

	}
}

void EnhanceImage::hboxfilter_y(Mat id, Mat &od, int w, int h, int r)//(float *id, float *od, int w, int h, int r)
{
	od.create(h, w, CV_32FC1);

	float scale = 1.0f / (2*r+1);

	for (int x = 0; x < w; x++)
	{

		float t;

		// do left edge
		t = id.at<float>(0,x) * r;//id[x] * r;

		for (int y = 0; y < r+1; y++)
		{
			t += id.at<float>(y,x);//id[y*w+x];
		}

		od.at<float>(0,x) = t * scale;//od[x] = t * scale;

		for (int y = 1; y < r+1; y++)
		{
			int c = x;//y*w+x;
			t += id.at<float>(y+r,c);//t += id[c+r*w];
			t -= id.at<float>(0,x);//t -= id[x];
			od.at<float>(y,c) = t * scale;//od[c] = t * scale;
		}

		// main loop
		for (int y = r+1; y < h-r; y++)
		{
			int c = x;//y*w+x;
			t += id.at<float>(y+r,c);//t += id[c+r*w];
			t -= id.at<float>(y-r-1, c);//id[c-(r*w)-w];
			od.at<float>(y,c) = t * scale;//od[c] = t * scale;
		}

		// do right edge
		for (int y = h-r; y < h; y++)
		{
			int c = x;//y*w+x;
			t += id.at<float>(h-1, x);//t += [(h-1)*w+x];
			t -= id.at<float>(y-r-1, c);//id[c-(r*w)-w];
			od.at<float>(y,c) = t * scale;//od[c] = t * scale;
		}

	}
}

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
//! @param image      pointer to input data
//! @param temp       pointer to temporary store
//! @param w          width of image
//! @param h          height of image
//! @param r          radius of filter
////////////////////////////////////////////////////////////////////////////////

void EnhanceImage::computeGold(Mat image, Mat &dst, int w, int h, int r)//(float *image, float *temp, int w, int h, int r)
{
	Mat temp;
	r=r/2;
	hboxfilter_x(image, temp, w, h, r);
	hboxfilter_y(temp, dst, w, h, r);
}

void EnhanceImage::show(Mat src, Mat dst)
{
	if(src.type() == CV_32FC1){
		src.convertTo(dst, CV_8UC1, 255);
	}

	imshow("src", dst);
	waitKey();
}

/////////////////////////////////×Ô¶¯É«œ×/////////////////////////////////////////////////
void EnhanceImage::AutoColorLevelProcess(Mat srcImg, Mat &dstImg)
{
	Mat Hist;
	int channel;
	unsigned char minR, minG, minB;
	unsigned char maxR, maxG, maxB;

	unsigned char *mapR = NULL;
	unsigned char *mapG = NULL;
	unsigned char *mapB = NULL;

	mapR = new unsigned char[GRAYLEVEL];
	mapG = new unsigned char[GRAYLEVEL];
	mapB = new unsigned char[GRAYLEVEL];
	memset(mapR, 0, GRAYLEVEL*sizeof(unsigned char));
	memset(mapG, 0, GRAYLEVEL*sizeof(unsigned char));
	memset(mapB, 0, GRAYLEVEL*sizeof(unsigned char));

	channel = srcImg.channels();

	if(channel == 3){
		MinMax_Histgram(srcImg, minR, minG, minB, maxR, maxG, maxB);

		creatMap(mapR, mapG, mapB, minR, minG, minB, maxR, maxG, maxB);

		MapImage(srcImg, mapR, mapG, mapB, dstImg);
	}

	if(mapR!=NULL)
		delete []mapR;
	if(maxG!=NULL)
		delete []mapG;
	if(mapB!=NULL)
		delete []mapB;
}

void EnhanceImage::MinMax_Histgram(Mat src, unsigned char &minR, unsigned char &minG, unsigned char &minB,
	                     unsigned char &maxR, unsigned char &maxG, unsigned char &maxB)
{
	int cols, rows;
	unsigned char minvalue, maxvalue;
	unsigned char *HistR = NULL;
	unsigned char *HistG = NULL;
	unsigned char *HistB = NULL;

	cols = src.cols;
	rows = src.rows;
	HistR = new unsigned char[GRAYLEVEL];
	HistG = new unsigned char[GRAYLEVEL];
	HistB = new unsigned char[GRAYLEVEL];
	memset(HistR, 0, GRAYLEVEL*sizeof(unsigned char));
	memset(HistG, 0, GRAYLEVEL*sizeof(unsigned char));
	memset(HistB, 0, GRAYLEVEL*sizeof(unsigned char));

	for(int j=0; j<rows; j++){

		uchar *data = src.ptr<uchar>(j);

		for(int i=0; i<cols; i++){
			uchar pixel_B = data[i];
			uchar pixel_G = data[i+1];
			uchar pixel_R = data[i+2];

			HistR[pixel_R]++;
			HistG[pixel_G]++;
			HistB[pixel_B]++;
		}
	}

	MinMax_Hist(HistR, cols, rows, minR, maxR);
	MinMax_Hist(HistG, cols, rows, minG, maxG);
	MinMax_Hist(HistB, cols, rows, minB, maxB);

	if(HistR!=NULL)
		delete []HistR;
	if(HistG!=NULL)
		delete []HistG;
	if(HistB!=NULL)
		delete []HistB;
}

void EnhanceImage::MinMax_Hist(unsigned char *Hist, int width, int height, unsigned char &min, unsigned char &max)
{
	int pixelNum;
	float lowCut, highCut;
	float sum = 0;
	min = 0;
	max = 0;

	lowCut = LOWCUT;
	highCut = HIGHTCUT;
	pixelNum = width*height;

	for(int i=0; i<GRAYLEVEL; i++){
		sum += Hist[i];
		if(sum>=pixelNum*lowCut){
			min = i;
			break;
		}
	}
	sum = 0;

	for(int i=GRAYLEVEL-1; i>0; i--){
		sum += Hist[i];
		if(sum>=pixelNum*highCut){
			max = i;
			break;
		}
	}
}
void EnhanceImage::creatMap(unsigned char *MapR, unsigned char *MapG, unsigned char *MapB,
							unsigned char minR, unsigned char minG, unsigned char minB,
							unsigned char maxR, unsigned char maxG, unsigned char maxB)
{
    setMap(MapR, minR, maxR);
	setMap(MapG, minG, maxG);
	setMap(MapB, minB, maxB);
}

void EnhanceImage::setMap(unsigned char *Map, unsigned char min, unsigned char max)
{
	for(int i=0; i<GRAYLEVEL; i++)
	{
		if(i<=min){
			Map[i] = 0;//0//¿Éµ÷
		}
		else if(i>=max){
			Map[i] = 255;//255//¿Éµ÷
		}
		else{
			Map[i] = (uchar)255*(i-min)/(max-min);
		}
	}
}

void EnhanceImage::MapImage(Mat srcImg, unsigned char *MapR, unsigned char *MapG, unsigned char *MapB, Mat &dstImg)
{
	int cols, rows;

	cols = srcImg.cols;
	rows = srcImg.rows;
	dstImg.create(rows, cols, CV_8UC3);
	dstImg = Mat::zeros(rows, cols, CV_8UC3);

	for(int j=0; j<rows; j++){
		uchar *data_src = srcImg.ptr<uchar>(j);
		uchar *data_dst = dstImg.ptr<uchar>(j);

		for(int i=0; i<cols; i++){
			uchar pixel_b = data_src[i*3];
			uchar pixel_g = data_src[i*3+1];
			uchar pixel_r = data_src[i*3+2];

			data_dst[i*3] = MapB[pixel_b /*+ 1*/];//+1¿ÉÈ¥
			data_dst[i*3+1] = MapG[pixel_g /*+ 1*/];
			data_dst[i*3+2] = MapR[pixel_r /*+ 1*/];

			if(MapB[pixel_b]==0 && MapG[pixel_g]==0 &&  MapR[pixel_r]==0){

				int a =0;
			}
		}
	}

	imshow("dst",dstImg);
	waitKey();
}
