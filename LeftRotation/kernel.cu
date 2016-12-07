//Поворот матрицы изображения, хранящегося по строчкам, на 90 градусов влево

/*
Оценка производительности rotate:
Чтений Кэш-линеек: N^2
Количество считываемых байт: 128*N^2 Bytes
Кол-во операций: 12*N^2 ops
Арифметическая интенсивность: 128*N^2 Bytes / 12*N^2 ops = 10.67 B/op
Пиковая производительность GeForce GTX 550 Ti: 691.2 GFLOPS
С какой скоростью надо считывать данные для достижения пиковой производительности: GMAC * Peak FLOPS = 10.67 * 691.2 = 7.2 TB/s
Реальная пропускная способность памяти GeForce GTX 550 Ti: 98.4 GB/s
Получится производительность: Actual BW / GMAC = 98.4 / 10.67 = 9.2 GFLOPS

Оценка производительности frotate(TILE_WIDTH = 128):
Чтений Кэш-линеек: N*(3*3*N/TILE_WIDTH) = N*(9*N/128)
Количество считываемых байт: 128*N*(9*N/128)  Bytes = 9*N^2 Bytes
Кол-во операций: 15*N^2 ops
Арифметическая интенсивность: 9*N^2 Bytes / 15*N^2 ops = 0.6 B/op
Пиковая производительность GeForce GTX 550 Ti: 691.2 GFLOPS
С какой скоростью надо считывать данные для достижения пиковой производительности: GMAC * Peak FLOPS = 0.6 * 691.2 = 414.72 GB/s
Реальная пропускная способность памяти GeForce GTX 550 Ti: 98.4 GB/s
Получится производительность: Actual BW / GMAC = 98.4 / 0.6 = 164 GFLOPS

Сравнение алгоритмов:  Простой            Оптимизированный
Чтений Кэш-линеек        N^2              N*(9*N/TILE_WIDTH)
Производительность    9.2 GFLOPS             164 GFLOPS
SLOCs                    15                     18
Ускорение                1х                    17.8х
Ускорение/SLOCs          1х                    14.8х

На изображение 4096*4096 средняя время выполнения на GeForce GTX 550 Ti:
						93.2 мс                7.4 мс
*/

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <ctime>

using namespace cv;
using namespace std;

#define TILE_WIDTH 128
#define GRID_SIZE 1024
#define IM_WIDTH 4096
#define IM_HEIGHT 4096

#define CHECK(value) {											\
	cudaError_t _m_cudaStat = value;										\
	if (_m_cudaStat != cudaSuccess) {										\
		cout<< "Error:" << cudaGetErrorString(_m_cudaStat) \
			<< " at line " << __LINE__ << " in file " << __FILE__ << "\n"; \
		exit(1);															\
	} }



__global__ void rotate(uchar *a, uchar *r, int n, int m)
{
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	int threadsNum = blockDim.x*gridDim.x;
	for (int i = id; i < n*m; i += threadsNum)
	{
		int cur_str = i / m;
		int cur_col = i % m;
		int new_i = 3 * ((m - 1 - cur_col)*n + cur_str);
		int i3 = 3 * i;
		r[new_i] = a[i3];
		r[new_i + 1] = a[i3 + 1];
		r[new_i + 2] = a[i3 + 2];
	}
}



__global__ void frotate(uchar *a, uchar *r, int n, int m)
{
	int tx = threadIdx.x;
	int dx = blockIdx.x;
	int off = dx / (m / TILE_WIDTH) * m * 3 * TILE_WIDTH + dx % (m / TILE_WIDTH) * 3 * TILE_WIDTH;
	int cur_str = 3 * (TILE_WIDTH*(dx / (m / TILE_WIDTH)));
	int cur_col = TILE_WIDTH*(dx % (m / TILE_WIDTH));

	__shared__ uchar s_a[TILE_WIDTH][TILE_WIDTH + 1];

	for (int j = 0; j < 3; ++j)
	{
		for (int i = 0; i < TILE_WIDTH; ++i)//i-row
			s_a[i][tx] = a[3 * i*m + 3 * tx + off + j];
		__syncthreads();
		for (int i = 0; i < TILE_WIDTH; ++i)
			r[3 * (m - 1 - cur_col - i)*n + cur_str + 3 * tx + j] = s_a[tx][i];
		__syncthreads();
	}
}



int main(void)
{
	//по случайному изображению
	srand(time(0)); // автоматическая рандомизация
	int n = IM_HEIGHT;
	int m = IM_WIDTH;
	int im_l = 3 * n*m;
	uchar *host_a = new uchar[im_l];
	uchar *host_r = new uchar[im_l];

	for (int i = 0; i<im_l; ++i)
		host_a[i] = rand() % 256;
	Mat image(Size(n, m), CV_8UC3);
	Mat img_res(Size(n, m), CV_8UC3);
	memcpy(image.data, host_a, im_l * sizeof(uchar));
	imwrite("pic1.jpg", image);


	//по загруженному изображению
	//Mat image, image2;
	//    image = imread("pic.jpg", CV_LOAD_IMAGE_COLOR);   // Read the file
	//    if(! image.data )                              // Check for invalid input
	//    {
	//        cout <<  "Could not open or find the image" << std::endl ;
	//        return -1;
	//    }
	//int n = image.rows;
	//int m = image.cols;
	//int im_l = 3*n*m;
	//Mat img_res(Size(n,m),CV_8UC3);
	//image2 = image.clone();
	//uchar *host_a = image2.data;
	//uchar *host_r = new uchar[im_l];

	uchar *dev_a, *dev_r, *host_r_check;

	cudaEvent_t startCUDA, stopCUDA;
	clock_t startCPU;
	float elapsedTimeCUDA, elapsedTimeCPU;

	cudaEventCreate(&startCUDA);
	cudaEventCreate(&stopCUDA);


	//CPU rotate
	startCPU = clock();
	for (int i = 0; i < n*m; i++)
	{
		int cur_str = i / m;
		int cur_col = i % m;
		int new_i = 3 * ((m - 1 - cur_col)*n + cur_str);
		host_r[new_i] = host_a[3 * i];
		host_r[new_i + 1] = host_a[3 * i + 1];
		host_r[new_i + 2] = host_a[3 * i + 2];
	}
	elapsedTimeCPU = (double)(clock() - startCPU) / CLOCKS_PER_SEC;
	cout << "CPU sum time = " << elapsedTimeCPU * 1000 << " ms\n";
	cout << "CPU memory throughput = " << im_l * sizeof(uchar) / elapsedTimeCPU / 1024 / 1024 / 1024 << " Gb/s\n" << endl;



	//rotate
	CHECK(cudaMalloc(&dev_a, im_l * sizeof(uchar)));
	CHECK(cudaMemcpy(dev_a, image.data, im_l * sizeof(uchar), cudaMemcpyHostToDevice));
	CHECK(cudaMalloc(&dev_r, im_l * sizeof(uchar)));

	cudaEventRecord(startCUDA, 0);

	rotate << <1024, 1024 >> >(dev_a, dev_r, n, m);

	cudaEventRecord(stopCUDA, 0);
	cudaEventSynchronize(stopCUDA);
	CHECK(cudaGetLastError());

	cudaEventElapsedTime(&elapsedTimeCUDA, startCUDA, stopCUDA);

	cout << "CUDA rotate sum time = " << elapsedTimeCUDA << " ms\n";
	cout << "CUDA rotate memory throughput = " << im_l * sizeof(uchar) / elapsedTimeCUDA / 1024 / 1024 / 1.024 << " Gb/s\n" << endl;

	CHECK(cudaMemcpy(img_res.data, dev_r, im_l * sizeof(uchar), cudaMemcpyDeviceToHost));

	// check
	host_r_check = img_res.data;
	for (int i = 0; i < im_l; i++)
		if (host_r[i] != host_r_check[i])
		{
			cout << "Error in element N " << i << ": r[i] = " << host_r[i]
				<< " r_check[i] = " << (int)host_r_check[i] << "\n";
			exit(1);
		}



	//frotate
	CHECK(cudaMemcpy(dev_a, image.data, im_l * sizeof(uchar), cudaMemcpyHostToDevice));
	cudaEventRecord(startCUDA, 0);

	frotate << <GRID_SIZE, TILE_WIDTH >> >(dev_a, dev_r, n, m);

	cudaEventRecord(stopCUDA, 0);
	cudaEventSynchronize(stopCUDA);
	CHECK(cudaGetLastError());

	cudaEventElapsedTime(&elapsedTimeCUDA, startCUDA, stopCUDA);

	cout << "CUDA frotate sum time = " << elapsedTimeCUDA << " ms\n";
	cout << "CUDA frotate memory throughput = " << im_l * sizeof(uchar) / elapsedTimeCUDA / 1024 / 1024 / 1.024 << " Gb/s\n" << endl;

	CHECK(cudaMemcpy(img_res.data, dev_r, im_l * sizeof(uchar), cudaMemcpyDeviceToHost));

	// check
	host_r_check = img_res.data;
	for (int i = 0; i < im_l; i++)
		if (host_r[i] != host_r_check[i])
		{
			cout << "Error in element N " << i << ": r[i] = " << host_r[i]
				<< " r_check[i] = " << (int)host_r_check[i] << "\n";
			exit(1);
		}



	imwrite("pic2.jpg", img_res);

	CHECK(cudaFree(dev_a));
	CHECK(cudaFree(dev_r));
	
	system("pause");
	
	return 0;
}
