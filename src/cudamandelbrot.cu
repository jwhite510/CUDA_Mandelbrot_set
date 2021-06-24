#include <SFML/Graphics.hpp>
#include <iostream>
#include <complex>
#include "cudamandelbrot.h"

using namespace std;

__global__
void increment_values(double* d_arr, int n) {
  int index = blockIdx.x*blockDim.x+threadIdx.x;
  int stride = blockDim.x*gridDim.x;
  for(int thread_i = index; thread_i < n; thread_i += stride) {
    d_arr[thread_i] = blockIdx.x;
    // d_arr[thread_i] = d_arr[thread_i] + 1;
  }
}
MandelBrotCuda::MandelBrotCuda(int W, int H):W(W),H(H) {
  cout<<"constructor running"<<endl;
  // copy an array to the server and back to host

  // allocate on device:

  // allocate memory on the host for result
  h_arr = new result[W*H];

  // space for x and y linspace, 2d grid for calculate result
  cudaMalloc(&d_x_arr, W * sizeof(double)); // allocate memory on the device
  cudaMalloc(&d_y_arr, H * sizeof(double)); // allocate memory on the device
  // for (int i = 0; i < n; i++) {
  //   h_arr[i] = i;
  // }
  cudaMalloc(&d_arr, W*H*sizeof(result)); // allocate memory on the device
}
void MandelBrotCuda::gpu_calculate(double *x, double *y) {
  // copy the x and y to device
  cudaMemcpy(d_x_arr, x, W * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y_arr, y, H * sizeof(double), cudaMemcpyHostToDevice);
  cout<<"gpu_calculate running"<<endl;

  int blocksize = 5;
  int numBlocks = 3;
  // increment_values<<<numBlocks,blocksize>>>(d_arr, W*H);
  // mandelbrot<<<numBlocks,blocksize>>>(d_arr, W, H, d_x_arr, d_y_arr);

  // copy to the host
  cudaMemcpy(h_arr, d_arr, W * H * sizeof(double), cudaMemcpyDeviceToHost);

  // run on cpu

}
MandelBrotCuda::~MandelBrotCuda()  {
  delete [] h_arr;
  cudaFree(&d_arr);
  cudaFree(&d_x_arr);
  cudaFree(&d_y_arr);
  cout<<"destructor running"<<endl;
}

