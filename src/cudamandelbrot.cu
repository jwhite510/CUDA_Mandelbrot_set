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
MandelBrotCuda::MandelBrotCuda() {
  cout<<"constructor running"<<endl;
  // copy an array to the server and back to host

  n = 20; // length of the array
  // allocate memory on the host
  h_arr = new double[n];
  for (int i = 0; i < n; i++) {
    h_arr[i] = i;
  }

  cudaMalloc(&d_arr, n*sizeof(double)); // allocate memory on the device
}
void MandelBrotCuda::gpu_calculate() {
  cout<<"gpu_calculate running"<<endl;

  // print before
  for (int i = 0; i < n; i++) {
    printf("%-5i ",(int)h_arr[i]);
  }cout<<endl;

  // copy to the device
  cudaMemcpy(d_arr, h_arr, n*sizeof(double), cudaMemcpyHostToDevice);

  int blocksize = 5;
  int numBlocks = 3;
  increment_values<<<numBlocks,blocksize>>>(d_arr, n);

  // copy to the host
  cudaMemcpy(h_arr, d_arr, n*sizeof(double), cudaMemcpyDeviceToHost);
  // print after
  for (int i = 0; i < n; i++) {
    printf("%-5i ",(int)h_arr[i]);
  }cout<<endl;
}
MandelBrotCuda::~MandelBrotCuda()  {
  delete [] h_arr;
  cudaFree(&d_arr);
  cout<<"destructor running"<<endl;
}

