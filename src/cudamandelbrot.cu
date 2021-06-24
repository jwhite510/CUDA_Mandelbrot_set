#include <SFML/Graphics.hpp>
#include <iostream>
#include <complex>
#include "cudamandelbrot.h"
#include <thrust/complex.h>
#include "params.h"

using namespace std;

void mandelbrot(mtype c_real, mtype c_imaginary, int &iterations, mtype &real, mtype &imag)
{

  const int num_iterations = 400;
  const mtype max_radius = 10000;


  complex<mtype> z = complex<mtype>(0,0);
  complex<mtype> z_next = complex<mtype>(0,0);
  const complex<mtype> c = complex<mtype>(c_real, c_imaginary);

  iterations = 0;
  while(iterations < num_iterations) {
    z_next = pow(z,2) + c;
    if(abs(z_next) > max_radius)
      break;
    z = z_next;
    iterations++;
  }

  // decrease error
  for(int e=0; e < 4; e++)
    z = pow(z,2) + c;

  real = z.real();
  imag = z.imag();

}

__global__
void increment_values(mtype* d_arr, int n) {
  int index = blockIdx.x*blockDim.x+threadIdx.x;
  int stride = blockDim.x*gridDim.x;
  for(int thread_i = index; thread_i < n; thread_i += stride) {
    d_arr[thread_i] = blockIdx.x;
    // d_arr[thread_i] = d_arr[thread_i] + 1;
  }
}
__global__
void mandelbrotgpu(result *d_arr, int W, int H, mtype *d_x_arr, mtype *d_y_arr) {
  int index = blockIdx.x*blockDim.x+threadIdx.x;
  int stride = blockDim.x*gridDim.x;
  for(int thread_i = index; thread_i < W * H; thread_i += stride) {
    // d_arr[thread_i]. = blockIdx.x;
    // unravel index
    int i = thread_i / H;
    int j = thread_i % H;
    mtype c_real = d_x_arr[i];
    mtype c_imaginary = d_y_arr[j];

    const int num_iterations = 400;
    const mtype max_radius = 10000;

    // complex<mtype> z = complex<mtype>(0,0);
    // complex<mtype> z_next = complex<mtype>(0,0);
    // const complex<mtype> c = complex<mtype>(c_real, c_imaginary);
    thrust::complex<float> z = thrust::complex<float>(0,0);
    thrust::complex<float> z_next = thrust::complex<float>(0,0);
    const thrust::complex<float> c = thrust::complex<float>(c_real,c_imaginary);
    d_arr[i * H + j].iterations = 0;
    while(d_arr[i * H + j].iterations < num_iterations) {
      z_next = pow(z,2) + c;
      if (abs(z_next) > max_radius) { break; }
      z = z_next;
      d_arr[i * H + j].iterations++;
    }
    // decrease error
    for(int e=0; e < 4; e++) { z = pow(z,2) + c; }
    d_arr[i * H + j].real = z.real();
    d_arr[i * H + j].imag = z.imag();
    // d_arr[i * H + j] = thread_i;
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
  cudaMalloc(&d_x_arr, W * sizeof(mtype)); // allocate memory on the device
  cudaMalloc(&d_y_arr, H * sizeof(mtype)); // allocate memory on the device
  // for (int i = 0; i < n; i++) {
  //   h_arr[i] = i;
  // }
  cudaMalloc(&d_arr, W*H*sizeof(result)); // allocate memory on the device
}
void MandelBrotCuda::gpu_calculate(mtype *x, mtype *y) {
  // copy the x and y to device
  cudaMemcpy(d_x_arr, x, W * sizeof(mtype), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y_arr, y, H * sizeof(mtype), cudaMemcpyHostToDevice);
  // cout<<"gpu_calculate running"<<endl;
  // for(int i=0; i < 10; i++){
  //   cout<<x[i]<<" ";
  // }cout<<endl;

  int blocksize = 1024;
  int numBlocks = 500;
  // increment_values<<<numBlocks,blocksize>>>(d_arr, W*H);
  cout<<"start gpu"<<endl;
  mandelbrotgpu<<<numBlocks,blocksize>>>(d_arr, W, H, d_x_arr, d_y_arr);
  cout<<"end gpu"<<endl;

  // copy to the host
  cudaMemcpy(h_arr, d_arr, W * H * sizeof(result), cudaMemcpyDeviceToHost);

}
MandelBrotCuda::~MandelBrotCuda()  {
  delete [] h_arr;
  cudaFree(&d_arr);
  cudaFree(&d_x_arr);
  cudaFree(&d_y_arr);
  cout<<"destructor running"<<endl;
}

