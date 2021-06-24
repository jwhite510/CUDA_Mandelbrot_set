#include <SFML/Graphics.hpp>
#include <iostream>
#include <complex>
#include "cudamandelbrot.h"

using namespace std;

void mandelbrot(double c_real, double c_imaginary, int &iterations, double &real, double &imag)
{

  const int num_iterations = 400;
  const double max_radius = 10000;


  complex<double> z = complex<double>(0,0);
  complex<double> z_next = complex<double>(0,0);
  const complex<double> c = complex<double>(c_real, c_imaginary);

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
  // cout<<"gpu_calculate running"<<endl;
  // for(int i=0; i < 10; i++){
  //   cout<<x[i]<<" ";
  // }cout<<endl;

  int blocksize = 5;
  int numBlocks = 3;
  // increment_values<<<numBlocks,blocksize>>>(d_arr, W*H);
  // mandelbrot<<<numBlocks,blocksize>>>(d_arr, W, H, d_x_arr, d_y_arr);


  // copy to the host
  cudaMemcpy(h_arr, d_arr, W * H * sizeof(double), cudaMemcpyDeviceToHost);

  // run on cpu
  cout<<"start"<<endl;
  for (int i = 0; i < W; i++) {
    for (int j = 0; j < H; j++) {
      int iterations;
      double real;
      double imag;
      mandelbrot(x[i], y[j],
          iterations, // OUT
          real, // OUT
          imag); // OUT
      h_arr[i * H + j].iterations = iterations;
      h_arr[i * H + j].real = real;
      h_arr[i * H + j].imag = imag;
    }
  }
  cout<<"finish"<<endl;



}
MandelBrotCuda::~MandelBrotCuda()  {
  delete [] h_arr;
  cudaFree(&d_arr);
  cudaFree(&d_x_arr);
  cudaFree(&d_y_arr);
  cout<<"destructor running"<<endl;
}

