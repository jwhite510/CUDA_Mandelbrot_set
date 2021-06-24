#include <SFML/Graphics.hpp>
#include <iostream>
#include <complex>

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

class MandelBrotCuda {
  public:
  MandelBrotCuda() {
    cout<<"hello main is running"<<endl;
    // copy an array to the server and back to host

    int n = 20; // length of the array
    // allocate memory on the host
    double *h_arr = new double[n];
    for (int i = 0; i < n; i++) {
      h_arr[i] = i;
    }

    // print before
    for (int i = 0; i < n; i++) {
      printf("%-5i ",(int)h_arr[i]);
    }cout<<endl;

    double *d_arr; // allocate on device
    cudaMalloc(&d_arr, n*sizeof(double)); // allocate memory on the device

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

    delete [] h_arr;
    cudaFree(&d_arr);
  }
};
