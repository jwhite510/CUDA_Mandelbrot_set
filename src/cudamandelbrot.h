#include "params.h"
#ifndef CUDAMANDELBROT_H
#define CUDAMANDELBROT_H

struct result {
  int iterations;
  mtype real;
  mtype imag;
};

class MandelBrotCuda {
  public:

    result* h_arr;
    result* d_arr;
    mtype* d_x_arr;
    mtype* d_y_arr;
    int W;
    int H;
    MandelBrotCuda(int W, int H);
    void gpu_calculate(mtype* x, mtype* y);
    ~MandelBrotCuda();
};

#endif

