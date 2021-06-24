#ifndef CUDAMANDELBROT_H
#define CUDAMANDELBROT_H

class MandelBrotCuda {
  public:

    double* h_arr;
    double* d_arr;
    int n;
    MandelBrotCuda();
    void gpu_calculate();
    ~MandelBrotCuda();
};

#endif

