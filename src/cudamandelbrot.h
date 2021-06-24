#ifndef CUDAMANDELBROT_H
#define CUDAMANDELBROT_H

struct result {
  int iterations;
  double real;
  double imag;
};

class MandelBrotCuda {
  public:

    result* h_arr;
    result* d_arr;
    double* d_x_arr;
    double* d_y_arr;
    int W;
    int H;
    MandelBrotCuda(int W, int H);
    void gpu_calculate(double* x, double* y);
    ~MandelBrotCuda();
};

#endif

