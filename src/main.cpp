#include <SFML/Graphics.hpp>
#include <iostream>
#include <complex>

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
void Linspace(double* arr, double begin, double end, int N)
{
  double dx = (end - begin) / N;
  double value = begin;
  for(int i=0; i < N; i++) {
    arr[i] = value;
    value+=dx;
  }
}
struct PixelGrid
{
  sf::Uint8* pixels;
  int W;
  int H;
  PixelGrid(int W_in, int H_in)
  {
    W = W_in;
    H = H_in;
    pixels = new sf::Uint8[W*H*4];
  }
  inline sf::Uint8 operator() (int row, int col, int color) const
  {
    return pixels[4*W*row + 4*col + color];
  }
  inline sf::Uint8& operator() (int row, int col, int color)
  {
    return pixels[4*W*row + 4*col + color];
  }
  ~PixelGrid()
  {
    cout << "PixelGrid destructor called" << endl;
    delete [] pixels;
  }

};
struct Application
{
  sf::RenderWindow* window;
  PixelGrid* pixelgrid;
  unsigned int W;
  unsigned int H;
  sf::Texture* texture;
  sf::Sprite* sprite;
  int mouseDragging = 0;
  double center_x;
  int center_col_delta;
  double center_y;
  int center_row_delta;
  int mouse_down_col;
  int mouse_down_row;
  double x_span;
  double* x;
  double* y;
  double delta_linspace_x;
  double delta_linspace_y;

  Application()
  {
    W = 600;
    H = 600; // you can change this to full window size later

    delta_linspace_x = 0;
    delta_linspace_y = 0;

    x = new double[W];
    y = new double[H];
    x_span = 3;
    center_x = 0;
    center_y = 0;

    // sf::RenderWindow window(sf::VideoMode(W, H), "SFML works!");
    window = new sf::RenderWindow(sf::VideoMode(W, H), "Mandelbrot Set");

    // sf::Uint8* pixels = new sf::Uint8[W*H*4];
    pixelgrid = new PixelGrid(W,H);

    texture = new sf::Texture;
    texture->create(W, H);
    sprite = new sf::Sprite(*texture);

  }
  void CaptureEvents()
  {
    sf::Event event;
    while (window->pollEvent(event))
    {
      if (event.type == sf::Event::Closed)
        window->close();
      if(event.type == sf::Event::MouseButtonPressed) {
        cout << "mouse pressed" << endl;
        mouseDragging = 1;
        // set the mouse clicked location
        sf::Vector2i position = sf::Mouse::getPosition();
        mouse_down_col = position.x;
        mouse_down_row = position.y;
      }
      if(event.type == sf::Event::MouseButtonReleased) {
        mouseDragging = 0;
        cout << "mouse released" << endl;
        center_x += delta_linspace_x;
        center_y += delta_linspace_y;
        delta_linspace_x = 0;
        delta_linspace_y = 0;
      }
      if(event.type == sf::Event::MouseWheelScrolled) {
        cout << "mouse scrolling" << endl;
        cout << event.mouseWheelScroll.delta << endl;
        double deltazoom = ((double)event.mouseWheelScroll.delta / 10);
        double zoomlevel = 1 + deltazoom;
        std::cout << "deltazoom" << " => " << deltazoom << std::endl;
        x_span *= zoomlevel;
      }
    }
    if(mouseDragging) {
      sf::Vector2i position = sf::Mouse::getPosition();
      int delta_col = position.x - mouse_down_col;
      int delta_row = position.y - mouse_down_row;

      delta_linspace_x = x_span * ((double)delta_row / (double)W);
      delta_linspace_y = x_span * ((double)delta_col / (double)H);
    }
  }
  void DrawCoordinateSpace()
  {
    Linspace(x, -(x_span/2)-delta_linspace_x-center_x, (x_span/2)-delta_linspace_x-center_x, W);
    Linspace(y, -(x_span/2)-delta_linspace_y-center_y, (x_span/2)-delta_linspace_y-center_y, H);
  }
  void DrawPixels()
  {
      texture->update((*pixelgrid).pixels);
      window->clear();
      window->draw(*sprite);
      window->display();
  }
  ~Application()
  {
    std::cout << "application destructor called" << std::endl;
    delete [] x;
    delete [] y;
    delete window;
    delete pixelgrid;
    delete texture;
    delete sprite;
  }
};


int main()
{
  Application app;
  while(app.window->isOpen())
  {
    // capture mouse events
    app.CaptureEvents();

    // update x and y arrays
    app.DrawCoordinateSpace();

    for(int i=0; i < app.W; i++)
      for(int j=0; j < app.H; j++) {

        // calculate divergence for mandelbrot set
        int iterations;
        double real;
        double imag;
        mandelbrot(app.x[i], app.y[j],
            iterations, // OUT
            real, // OUT
            imag); // OUT

        (*app.pixelgrid)(i,j,0) = real;
        (*app.pixelgrid)(i,j,1) = imag+iterations;
        (*app.pixelgrid)(i,j,2) = iterations;
        (*app.pixelgrid)(i,j,3) = 255;
      }
    // draw on screen
    app.DrawPixels();

  }





  return 0;
}
