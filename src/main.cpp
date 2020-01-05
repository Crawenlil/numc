#include <iostream>
#include <numc.cuh>

int main(){
    Matrix<float> x(2, 4);
    Matrix<float> y(4, 2);
    Matrix<float> z;

    x(0,0) = 1.0;
    x(0,1) = 2.0;
    x(0,2) = 3.0;
    x(0,3) = 4.0;
    x(1,0) = 5.0;
    x(1,1) = 6.0;
    x(1,2) = 7.0;
    x(1,3) = 8.0;

    y(0,0) = 1.0;
    y(1,0) = 2.0;
    y(2,0) = 3.0;
    y(3,0) = 4.0;
    y(0,1) = 5.0;
    y(1,1) = 6.0;
    y(2,1) = 7.0;
    y(3,1) = 8.0;
    z = x.mm(y);
    std::cout << z << std::endl;

    const unsigned int nrows = 40;
    const unsigned int ncols = 40;
    Matrix<float> a(nrows, ncols);
    Matrix<float> b(nrows, ncols);
    Matrix<float> c;

    for (size_t i = 0; i < nrows; ++i){
        for (size_t j = 0; j < ncols; ++j){
            a(i, j) = 2.0f;
            b(i, j) = 5.0f;
        }
    }

    c = a + b;
    std::cout << c << std::endl; 
    return 0;
} 
