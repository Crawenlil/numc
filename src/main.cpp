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

    // const unsigned int nrows = 100;
    // const unsigned int ncols = 100;
    // Matrix<float> x(nrows, ncols);
    // Matrix<float> y(nrows, ncols);
    // Matrix<float> z;

    // for (size_t i = 0; i < nrows; ++i){
    //     for (size_t j = 0; j < ncols; ++j){
    //         x(i, j) = 3.0f;
    //         y(i, j) = 2.0f;
    //     }
    // }

    // z = x + y;
    // std::cout << x(nrows-1, ncols-1) << " + " << y(nrows-1, ncols-1) << " = "<< z(nrows-1, ncols-1) << std::endl;
    // z = x - y;
    // std::cout << x(nrows-1, ncols-1) << " - " << y(nrows-1, ncols-1) << " = "<< z(nrows-1, ncols-1) << std::endl;
    // z = x * y;
    // std::cout << x(nrows-1, ncols-1) << " * " << y(nrows-1, ncols-1) << " = "<< z(nrows-1, ncols-1) << std::endl;
    // z = x / y;
    // std::cout << x(nrows-1, ncols-1) << " / " << y(nrows-1, ncols-1) << " = "<< z(nrows-1, ncols-1) << std::endl;
    z = x.mm(y);
    for (int i = 0; i < z.getRows(); ++i){
        for (int j = 0; j < z.getCols(); ++j){
            std::cout << "z(" << i << ", " << j << ") = " << z(i,j) << std::endl;
        }
    }

    return 0;
} 
