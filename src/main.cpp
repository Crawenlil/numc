#include <iostream>
#include "numc.cuh"

int main(){

    const unsigned int xDim = 1000;
    const unsigned int yDim = 10;

    Matrix<float> x(xDim, yDim);
    Matrix<float> y(xDim, yDim);
    Matrix<float> z(xDim, yDim);

    for (size_t i = 0; i < xDim; ++i){
        for (size_t j = 0; j < yDim; ++j){
            x(i, j) = 2.0f;
            y(i, j) = 3.0f;
        }
    }
    z = x + y;
    std::cout << x(xDim-1, yDim-1) << " + " << y(xDim-1, yDim-1) << " = "<< z(xDim-1, yDim-1) << std::endl;
    z = x - y;
    std::cout << x(xDim-1, yDim-1) << " - " << y(xDim-1, yDim-1) << " = "<< z(xDim-1, yDim-1) << std::endl;
    z = x * y;
    std::cout << x(xDim-1, yDim-1) << " * " << y(xDim-1, yDim-1) << " = "<< z(xDim-1, yDim-1) << std::endl;
    z = x / y;
    std::cout << x(xDim-1, yDim-1) << " / " << y(xDim-1, yDim-1) << " = "<< z(xDim-1, yDim-1) << std::endl;

    return 0;
}
