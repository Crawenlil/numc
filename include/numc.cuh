#ifndef NUMC_H
#define NUMC_H

#include <stdio.h>
#include <cuda.h>
#include <functional>
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

#define THREADS_PER_BLOCK_1D 256
#define THREADS_PER_BLOCK_2D 16


template <typename T>
class Matrix;

template <typename Op, typename T>
__host__
void apply(const Matrix<T> &x, const Matrix<T> &y, Matrix<T> &dest, Op op);

template<typename Op, typename T>
__global__
void operation_kernel(const typename Matrix<T>::MatrixGPU *x, 
                      const typename Matrix<T>::MatrixGPU *y,
                      typename Matrix<T>::MatrixGPU *dest,
                      Op op);

template<typename T>
class Add{
public:
    __device__  T operator() (T a, T b) const {return a + b;}
};

template<typename T>
class Sub{
public:
    __device__  T operator() (T a, T b) const {return a - b;}
};

template<typename T>
class Mul{
public:
    __device__  T operator() (T a, T b) const {return a * b;}
};

template<typename T>
class Div{
public:
    __device__  T operator() (T a, T b) const {return a - b;}
};

template <typename T>
class Matrix{
    private: 

        class MatrixGPU {
        private:
            T *elements;
            size_t *xDim;
            size_t *yDim;

        public:
            __host__ MatrixGPU();
            __host__ MatrixGPU(const MatrixGPU &other);
            __host__ MatrixGPU(const size_t xDim, const size_t yDim);
            __host__ ~MatrixGPU();
            __host__ __device__ size_t& getXDim() const;
            __host__ __device__ size_t& getYDim() const;
            __host__ __device__ MatrixGPU& operator=(const MatrixGPU &other);
            __host__ __device__ const T& operator()(size_t i, size_t j) const;
            __host__ __device__ T& operator()(size_t i, size_t j);

    };

    public: 
        MatrixGPU *matrixGPU;
        __host__ Matrix();
        __host__ Matrix(const Matrix<T> &other);
        __host__ Matrix(const size_t xDim, const size_t yDim);
        __host__ Matrix(const MatrixGPU &other);
        __host__ ~Matrix();
        __host__ __device__ size_t& getXDim() const;
        __host__ __device__ size_t& getYDim() const;
        __host__ __device__ Matrix<T>& operator=(const Matrix<T> &other);
        __host__ __device__ const T& operator()(size_t i, size_t j) const;
        __host__ __device__ T& operator()(size_t i, size_t j);
        __host__ Matrix<T> operator+(const Matrix& other) const;
        // __host__ Matrix<T> operator-(const Matrix& other) const;
        // __host__ Matrix<T> operator*(const Matrix& other) const;
        // __host__ Matrix<T> operator/(const Matrix& other) const;
};

///////////////
// MatrixGPU //
///////////////

template<typename T>
__host__
Matrix<T>::MatrixGPU::MatrixGPU(){
    printf("MatrixGPU constructor\n");
    cudaMallocManaged(&elements, sizeof(T));
    cudaMallocManaged(&xDim, sizeof(size_t));
    cudaMallocManaged(&yDim, sizeof(size_t));
    *xDim = 0;
    *yDim = 0;
}


template <typename T>
__host__  
Matrix<T>::MatrixGPU::MatrixGPU(const MatrixGPU &other){
    printf("Copy constructor, other value: %f\n", other(0,0));
    cudaMallocManaged(&elements, sizeof(T) * other.getXDim() * other.getYDim());
    cudaMallocManaged(&xDim, sizeof(size_t));
    cudaMallocManaged(&yDim, sizeof(size_t));

    *xDim = other.getXDim();
    *yDim = other.getYDim();
    for(size_t i = 0; i < getXDim() * getYDim(); ++i){
       elements[i] = other.elements[i];
    }
}


template <typename T>
__host__
Matrix<T>::MatrixGPU::MatrixGPU(const size_t _xDim, const size_t _yDim) {
    cudaMallocManaged(&elements, sizeof(T) * _xDim * _yDim);
    cudaMallocManaged(&xDim, sizeof(size_t));
    cudaMallocManaged(&yDim, sizeof(size_t));
    *xDim = _xDim;
    *yDim = _yDim;
}

template <typename T>
__host__
Matrix<T>::MatrixGPU::~MatrixGPU(){
    cudaFree(elements);
    cudaFree(xDim);
    cudaFree(yDim);
}

template <typename T>
__host__ __device__ 
size_t& Matrix<T>::MatrixGPU::getXDim() const{
    return *xDim;
}

template <typename T>
__host__ __device__ 
size_t& Matrix<T>::MatrixGPU::getYDim() const{
    return *yDim;
}


template<class T>
__host__ __device__
typename Matrix<T>::MatrixGPU& Matrix<T>::MatrixGPU::operator=(const typename Matrix<T>::MatrixGPU &other) {
    *xDim = other.getXDim();
    *yDim = other.getYDim();
    for(size_t i = 0; i < getXDim() * getYDim(); ++i){
       elements[i] = other.elements[i];
    }
    return *this;
}

template <typename T>
__host__ __device__
const T& Matrix<T>::MatrixGPU::operator()(size_t i, size_t j) const {
    return elements[(getXDim() * i) + j];
}

template <typename T>
__host__ __device__
T& Matrix<T>::MatrixGPU::operator()(size_t i, size_t j) {
    return elements[(getXDim() * i) + j];
}



////////////
// Matrix //
////////////

template<typename T>
__host__
Matrix<T>::Matrix(){
    printf("Matrix constructor\n");
    cudaMallocManaged(&matrixGPU, sizeof(MatrixGPU));
}


template <typename T>
__host__  
Matrix<T>::Matrix(const Matrix<T> &other){
    printf("Copy constructor, other value: %f\n", other(0,0));
    cudaMallocManaged(&matrixGPU, sizeof(MatrixGPU));
    *matrixGPU = *(other.matrixGPU);
}


template <typename T>
__host__
Matrix<T>::Matrix(const size_t _xDim, const size_t _yDim) {
    cudaMallocManaged(&matrixGPU, sizeof(MatrixGPU));
    *matrixGPU = MatrixGPU(_xDim, _yDim);
}

template <typename T>
__host__
Matrix<T>::Matrix(const MatrixGPU &_matrixGPU){
    cudaMallocManaged(&matrixGPU, sizeof(MatrixGPU));
    *matrixGPU = _matrixGPU;
}

template <typename T>
__host__
Matrix<T>::~Matrix(){
    cudaFree(matrixGPU);
}

template <typename T>
__host__
size_t& Matrix<T>::getXDim() const{
    return matrixGPU->getXDim();
}

template <typename T>
__host__
size_t& Matrix<T>::getYDim() const{
    return matrixGPU->getYDim();
}


template<class T>
__host__ 
Matrix<T>& Matrix<T>::operator=(const Matrix<T>& other) {
    *matrixGPU = *(other.matrixGPU);
    return *this;
}

template <typename T>
__host__
const T& Matrix<T>::operator()(size_t i, size_t j) const {
    return (*matrixGPU)(i, j);
}

template <typename T>
__host__ 
T& Matrix<T>::operator()(size_t i, size_t j) {
    return (*matrixGPU)(i, j);
}


template <typename T>
__host__
Matrix<T> Matrix<T>::operator+(const Matrix<T> &other) const {
    Matrix dest(getXDim(), getYDim());
    apply(*this, other, dest, Sub<T>());
    return dest;
}

// template <typename T>
// __host__
// Matrix<T> Matrix<T>::operator-(const Matrix &other) const {
//     Matrix dest(getXDim(), getYDim());
//     apply(*this, other, dest, Sub<T>());
//     return dest;
// }
// 
// template <typename T>
// __host__
// Matrix<T> Matrix<T>::operator*(const Matrix &other) const {
//     Matrix dest(getXDim(), getYDim());
//     apply(*this, other, dest, Mul<T>());
//     return dest;
// }
// 
// template <typename T>
// __host__
// Matrix<T> Matrix<T>::operator/(const Matrix &other) const {
//     Matrix dest(getXDim(), getYDim());
//     apply(*this, other, dest, Div<T>());
//     return dest;
// }

#endif // NUMC_H
