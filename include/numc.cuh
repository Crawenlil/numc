#ifndef NUMC_H
#define NUMC_H

#include <stdio.h>
#include <cuda.h>
#include <functional>
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

// #define DEBUG

#define THREADS_PER_BLOCK_1D 256
#define THREADS_PER_BLOCK_2D 16

#define INSTANTIATE_operation_kernel(TYPE, OP) \
    template __global__ void operation_kernel<OP<TYPE>, TYPE>(\
            const MatrixGPU<TYPE> &,\
            const MatrixGPU<TYPE> &,\
            MatrixGPU<TYPE> &, OP<TYPE>);


template <typename T>
class MatrixGPU;

template <typename T>
class Matrix;

template <typename Op, typename T>
__host__
void apply(const Matrix<T> &x, const Matrix<T> &y, Matrix<T> &dest, Op op);

template<typename Op, typename T>
__global__
void operation_kernel(const MatrixGPU<T> *x, 
                      const MatrixGPU<T> *y,
                      MatrixGPU<T> *dest,
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
    __device__  T operator() (T a, T b) const {return a / b;}
};

class Managed {
    public:
        void *operator new(size_t len);
        void operator delete(void *ptr);
};

template <typename T>
class MatrixGPU: public Managed {
    private:
        T *elements;
        size_t *xDim;
        size_t *yDim;
    
    public:
        // constructors
        __host__ MatrixGPU();
        __host__ MatrixGPU(const size_t xDim, const size_t yDim);
        // destructor
        __host__ ~MatrixGPU();
        // copy
        __host__ MatrixGPU(const MatrixGPU<T> &other);
        // assignment
        __host__ MatrixGPU<T>& operator=(const MatrixGPU<T> &other);
        // functions
        __host__ __device__ size_t& getXDim() const;
        __host__ __device__ size_t& getYDim() const;
        __host__ __device__ const T& operator()(size_t i, size_t j) const;
        __host__ __device__ T& operator()(size_t i, size_t j);
};

template <typename T>
class Matrix{
    public:
        //fields
        MatrixGPU<T> * matrixGPU;
        // constructors
        Matrix();
        Matrix(const MatrixGPU<T> &_matrixGPU);
        Matrix(const size_t xDim, const size_t yDim);
        // destructor
        ~Matrix();
        // copy
        Matrix(const Matrix<T> &other);
        // assignment
        Matrix<T>& operator=(const Matrix<T> &other);
        // functions
        size_t& getXDim() const;
        size_t& getYDim() const;
        const T& operator()(size_t i, size_t j) const;
        T& operator()(size_t i, size_t j);
        Matrix<T> operator+(const Matrix& other) const;
        Matrix<T> operator-(const Matrix& other) const;
        Matrix<T> operator*(const Matrix& other) const;
        Matrix<T> operator/(const Matrix& other) const;
};
///////////////
// MatrixGPU //
///////////////

template<typename T>
__host__
MatrixGPU<T>::MatrixGPU(){
#ifdef DEBUG
    printf("MatrixGPU() constructor\n");
#endif
    cudaMallocManaged(&elements, sizeof(T));
    cudaMallocManaged(&xDim, sizeof(size_t));
    cudaMallocManaged(&yDim, sizeof(size_t));
    *xDim = 0;
    *yDim = 0;
}



template <typename T>
__host__
MatrixGPU<T>::MatrixGPU(const size_t _xDim, const size_t _yDim) {
#ifdef DEBUG
    printf("MatrixGPU(xDim=%d, yDim=%d) constructor\n", _xDim, _yDim);
#endif
    cudaMallocManaged(&elements, sizeof(T) * _xDim * _yDim);
    cudaMallocManaged(&xDim, sizeof(size_t));
    cudaMallocManaged(&yDim, sizeof(size_t));
    *xDim = _xDim;
    *yDim = _yDim;
}

template <typename T>
__host__
MatrixGPU<T>::~MatrixGPU(){
#ifdef DEBUG
    printf("~MatrixGPU\n");
#endif
    cudaFree(elements);
    cudaFree(xDim);
    cudaFree(yDim);
}

template <typename T>
__host__  
MatrixGPU<T>::MatrixGPU(const MatrixGPU &other){
#ifdef DEBUG
    printf("MatrixGPU copy constructor\n");
#endif
    cudaMallocManaged(&elements, sizeof(T) * other.getXDim() * other.getYDim());
    cudaMallocManaged(&xDim, sizeof(size_t));
    cudaMallocManaged(&yDim, sizeof(size_t));

    *xDim = other.getXDim();
    *yDim = other.getYDim();
    for(size_t i = 0; i < getXDim() * getYDim(); ++i){
       elements[i] = other.elements[i];
    }
}

template<class T>
__host__
MatrixGPU<T>& MatrixGPU<T>::operator=(const MatrixGPU<T> &other) {
    if (this != &other){
        *xDim = other.getXDim();
        *yDim = other.getYDim();
        for(size_t i = 0; i < getXDim() * getYDim(); ++i){
           elements[i] = other.elements[i];
        }
    }
    return *this;
}

template <typename T>
__host__ __device__ 
size_t& MatrixGPU<T>::getXDim() const{
    return *xDim;
}

template <typename T>
__host__ __device__ 
size_t& MatrixGPU<T>::getYDim() const{
    return *yDim;
}


template <typename T>
__host__ __device__
const T& MatrixGPU<T>::operator()(size_t i, size_t j) const {
    return elements[(getYDim() * i) + j];
}

template <typename T>
__host__ __device__
T& MatrixGPU<T>::operator()(size_t i, size_t j) {
    return elements[(getYDim() * i) + j];
}

////////////
// Matrix //
////////////

template<typename T>
Matrix<T>::Matrix(){
#ifdef DEBUG
    printf("Matrix() constructor\n");
#endif
    matrixGPU = new MatrixGPU<T>;
}

template<typename T>
Matrix<T>::Matrix(const MatrixGPU<T> &_matrixGPU) {
#ifdef DEBUG
    printf("Matrix(const MatrixGPU<T> &_matrixGPU) constructor\n");
#endif
    matrixGPU = new MatrixGPU<T>;
    *matrixGPU = _matrixGPU;
}

template <typename T>
Matrix<T>::Matrix(const size_t _xDim, const size_t _yDim) {
#ifdef DEBUG
    printf("Matrix(xDim, yDim) constructor\n");
#endif
    matrixGPU = new MatrixGPU<T>(_xDim, _yDim);
}

template <typename T>
Matrix<T>::~Matrix(){
#ifdef DEBUG
    printf("~Matrix\n");
#endif
    delete matrixGPU;
}

template <typename T>
Matrix<T>::Matrix(const Matrix &other){
#ifdef DEBUG
    printf("Matrix copy constructor\n");
#endif
    if (this != &other){
        matrixGPU = new MatrixGPU<T>;
        *matrixGPU = *(other.matrixGPU);
    }
}

template<class T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T> &other) {
#ifdef DEBUG
    printf("Matrix assignment\n");
#endif
    if (this != &other) {
        delete matrixGPU;
        matrixGPU = new MatrixGPU<T>(other.getXDim(), other.getYDim());
        *matrixGPU = *(other.matrixGPU);
    }
    return *this;
}

template <typename T>
size_t& Matrix<T>::getXDim() const{
    return matrixGPU->getXDim();
}

template <typename T>
size_t& Matrix<T>::getYDim() const{
    return matrixGPU->getYDim();
}

template <typename T>
const T& Matrix<T>::operator()(size_t i, size_t j) const {
    return (*matrixGPU)(i, j);
}

template <typename T>
T& Matrix<T>::operator()(size_t i, size_t j) {
    return (*matrixGPU)(i, j);
}


template <typename T>
Matrix<T> Matrix<T>::operator+(const Matrix<T> &other) const {
    Matrix dest(getXDim(), getYDim());
    apply(*this, other, dest, Add<T>());
    return dest;
}

template <typename T>
Matrix<T> Matrix<T>::operator-(const Matrix &other) const {
    Matrix dest(getXDim(), getYDim());
    apply(*this, other, dest, Sub<T>());
    return dest;
}

template <typename T>
Matrix<T> Matrix<T>::operator*(const Matrix &other) const {
    Matrix dest(getXDim(), getYDim());
    apply(*this, other, dest, Mul<T>());
    return dest;
}

template <typename T>
Matrix<T> Matrix<T>::operator/(const Matrix &other) const {
    Matrix dest(getXDim(), getYDim());
    apply(*this, other, dest, Div<T>());
    return dest;
}

#endif // NUMC_H
