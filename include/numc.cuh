#ifndef NUMC_H
#define NUMC_H

#include <algorithm>
#include <cmath>
#include <ctime>
#include <stdio.h>
#include <iomanip>
#include <cuda.h>
#include <functional>
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

// #define DEBUG

#define THREADS_PER_BLOCK_1D 256
#define THREADS_PER_BLOCK_2D 16

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

template <typename T>
class MatrixGPU;

template <typename T>
class Matrix;

template <typename Op, typename T>
__host__
void applyElementWise(const Matrix<T> &x, const Matrix<T> &y, Matrix<T> &dest, Op op);

template <typename T>
__host__
void applyMatMul(const Matrix<T> &x, const Matrix<T> &y, Matrix<T> &dest);

template<typename Op, typename T>
__global__
void elementWiseKernel(const MatrixGPU<T> *x, const MatrixGPU<T> *y, MatrixGPU<T> *dest, Op op);

template<typename T>
__global__
void matMulKernel(const MatrixGPU<T> *x, const MatrixGPU<T> *y, MatrixGPU<T> *dest);

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
        size_t *nrows;
        size_t *ncols;
    
    public:
        // constructors
        __host__ MatrixGPU();
        __host__ MatrixGPU(const size_t nrows, const size_t ncols);
        // destructor
        __host__ ~MatrixGPU();
        // copy
        __host__ MatrixGPU(const MatrixGPU<T> &other);
        // assignment
        __host__ MatrixGPU<T>& operator=(const MatrixGPU<T> &other);
        // functions
        __host__ __device__ T* begin() const{return elements;}
        __host__ __device__ T* end() const{return elements + (*nrows) * (*ncols);}
        __host__ __device__ T* begin() {return elements;}
        __host__ __device__ T* end() {return elements + (*nrows) * (*ncols);}
        __host__ __device__ size_t& getRows() const;
        __host__ __device__ size_t& getCols() const;
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
        Matrix(const size_t nrows, const size_t ncols);
        // destructor
        ~Matrix();
        // copy
        Matrix(const Matrix<T> &other);
        // assignment
        Matrix<T>& operator=(const Matrix<T> &other);
        // functions
        size_t& getRows() const;
        size_t& getCols() const;
        const T& operator()(size_t i, size_t j) const;
        T& operator()(size_t i, size_t j);
        Matrix<T> operator+(const Matrix& other) const;
        Matrix<T> operator-(const Matrix& other) const;
        Matrix<T> operator*(const Matrix& other) const;
        Matrix<T> operator/(const Matrix& other) const;
        Matrix<T> mm(const Matrix& other) const;
        Matrix<T>& operator+=(const Matrix& other);
        Matrix<T>& operator-=(const Matrix& other);
        Matrix<T>& operator*=(const Matrix& other);
        Matrix<T>& operator/=(const Matrix& other);
        T getMin() const;
        T getMax() const;
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
    gpuErrchk(cudaMallocManaged(&nrows, sizeof(size_t)));
    gpuErrchk(cudaMallocManaged(&ncols, sizeof(size_t)));
    *nrows = 0;
    *ncols = 0;
}



template <typename T>
__host__
MatrixGPU<T>::MatrixGPU(const size_t _nrows, const size_t _ncols) {
    #ifdef DEBUG
        printf("MatrixGPU(nrows=%d, ncols=%d) constructor\n", _nrows, _ncols);
    #endif
    gpuErrchk(cudaMallocManaged(&elements, sizeof(T) * _nrows * _ncols));
    gpuErrchk(cudaMallocManaged(&nrows, sizeof(size_t)));
    gpuErrchk(cudaMallocManaged(&ncols, sizeof(size_t)));
    *nrows = _nrows;
    *ncols = _ncols;
}

template <typename T>
__host__
MatrixGPU<T>::~MatrixGPU(){
    #ifdef DEBUG
        printf("~MatrixGPU\n");
    #endif
    cudaFree(elements);
    cudaFree(nrows);
    cudaFree(ncols);
}

template <typename T>
__host__  
MatrixGPU<T>::MatrixGPU(const MatrixGPU &other){
    #ifdef DEBUG
        printf("MatrixGPU copy constructor\n");
    #endif
    gpuErrchk(cudaMallocManaged(&elements, sizeof(T) * other.getRows() * other.getCols()));
    gpuErrchk(cudaMallocManaged(&nrows, sizeof(size_t)));
    gpuErrchk(cudaMallocManaged(&ncols, sizeof(size_t)));

    *nrows = other.getRows();
    *ncols = other.getCols();
    for(size_t i = 0; i < getRows() * getCols(); ++i){
       elements[i] = other.elements[i];
    }
}

template<class T>
__host__
MatrixGPU<T>& MatrixGPU<T>::operator=(const MatrixGPU<T> &other) {
    if (this != &other){
        *nrows = other.getRows();
        *ncols = other.getCols();
        for(size_t i = 0; i < getRows() * getCols(); ++i){
           elements[i] = other.elements[i];
        }
    }
    return *this;
}

template <typename T>
__host__ __device__ 
size_t& MatrixGPU<T>::getRows() const{
    return *nrows;
}

template <typename T>
__host__ __device__ 
size_t& MatrixGPU<T>::getCols() const{
    return *ncols;
}


template <typename T>
__host__ __device__
const T& MatrixGPU<T>::operator()(size_t i, size_t j) const {
    return elements[(getCols() * i) + j];
}

template <typename T>
__host__ __device__
T& MatrixGPU<T>::operator()(size_t i, size_t j) {
    return elements[(getCols() * i) + j];
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
Matrix<T>::Matrix(const size_t _nrows, const size_t _ncols) {
    #ifdef DEBUG
        printf("Matrix(nrows, ncols) constructor\n");
    #endif
    matrixGPU = new MatrixGPU<T>(_nrows, _ncols);
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
        matrixGPU = new MatrixGPU<T>(other.getRows(), other.getCols());
        *matrixGPU = *(other.matrixGPU);
    }
    return *this;
}

template <typename T>
size_t& Matrix<T>::getRows() const{
    return matrixGPU->getRows();
}

template <typename T>
size_t& Matrix<T>::getCols() const{
    return matrixGPU->getCols();
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
    Matrix dest(getRows(), getCols());
    applyElementWise(*this, other, dest, Add<T>());
    return dest;
}

template <typename T>
Matrix<T> Matrix<T>::operator-(const Matrix &other) const {
    Matrix dest(getRows(), getCols());
    applyElementWise(*this, other, dest, Sub<T>());
    return dest;
}

template <typename T>
Matrix<T> Matrix<T>::operator*(const Matrix &other) const {
    Matrix dest(getRows(), getCols());
    applyElementWise(*this, other, dest, Mul<T>());
    return dest;
}

template <typename T>
Matrix<T> Matrix<T>::operator/(const Matrix &other) const {
    Matrix dest(getRows(), getCols());
    applyElementWise(*this, other, dest, Div<T>());
    return dest;
}

template <typename T>
Matrix<T> Matrix<T>::mm(const Matrix &other) const {
    Matrix dest(getRows(), other.getCols());
    applyMatMul(*this, other, dest);
    return dest;
}

template <typename T>
Matrix<T>& Matrix<T>::operator+=(const Matrix<T> &other) {
    applyElementWise(*this, other, *this, Add<T>());
    return *this;
}

template <typename T>
Matrix<T>& Matrix<T>::operator-=(const Matrix<T> &other) {
    applyElementWise(*this, other, *this, Sub<T>());
    return *this;
}

template <typename T>
Matrix<T>& Matrix<T>::operator*=(const Matrix<T> &other) {
    applyElementWise(*this, other, *this, Mul<T>());
    return *this;
}

template <typename T>
Matrix<T>& Matrix<T>::operator/=(const Matrix<T> &other) {
    applyElementWise(*this, other, *this, Div<T>());
    return *this;
}

template <typename T>
T Matrix<T>::getMin() const {
    return *std::min_element(matrixGPU->begin(), matrixGPU->end());
}

template <typename T>
T Matrix<T>::getMax() const {
    return *std::max_element(matrixGPU->begin(), matrixGPU->end());
}

template <typename T>
int numDigits(T number)
{
    int digits = 0;
    if (number < 0){
        digits = 1;
        number *= -1;
    }
    while (number > 10) {
        number /= 10;
        digits++;
    }
    return ++digits;
}

template <typename T>
std::ostream& _printRow(std::ostream &out, const Matrix<T> &matrix, const int row, const int cols, const int maxlen) {
    out << "[";
    if (cols < 10) { //rows < 10 cols < 10
        for (int col = 0; col < cols; ++col) {
            out << std::setw(maxlen + 3) << std::right << matrix(row, col);
            if (col + 1 < cols) {
                out << ", ";
            }
        }
        out << "]" << std::endl;
    } else {  // cols > 10
        for (int col = 0; col < 3; ++col) {
            out << std::setw(maxlen + 3) << std::right << matrix(row, col);
            out << ", ";
        }
        out << std::setw(5) << std::left << "...,";
        for (int col = cols - 3; col < cols; ++col) {
            out << std::setw(maxlen + 3) << std::right << matrix(row, col);
            if (col + 1 < cols) {
                out << ", ";
            }
        }
        out << "]" << std::endl;
    }
    return out;
}

template <typename T>
std::ostream& operator<< (std::ostream &out, const Matrix<T> &matrix) {
    int maxlen = std::max(numDigits(matrix.getMax()), numDigits(matrix.getMin()));
    size_t rows = matrix.getRows();
    size_t cols = matrix.getCols();
    out << "Matrix [" << rows << ", " << cols << "]" << std::endl;
    out << std::fixed << std::setprecision(2) << std::setfill(' ');
    if (rows < 10){
        for (int row = 0; row < rows; ++row) {
            _printRow(out, matrix, row, cols, maxlen);
        }
    } else { // rows > 10
        for (int row = 0; row < 3; ++row) {
            _printRow(out, matrix, row, cols, maxlen);
        }
        out  << "...," << std::endl;
        for (int row = rows - 3; row < rows; ++row) {
            _printRow(out, matrix, row, cols, maxlen);
        }
    } 
    return out;
}
#endif // NUMC_H
