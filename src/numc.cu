#include "numc.cuh"

void* Managed::operator new(size_t len) {
#ifdef DEBUG
    printf("Managed new\n");
#endif
    void *ptr;
    cudaMallocManaged(&ptr, len);
    cudaDeviceSynchronize();
    return ptr;
}

void Managed::operator delete(void *ptr) {
#ifdef DEBUG
    printf("Managed delete\n");
#endif
    cudaDeviceSynchronize();
    cudaFree(ptr);
}

template <typename Op, typename T>
__host__
void apply(const Matrix<T> &x, const Matrix<T> &y, Matrix<T> &dest, Op op) {
    dim3 threadsPerBlock(THREADS_PER_BLOCK_2D, THREADS_PER_BLOCK_2D);
    dim3 numBlocks(
        (dest.getXDim() + THREADS_PER_BLOCK_2D - 1) / THREADS_PER_BLOCK_2D, 
        (dest.getYDim() + THREADS_PER_BLOCK_2D - 1) / THREADS_PER_BLOCK_2D
    );
    operation_kernel<<<numBlocks, threadsPerBlock>>>(*x.matrixGPU, *y.matrixGPU,
            *dest.matrixGPU, op);
    cudaError_t errSync  = cudaGetLastError();
    cudaError_t errAsync = cudaDeviceSynchronize();
    if (errSync != cudaSuccess){
        printf("Sync kernel error: %d %s\n", errSync, cudaGetErrorString(errSync));
    }
    if (errAsync != cudaSuccess){
        printf("Async kernel error: %d %s\n", errAsync, cudaGetErrorString(errAsync));
    }
}

template<typename Op, typename T>
__global__
void operation_kernel(const MatrixGPU<T> &x, const MatrixGPU<T> &y, MatrixGPU<T> &dest, Op op) {
    size_t xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    size_t xStride = gridDim.x * blockDim.x;
    size_t yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    size_t yStride = gridDim.y * blockDim.y;
    for (size_t i = xIndex; i < dest.getXDim(); i+= xStride){
        for (size_t j = yIndex; j < dest.getYDim(); j+= yStride){
            dest(i, j) = op(x(i, j), y(i, j));
        }
    }
}

template class Matrix<float>;
INSTANTIATE_operation_kernel(float, Add);
