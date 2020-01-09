#include "numc.cuh"

void* Managed::operator new(size_t len) {
    #ifdef DEBUG
    printf("Managed new\n");
    #endif
    void *ptr;
    gpuErrchk(cudaMallocManaged(&ptr, len));
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
void applyElementWise(const Matrix<T> &x, const Matrix<T> &y, Matrix<T> &dest, Op op) {
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    dim3 threadsPerBlock(THREADS_PER_BLOCK_2D, THREADS_PER_BLOCK_2D);
    dim3 numBlocks;
    size_t threshold = numSMs * 32;
    numBlocks.x = std::min(threshold, (dest.getRows() + THREADS_PER_BLOCK_2D - 1) / THREADS_PER_BLOCK_2D);
    numBlocks.y = std::min(threshold, (dest.getCols() + THREADS_PER_BLOCK_2D - 1) / THREADS_PER_BLOCK_2D);
    elementWiseKernel<<<numBlocks, threadsPerBlock>>>(*x.matrixGPU, *y.matrixGPU, *dest.matrixGPU, op);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
}

template <typename T>
__host__
void applyMatMul(const Matrix<T> &x, const Matrix<T> &y, Matrix<T> &dest) {
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    dim3 threadsPerBlock(THREADS_PER_BLOCK_2D, THREADS_PER_BLOCK_2D);
    dim3 numBlocks;
    size_t threshold = numSMs * 32;
    numBlocks.x = std::min(threshold, (dest.getRows() + THREADS_PER_BLOCK_2D - 1) / THREADS_PER_BLOCK_2D);
    numBlocks.y = std::min(threshold, (dest.getCols() + THREADS_PER_BLOCK_2D - 1) / THREADS_PER_BLOCK_2D);
    matMulKernel<<<numBlocks, threadsPerBlock>>>(*x.matrixGPU, *y.matrixGPU, *dest.matrixGPU);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
}

template <typename T>
__host__
void applyRepeat(const Matrix<T> &x, Matrix<T> &dest) {
    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    dim3 threadsPerBlock(THREADS_PER_BLOCK_2D, THREADS_PER_BLOCK_2D);
    dim3 numBlocks;
    size_t threshold = numSMs * 32;
    numBlocks.x = std::min(threshold, (dest.getRows() + THREADS_PER_BLOCK_2D - 1) / THREADS_PER_BLOCK_2D);
    numBlocks.y = std::min(threshold, (dest.getCols() + THREADS_PER_BLOCK_2D - 1) / THREADS_PER_BLOCK_2D);
    repeatKernel<<<numBlocks, threadsPerBlock>>>(*x.matrixGPU, *dest.matrixGPU);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());
}

template<typename Op, typename T>
__global__
void elementWiseKernel(const MatrixGPU<T> &x, const MatrixGPU<T> &y, MatrixGPU<T> &dest, Op op) {
    const size_t xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t xStride = gridDim.x * blockDim.x;
    const size_t yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t yStride = gridDim.y * blockDim.y;
    for (size_t i = xIndex; i < dest.getRows(); i+= xStride){
        for (size_t j = yIndex; j < dest.getCols(); j+= yStride){
            dest(i, j) = op(x(i, j), y(i, j));
        }
    }
}

template<typename T>
__global__ 
void matMulKernel(const MatrixGPU<T> &x, const MatrixGPU<T> &y, MatrixGPU<T> &dest){
    const dim3 blocksLen((dest.getRows() + blockDim.x - 1) / blockDim.x, (dest.getCols() + blockDim.y - 1) / blockDim.y);
    const dim3 blocksStride(gridDim.x, gridDim.y);
    for (size_t blockRow = blockIdx.x; blockRow < blocksLen.x; blockRow += blocksStride.x) {
        for (size_t blockCol = blockIdx.y; blockCol < blocksLen.y; blockCol += blocksStride.y) {
            const size_t destBlockRow = blockRow * blockDim.x;
            const size_t destBlockCol = blockCol * blockDim.y;
            const size_t destRow = destBlockRow + threadIdx.x;
            const size_t destCol = destBlockCol + threadIdx.y;
            T destValue = 0;
            for (size_t blockXYIdx = 0; blockXYIdx < ((x.getCols() + THREADS_PER_BLOCK_2D - 1) / THREADS_PER_BLOCK_2D); ++blockXYIdx) {
                const size_t xBlockCol = blockXYIdx * THREADS_PER_BLOCK_2D;
                const size_t numSums = min((size_t)THREADS_PER_BLOCK_2D, x.getCols() - xBlockCol);
                const size_t xCol = blockXYIdx * THREADS_PER_BLOCK_2D + threadIdx.y;
                const size_t yRow = blockXYIdx * THREADS_PER_BLOCK_2D + threadIdx.x;
                __shared__ T xs[THREADS_PER_BLOCK_2D][THREADS_PER_BLOCK_2D];
                __shared__ T ys[THREADS_PER_BLOCK_2D][THREADS_PER_BLOCK_2D];
                if (destRow < x.getRows() && xCol < x.getCols()){
                    xs[threadIdx.x][threadIdx.y] = x(destRow, xCol);
                    #ifdef DEBUG
                        printf("xs[%d][%d] = %f, destRow=%d, xCol=%d \n", threadIdx.x, threadIdx.y, xs[threadIdx.x][threadIdx.y], (int)destRow, (int)xCol);
                    #endif
                }
                if (yRow < y.getRows() && destCol < y.getCols()) {
                    ys[threadIdx.x][threadIdx.y] = y(yRow, destCol);
                    #ifdef DEBUG
                        printf("ys[%d][%d] = %f, yRow=%d, destCol=%d \n", threadIdx.x, threadIdx.y, ys[threadIdx.x][threadIdx.y], (int)yRow, (int)destCol);
                    #endif
                }
                __syncthreads();
                if (destRow < dest.getRows() && destCol < dest.getCols()){
                    for (size_t i = 0; i < numSums; ++i) {
                        destValue += xs[threadIdx.x][i] * ys[i][threadIdx.y];
                        #ifdef DEBUG
                            printf("i = %d, %f * %f = %f\n", (int)i, xs[threadIdx.x][i], ys[i][threadIdx.y], destValue);
                        #endif
                    }
                }
                __syncthreads();
            }
            if (destRow < dest.getRows() && destCol < dest.getCols()){
                dest(destRow, destCol) = destValue;
                #ifdef DEBUG
                    printf("dest value : %f\n", destValue);
                #endif
            }
        }
    }
}

template<typename T>
__global__
void repeatKernel(const MatrixGPU<T> &x, MatrixGPU<T> &dest) {
    const size_t xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t xStride = gridDim.x * blockDim.x;
    const size_t yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t yStride = gridDim.y * blockDim.y;
    for (size_t i = xIndex; i < dest.getRows(); i+= xStride){
        for (size_t j = yIndex; j < dest.getCols(); j+= yStride){
            dest(i, j) = x(i % x.getRows(), j % x.getCols());
        }
    }
}

template class Matrix<float>;
