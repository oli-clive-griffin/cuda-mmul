#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

__global__ void matrixMulShared(float *A, float *B, float *C, int M, int N, int K) {
    __shared__ float sharedA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sharedB[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float sum = 0.0f;

    for (int t = 0; t < (K - 1) / TILE_WIDTH + 1; ++t) {
        if (row < M && t * TILE_WIDTH + tx < K)
            sharedA[ty][tx] = A[row * K + t * TILE_WIDTH + tx];
        else
            sharedA[ty][tx] = 0.0f;

        if (col < N && t * TILE_WIDTH + ty < K)
            sharedB[ty][tx] = B[(t * TILE_WIDTH + ty) * N + col];
        else
            sharedB[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
            sum += sharedA[ty][k] * sharedB[k][tx];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

void initMatrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; ++i)
        mat[i] = rand() / (float)RAND_MAX;
}

void printMatrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j)
            printf("%f ", mat[i * cols + j]);
        printf("\n");
    }
}

int main() {
    int M = 64, N = 64, K = 64;  // Matrix dimensions
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    float *h_A, *h_B, *h_C;
    h_A = (float*)malloc(sizeA);
    h_B = (float*)malloc(sizeB);
    h_C = (float*)malloc(sizeC);

    initMatrix(h_A, M, K);
    initMatrix(h_B, K, N);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    matrixMulShared<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);

    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    printf("First few elements of the result matrix:\n");
    for (int i = 0; i < 3 && i < M; ++i) {
        for (int j = 0; j < 3 && j < N; ++j)
            printf("%f ", h_C[i * N + j]);
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}