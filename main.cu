#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

__global__ void mmul(float* A, float* B, float* C,
                     int M, int K, int N) {
    __shared__ float sharedA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sharedB[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float sum = 0.0f;

    int num_threads = (K - 1) / TILE_WIDTH;
    for (int tile=0; tile < num_threads; ++tile) {
        // load sharedA from A
        int within_vert_a = row < M;
        int within_horiz_a = tile * TILE_WIDTH + tx < K;
        if (within_vert_a && within_horiz_a) {
            int col_idx = tile * TILE_WIDTH + tx;
            sharedA[ty][tx] = A[row * K + col_idx];
        } else {
            sharedA[ty][tx] = 0.0f;
        }

        // load sharedA from A
        int within_horiz_b = col < N;
        int within_vert_b = tile * TILE_WIDTH + ty < K;
        if (within_horiz_b && within_vert_b) {
            int row_idx = (tile * TILE_WIDTH + ty);
            sharedB[ty][tx] = B[row_idx * N + col];
        } else {
            sharedB[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int i=0; i < TILE_WIDTH; ++i) {
            sum += sharedA[ty][K] * sharedB[K][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = 3.f; // sum;
    }
}

void initialize_random(float* arr, int n) {
    for (int i = 0; i < n; i++)
        arr[i] = rand() / (float)RAND_MAX;
}

int main1() {
    int M = 64, K = 64, N = 64;

    int nA = M * K;
    int nB = K * N;
    int nC = M * N;

    size_t size_A = sizeof(float) * nA;
    size_t size_B = sizeof(float) * nB;
    size_t size_C = sizeof(float) * nC;

    float *h_A, *h_B, *h_C;
    h_A = (float*)malloc(size_A);
    h_B = (float*)malloc(size_B);
    h_C = (float*)malloc(size_C);

    initialize_random(h_A, nA);
    initialize_random(h_B, nB);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH); // each thread block is TILE_WIDTH x TILE_WIDTH threads
    dim3 dimGrid((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH); // we need ≈ N x M thread blocks

    mmul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, K, N);

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

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

int main() {
    main1();
}