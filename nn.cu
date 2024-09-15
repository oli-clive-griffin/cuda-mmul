// TODO:
// - backprop
// - python bindings
// - ...

#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

#define CEIL_DIV(numer, denom) (numer + denom - 1) / (denom)

cudaError_t err;


__global__ void fusedLinear(
    float *in, // (batch, in,)
    float *W, // (in, out)
    float *b, // (out,)
    float *out, // (out,)
    int nBatches,
    int inDim,
    int outDim,
) {
    __shared__ float sharedW[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sharedIn[TILE_WIDTH][TILE_WIDTH];


    int by = blockIdx.y; int thY = threadIdx.y;
    int bx = blockIdx.x; int thX = threadIdx.x;

    int batchIdx = by * TILE_WIDTH + thY;
    int outFeatureIdx = bx * TILE_WIDTH + thX;

    float sum = 0.0f;
    for (int tileIdx = 0; tileIdx < CEIL_DIV(K, TILE_WIDTH) + 1; ++tileIdx) {
        int inFeatureIdxIn = tileIdx * TILE_WIDTH + thX;
        bool shouldLoadIn = batchIdx < nBatches && inFeatureIdxIn < inDim;
        sharedIn[thY][thX] = shouldLoadA ? a[batchIdx * inDim + inFeatureIdxIn] : 0.0f;

        int inFeatureIdxW = tileIdx * TILE_WIDTH + thY;
        bool shouldLoadW = outFeatureIdx < outDim && inFeatureIdxW < inDim;
        sharedW[thY][thX] = shouldLoadB ? b[inFeatureIdxW * outDim + outFeatureIdx] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
            sum += sharedIn[thY][k] * sharedW[k][thX]

        __syncthreads();
    }

    sum += b[outFeatureIdx]; // bias

    sum *= (sum > 0); // ReLU

    int outIdx = batchIdx * outDim + outFeatureIdx;
    if (batchIdx < batches && col < outDim)
        out[outIdx] = sum;
}


void initMatrixRandom(float *mat, int n_elem) {
    for (int i = 0; i < n_elem; ++i)
        mat[i] = rand() / (float)RAND_MAX;
}


void initMatrixZeros(float *mat, int n_elem) {
    for (int i = 0; i < n_elem; ++i)
        mat[i] = 0;
}

void linear(
    int nBatches,
    int inDim,
    int outDim,
    float* d_in,
    float* d_W,
    float* d_b,
    float* d_out
) {
    dim3 gridDim_(CEIL_DIV(N, TILE_WIDTH), CEIL_DIV(M, TILE_WIDTH))
    dim3 blockDim_(TILE_WIDTH, TILE_WIDTH);

    fusedLinear<<<gridDim_, blockDim_>>>(
        d_in, d_W, d_b, d_out,
        nBatches, inDim, outDim
    );
}

typedef struct NN {
    int sizeIn;
    float* d_W1;
    float* d_b1;
    int size1;
    float* d_W2;
    float* d_b2;
    int size2;
    float* d_W3;
    float* d_b3;
    int sizeOut;
} NN;


void initLayer(int sizeIn, int sizeOut, float** d_W, float** d_b) {
    size_t sizeW = sizeof(float) * sizeIn * sizeOut;
    size_t sizeB = sizeof(float) * sizeOut;

    float* h_W = (float*)malloc(sizeW);
    initMatrixRandom(h_W, sizeIn * sizeOut);

    err = cudaMalloc(d_W, sizeW);
    if (err != cudaSuccess) {
        printf("error `cudaMalloc`ing d_W. error: %s\n", cudaGetErrorString(err));
    }

    err = cudaMemcpy(*d_W, h_W, sizeW, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("error `cudaMemcpy`ing to d_W. error: %s\n", cudaGetErrorString(err));
    }

    float* h_b = (float*)malloc(sizeB);
    initMatrixZeros(h_b, sizeOut);

    err = cudaMalloc(d_b, sizeB);
    if (err != cudaSuccess) {
        printf("error `cudaMalloc`ing d_b. error: %s\n", cudaGetErrorString(err));
    }

    err = cudaMemcpy(*d_b, h_b, sizeB, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("error `cudaMemcpy`ing to d_b. error: %s\n", cudaGetErrorString(err));
    }
}


void printDeviceArr(float* d_x, int n_elem, char* name) {
    size_t size = sizeof(float) * n_elem;
    float* h_x = (float*)malloc(size);
    err = cudaMemcpy(h_x, d_x, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("error `cudaMemcpy`ing to d_b. error: %s\n", cudaGetErrorString(err));
    }
    printf("%s:\n", name);
    for (int i = 0; i < n_elem; ++i)
        printf("%f ", h_x[i]);
    printf("\n\n");  
    free(h_x);
}


NN* createNN(
    int sizeIn,
    int sizeLayer1,
    int sizeLayer2,
    int sizeOut
) {
    NN* nn = (NN*)malloc(sizeof(NN));

    nn->sizeIn = sizeIn;
    nn->size1 = sizeLayer1;
    nn->size2 = sizeLayer2;

    initLayer(sizeIn, sizeLayer1, &nn->d_W1, &nn->d_b1);
    initLayer(sizeLayer1, sizeLayer2, &nn->d_W2, &nn->d_b2);
    initLayer(sizeLayer2, sizeOut, &nn->d_W3, &nn->d_b3);
    printDeviceArr(nn->d_W1, nn->sizeIn * nn->size1, (char*)"d_W1");
    printDeviceArr(nn->d_W2, nn->size1 * nn->size2, (char*)"d_W2");

    nn->sizeOut = sizeOut;
    return nn;
}


void freeNN(NN* nn) {
    free(nn->d_W1);
    free(nn->d_b1);
    free(nn->d_W2);
    free(nn->d_b2);
    free(nn->d_W3);
    free(nn->d_b3);
    free(nn);
}


void runNN(
    NN* nn,
    float* d_in,
    float* d_out
) {
    float *d_h1, *d_h2;

    err = cudaMalloc(&d_h1, sizeof(float) * nn->size1);
    if (err != cudaSuccess) {
        printf("error `cudaMemcpy`ing to d_b. error: %s\n", cudaGetErrorString(err));
    }

    err = cudaMalloc(&d_h2, sizeof(float) * nn->size2);
    if (err != cudaSuccess) {
        printf("error `cudaMemcpy`ing to d_b. error: %s\n", cudaGetErrorString(err));
    }

    linear(nn->sizeIn, nn->size1, d_in, nn->d_W1, nn->d_b1, d_h1);
    linear(nn->size1, nn->size2, d_h1, nn->d_W2, nn->d_b2, d_h2);
    linear(nn->size2, nn->sizeOut, d_h2, nn->d_W3, nn->d_b3, d_out);

    cudaFree(d_h1);
    cudaFree(d_h2);
}


int main() {
    int inDim = 3;
    int hiddenDim1 = 5;
    int hiddenDim2 = 3;
    int outDim = 10;

    NN* nn = createNN(inDim, hiddenDim1, hiddenDim2, outDim);
    
    size_t sizeIn = sizeof(float) * inDim;

    float* h_in = (float*)malloc(sizeIn);
    initMatrixRandom(h_in, inDim);

    float* d_in;
    err = cudaMalloc(&d_in, sizeIn);
    if (err != cudaSuccess) {
        printf("error `cudaMalloc`ing d_in. error: %s\n", cudaGetErrorString(err));
    }

    err = cudaMemcpy(d_in, h_in, sizeIn, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("error `cudaMemcpy`ing h_in to d_in. error: %s\n", cudaGetErrorString(err));
    }

    size_t sizeOut = sizeof(float) * outDim;

    float* d_out;
    err = cudaMalloc(&d_out, sizeOut);
    if (err != cudaSuccess) {
        printf("error `cudaMalloc`ing d_out. error: %s\n", cudaGetErrorString(err));
    }

    runNN(nn, d_in, d_out);

    float* h_out = (float*)malloc(sizeOut);
    err = cudaMemcpy(h_out, d_out, sizeOut, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("error `cudaMemcpy`ing d_out to h_out. error: %s\n", cudaGetErrorString(err));
    }

    // ==========

    printf("First few elements of `out`:\n");
    for (int i = 0; i < outDim; ++i)
        printf("%f ", h_out[i]);

}
