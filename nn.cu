// TODO:
// - backprop
// - python bindings
// - ...

#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

#define CEIL_DIV(numer, denom) ((numer + denom - 1) / (denom))

__global__ void fusedBareLinear(
    const float *in, // (batch, in)
    const float *W,  // (in, out)
    const float *b,  // (out,)
    float *out,      // (batch, out)
    int nBatches,
    int inFeatures,
    int outFeatures
) {
    __shared__ float sharedW[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sharedIn[TILE_WIDTH][TILE_WIDTH];


    int blY = blockIdx.y; int thY = threadIdx.y;
    int blX = blockIdx.x; int thX = threadIdx.x;

    int batchIdx = blY * TILE_WIDTH + thY;
    int outFeatureIdx = blX * TILE_WIDTH + thX;

    float sum = 0.0f;
    for (int tileIdx = 0; tileIdx < CEIL_DIV(inFeatures, TILE_WIDTH); ++tileIdx) {
        int inFeatureIdxIn = tileIdx * TILE_WIDTH + thX;
        bool shouldLoadIn = batchIdx < nBatches && inFeatureIdxIn < inFeatures;
        // sharedIn[thY][thX] = shouldLoadIn * in[batchIdx * inFeatures + inFeatureIdxIn];
        sharedIn[thY][thX] = shouldLoadIn ? in[batchIdx * inFeatures + inFeatureIdxIn] : 0.0f;

        int inFeatureIdxW = tileIdx * TILE_WIDTH + thY;
        bool shouldLoadW = outFeatureIdx < outFeatures && inFeatureIdxW < inFeatures;
        // sharedW[thY][thX] = shouldLoadW * W[inFeatureIdxW * outFeatures + outFeatureIdx];
        sharedW[thY][thX] = shouldLoadW ? W[inFeatureIdxW * outFeatures + outFeatureIdx] : 0.0f;

        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; ++k)
            sum += sharedIn[thY][k] * sharedW[k][thX];

        __syncthreads();
    }

    sum += b[outFeatureIdx]; // bias

    int outIdx = batchIdx * outFeatures + outFeatureIdx;
    if (batchIdx < nBatches && outFeatureIdx < outFeatures)
        out[outIdx] = sum;
}



__global__ void fusedLinearRelu(
    const float *in, // (batch, in)
    const float *W,  // (in, out)
    const float *b,  // (out,)
    float *out,      // (batch, out)
    int nBatches,
    int inFeatures,
    int outFeatures
) {
    __shared__ float sharedW[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sharedIn[TILE_WIDTH][TILE_WIDTH];


    int blY = blockIdx.y; int thY = threadIdx.y;
    int blX = blockIdx.x; int thX = threadIdx.x;

    int batchIdx = blY * TILE_WIDTH + thY;
    int outFeatureIdx = blX * TILE_WIDTH + thX;

    float sum = 0.0f;
    for (int tileIdx = 0; tileIdx < CEIL_DIV(inFeatures, TILE_WIDTH); ++tileIdx) {
        int inFeatureIdxIn = tileIdx * TILE_WIDTH + thX;
        bool shouldLoadIn = batchIdx < nBatches && inFeatureIdxIn < inFeatures;
        // sharedIn[thY][thX] = shouldLoadIn * in[batchIdx * inFeatures + inFeatureIdxIn];
        sharedIn[thY][thX] = shouldLoadIn ? in[batchIdx * inFeatures + inFeatureIdxIn] : 0.0f;

        int inFeatureIdxW = tileIdx * TILE_WIDTH + thY;
        bool shouldLoadW = outFeatureIdx < outFeatures && inFeatureIdxW < inFeatures;
        // sharedW[thY][thX] = shouldLoadW * W[inFeatureIdxW * outFeatures + outFeatureIdx];
        sharedW[thY][thX] = shouldLoadW ? W[inFeatureIdxW * outFeatures + outFeatureIdx] : 0.0f;

        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; ++k)
            sum += sharedIn[thY][k] * sharedW[k][thX];

        __syncthreads();
    }

    sum += b[outFeatureIdx]; // bias

    sum = fmaxf(sum, 0.0f); // ReLU

    int outIdx = batchIdx * outFeatures + outFeatureIdx;
    if (batchIdx < nBatches && outFeatureIdx < outFeatures)
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

void launchLinearRelu(
    int nBatches,
    float* d_in,
    float* d_out,
    Layer layer
) {
    dim3 gridDim_(CEIL_DIV(nBatches, TILE_WIDTH), CEIL_DIV(layer.outFeatures, TILE_WIDTH));
    dim3 blockDim_(TILE_WIDTH, TILE_WIDTH);

    fusedLinearRelu<<<gridDim_, blockDim_>>>(
        d_in, layer.d_W, layer.d_b, d_out,
        nBatches, layer.inFeatures, layer.outFeatures
    );
}

void launchBareLinear(
    int nBatches,
    float* d_in,
    float* d_out,
    Layer layer
) {
    dim3 gridDim_(CEIL_DIV(nBatches, TILE_WIDTH), CEIL_DIV(layer.outFeatures, TILE_WIDTH));
    dim3 blockDim_(TILE_WIDTH, TILE_WIDTH);

    fusedBareLinear<<<gridDim_, blockDim_>>>(
        d_in, layer.d_W, layer.d_b, d_out,
        nBatches, layer.inFeatures, layer.outFeatures
    );
}


typedef struct Layer {
    int inFeatures;
    int outFeatures;
    float* d_W;
    float* d_b;
} Layer;

typedef struct NN {
    int sizeIn;
    int numLayers;
    Layer* layers;
} NN;


Layer initLayer(int sizeIn, int sizeOut) {
    size_t sizeW = sizeof(float) * sizeIn * sizeOut;
    size_t sizeB = sizeof(float) * sizeOut;

    float* h_W = (float*)malloc(sizeW);
    initMatrixRandom(h_W, sizeIn * sizeOut);

    float* d_W;
    cudaError_t err1 = cudaMalloc(d_W, sizeW);
    if (err1 != cudaSuccess)
        printf("error `cudaMalloc`ing d_W. error: %s\n", cudaGetErrorString(err1));
    cudaError_t err2 = cudaMemcpy(*d_W, h_W, sizeW, cudaMemcpyHostToDevice);
    if (err2 != cudaSuccess)
        printf("error `cudaMemcpy`ing to d_W. error: %s\n", cudaGetErrorString(err2));

    free(h_W);

    float* h_b = (float*)malloc(sizeB);
    initMatrixZeros(h_b, sizeOut);

    float* d_b;
    cudaError_t err3 = cudaMalloc(d_b, sizeB);
    if (err3 != cudaSuccess)
        printf("error `cudaMalloc`ing d_b. error: %s\n", cudaGetErrorString(err3));
    cudaError_t err4 = cudaMemcpy(*d_b, h_b, sizeB, cudaMemcpyHostToDevice);
    if (err4 != cudaSuccess)
        printf("error `cudaMemcpy`ing to d_b. error: %s\n", cudaGetErrorString(err4));

    free(h_b);

    Layer layer;
    layer.inFeatures = sizeIn;
    layer.outFeatures = sizeOut;
    layer.d_W = *d_W;
    layer.d_b = *d_b;

    return layer;
}


void printDeviceArr(float* d_x, int n_elem, char* name) {
    size_t size = sizeof(float) * n_elem;
    float* h_x = (float*)malloc(size);
    cudaError_t err = cudaMemcpy(h_x, d_x, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
        printf("error `cudaMemcpy`ing to d_b. error: %s\n", cudaGetErrorString(err));
    printf("%s:\n", name);
    for (int i = 0; i < n_elem; ++i)
        printf("%f ", h_x[i]);
    printf("\n\n");  
    free(h_x);
}


NN* createNN(
    int sizeIn,
    int numLayers,
    int *layerSizes
) {
    NN* nn = (NN*)malloc(sizeof(NN));

    nn->sizeIn = sizeIn;
    nn->numLayers = numLayers;
    nn->layers = (Layer*)malloc(sizeof(Layer) * numLayers);

    nn->layers[0] = initLayer(sizeIn, layerSizes[0]);
    for (int i = 1; i < numLayers; ++i) {
        nn->layers[i] = initLayer(layerSizes[i-1], layerSizes[i]);
    }

    return nn;
}


void freeNN(NN* nn) {
    for (int i = 0; i < nn->numLayers; ++i) {
        cudaFree(nn->layers[i].d_W);
        cudaFree(nn->layers[i].d_b);
    }

    free(nn->layers);
    free(nn);
}


void runNN(
    NN* nn,
    int nBatches,
    float* d_in,
    float* d_out
) {
    for (int i = 0; i < nn->numLayers - 1; ++i) {
        Layer layer = nn->layers[i];
        float *d_out;

        cudaError_t err1 = cudaMalloc(&d_out, sizeof(float) * nn->layer.outFeatures);
        if (err1 != cudaSuccess)
            printf("error cudaMallocing d_out. error: %s\n", cudaGetErrorString(err1));

        launch(
            nBatches,
            layer.inFeatures,
            layer.outFeatures,
            d_in,
            layer.d_W,
            layer.d_b,
            d_out
        );

        d_in = d_out;

        // tricky, need to free the output of the previous layer
    }



    cudaFree(d_h2);
}


int main() {
    int inFeatures = 3;
    int hiddenDim1 = 5;
    int hiddenDim2 = 3;
    int outFeatures = 10;

    NN* nn = createNN(inFeatures, 3, [64, 16, 10]);
    
    size_t sizeIn = sizeof(float) * inFeatures;

    float* h_in = (float*)malloc(sizeIn);
    initMatrixRandom(h_in, inFeatures);

    float* d_in;
    cudaError_t err1 = cudaMalloc(&d_in, sizeIn);
    if (err1 != cudaSuccess)
        printf("error `cudaMalloc`ing d_in. error: %s\n", cudaGetErrorString(err1));

    cudaError_t err2 = cudaMemcpy(d_in, h_in, sizeIn, cudaMemcpyHostToDevice);
    if (err2 != cudaSuccess)
        printf("error `cudaMemcpy`ing h_in to d_in. error: %s\n", cudaGetErrorString(err2));

    size_t sizeOut = sizeof(float) * outFeatures;

    float* d_out;
    cudaError_t err3 = cudaMalloc(&d_out, sizeOut);
    if (err3 != cudaSuccess)
        printf("error `cudaMalloc`ing d_out. error: %s\n", cudaGetErrorString(err3));

    runNN(nn, d_in, d_out);
    freeNN(nn);

    float* h_out = (float*)malloc(sizeOut);
    cudaError_t err4 = cudaMemcpy(h_out, d_out, sizeOut, cudaMemcpyDeviceToHost);
    if (err4 != cudaSuccess)
        printf("error `cudaMemcpy`ing d_out to h_out. error: %s\n", cudaGetErrorString(err4));

    // ==========

    printf("First few elements of `out`:\n");
    for (int i = 0; i < outFeatures; ++i)
        printf("%f ", h_out[i]);

}
