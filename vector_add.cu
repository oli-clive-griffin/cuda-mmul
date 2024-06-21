// __global__ void vectorAdd(float* A, float* B, float* C, int n) {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;

//     if (tid < n)
//         C[tid] = A[tid] + B[tid];
// }

// const int HEAD = 3;
// void print_head(float* arr) {
//     for (int i = 0; i < HEAD; i++) {
//         printf("%lf", arr[i]);
//         if (i != HEAD-1)
//             printf(", ");
//     }
//     printf("\n");
// }

// int demo_vec_add() {
//     int n = 10000;
//     size_t size = n * sizeof(float);

//     float* h_A = (float*)malloc(size);
//     float* h_B = (float*)malloc(size);
//     float* h_C = (float*)malloc(size);

//     initialize_random(h_A, n);
//     initialize_random(h_B, n);
//     print_head(h_A);
//     print_head(h_B);

//     float *d_A, *d_B, *d_C;
//     cudaMalloc((void**)&d_A, size);
//     cudaMalloc((void**)&d_B, size);
//     cudaMalloc((void**)&d_C, size);

//     cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

//     int blockSize = 256;
//     int gridSize = (n + blockSize - 1) / blockSize;
//     vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);

//     cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

//     print_head(h_C);

//     cudaFree(d_A);
//     cudaFree(d_B);
//     cudaFree(d_C);

//     free(h_A);
//     free(h_B);
//     free(h_C);

//     return 0;
// }
