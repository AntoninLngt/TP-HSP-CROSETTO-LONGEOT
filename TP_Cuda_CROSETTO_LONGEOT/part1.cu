#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

void MatrixInit(float *M, int n, int p){
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < p; ++j) {
            M[i * p + j] = -1.0f + 2.0f * ((float)rand() / RAND_MAX);
        }
    }
}

void MatrixPrint(float *M, int n, int p){
    printf("Generated CPU Matrix :\n");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < p; ++j) {
            printf("%0.2f   ", M[i * p + j]);
        }
        printf("\n");
    }
}

void GPUMatrixPrint(float *M, int n, int p){
    printf("Generated GPU Matrix :\n");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < p; ++j) {
            printf("%0.2f   ", M[i * p + j]);
        }
        printf("\n");
    }
}

void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < p; ++j) {
            Mout[i * p + j] = M1[i * p + j] + M2[i * p + j];
        }
    }
}

void MatrixMult(float *M1, float *M2, float *Mout, int n, int p){
     for (int i = 0; i < n; i++) {         
        for (int j = 0; j < p; j++) {     
            Mout[i * p + j] = 0.0f;          
            for (int k = 0; k < p; k++) { 
                Mout[i * p + j] += M1[i * p + k] * M2[k * p + j];
            }
        }
    }
}
__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < p) {
        int idx = row * p + col;
        Mout[idx] = M1[idx] + M2[idx];
    }
}

__global__ void cudaMatrixMult(float *M1, float *M2, float *Mout, int n, int p){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

if(row < n && col < p) { 
    float value = 0.0f;
    for(int k = 0; k < p; k++) {
        value += M1[row * p + k] * M2[k * p + col];
    }
    Mout[row * p + col] = value;
    }
}

int main(int argc, char *argv[]){
    if (argc != 3) {
        printf("Usage: %s <n (number of rows)> <p (number of columns)>\n", argv[0]);
        return -1; 
    }

    srand(time(0));

    int n = atoi(argv[1]);
    int p = atoi(argv[2]);

    float *M1 = (float *)malloc(n * p * sizeof(float));
    float *M2 = (float *)malloc(n * p * sizeof(float));
    float *Mout = (float *)malloc(n * p * sizeof(float));

    MatrixInit(M1, n, p);
    //MatrixPrint(M1, n, p);

    MatrixInit(M2, n, p);
    //MatrixPrint(M2, n, p);

    //CPU clock 
    clock_t begin = clock();
    MatrixMult(M1, M2, Mout, n, p);
    clock_t end = clock();
    double millisadd = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Finished in %f s\n", millisadd);


    MatrixAdd(M1, M2, Mout, n, p);
    //MatrixPrint(Mout, n, p);

    float *d_M1, *d_M2, *d_Mout;

    cudaMalloc((void**)&d_M1, n * p * sizeof(float));
    cudaMalloc((void**)&d_M2, n * p * sizeof(float));
    cudaMalloc((void**)&d_Mout, n * p * sizeof(float));

    cudaMemcpy(d_M1, M1, n * p * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M2, M2, n * p * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(3, 3); // 3x3 block for 3x3 matrix
    dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                        (p + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaMatrixMult<<<blocksPerGrid, threadsPerBlock>>>(d_M1, d_M2, d_Mout, n, p);


    // Copy the result back to the host and print
    cudaMemcpy(Mout, d_Mout, n * p * sizeof(float), cudaMemcpyDeviceToHost);
    //GPUMatrixPrint(Mout, n, p);

    free(M1);
    free(M2);
    free(Mout);
    cudaFree(d_M1);
    cudaFree(d_M2);
    cudaFree(d_Mout);

    return 0;
}
