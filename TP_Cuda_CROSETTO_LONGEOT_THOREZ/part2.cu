#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>

#define INPUT_SIZE 32          // Taille de l'entrée 32x32
#define C1_KERNEL_SIZE 5       // Taille des noyaux de convolution 5x5
#define C1_OUTPUT_SIZE 28      // Taille après convolution
#define S1_OUTPUT_SIZE 14      // Taille après pooling
#define NUM_KERNELS 6          // Nombre de noyaux
#define POOL_SIZE 2            // Taille du pool (2x2)
#define STRIDE 2               // Stride pour le pooling

// Fonction d'activation tanh pour GPU
__device__ float activation_tanh(float x) {
    return tanh(x);
}

// Kernel de convolution avec activation tanh
__global__ void cudaConvolution2DWithActivation(float* input, float* output, float* kernel, int input_size, int kernel_size, int output_size, int num_kernels) {
    int k = blockIdx.z;  // Indice du noyau (pour chaque noyau)
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Indice de ligne
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Indice de colonne

    if (i < output_size && j < output_size) {
        float sum = 0.0f;
        for (int m = 0; m < kernel_size; m++) {
            for (int n = 0; n < kernel_size; n++) {
                int x = i + m;
                int y = j + n;
                if (x < input_size && y < input_size) {
                    sum += input[x * input_size + y] * kernel[k * kernel_size * kernel_size + m * kernel_size + n];
                }
            }
        }
        // Activation tanh
        output[k * output_size * output_size + i * output_size + j] = activation_tanh(sum);
        //output[k * output_size * output_size + i * output_size + j] = sum ;
    }
}

// Kernel de pooling (average pooling)
__global__ void cudaAveragePooling(float* input, float* output, int input_size, int output_size, int num_kernels, int pool_size, int stride) {
    int k = blockIdx.z;  // Indice du noyau (pour chaque noyau)
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Indice de ligne
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Indice de colonne

    if (i < output_size && j < output_size) {
        float sum = 0.0f;
        for (int m = 0; m < pool_size; m++) {
            for (int n = 0; n < pool_size; n++) {
                int x = i * stride + m;
                int y = j * stride + n;

                if (x < input_size && y < input_size) {
                    sum += input[k * input_size * input_size + x * input_size + y];
                }
            }
        }
        output[k * output_size * output_size + i * output_size + j] = sum / (pool_size * pool_size); // Moyenne pour average pooling
    }
}

// Fonction pour initialiser une matrice avec des valeurs comprises entre 0 et 1
void initializeRandomMatrix(float* matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = (float)rand() / (float)RAND_MAX;  // Valeur aléatoire entre 0 et 1
    }
}

// Fonction pour initialiser une matrice à 0
void initializeZeroMatrix(float* matrix, int size) {
    for (int i = 0; i < size; i++) {
        matrix[i] = 0.0f;
    }
}

// Fonction pour afficher une matrice 1D sous forme 2D ou 3D
void MatrixPrint(float* matrix, int depth, int rows, int cols) {
    if (depth > 1) {
        for (int d = 0; d < depth; d++) {
            printf("Profondeur %d :\n", d + 1);
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    printf("%0.4f ", matrix[d * rows * cols + i * cols + j]);
                }
                printf("\n");
            }
            printf("\n");
        }
    } else {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                printf("%0.4f ", matrix[i * cols + j]);
            }
            printf("\n");
        }
    }
}

// Fonction pour vérifier les erreurs CUDA
void checkCudaError(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        printf("CUDA error: %s at %s\n", cudaGetErrorString(error), message);
        exit(-1);
    }
}

int main() {
    srand(time(NULL));

    // Allocation de la mémoire pour les matrices CPU
    float* raw_data = (float*)malloc(INPUT_SIZE * INPUT_SIZE * sizeof(float));
    float* C1_data = (float*)malloc(NUM_KERNELS * C1_OUTPUT_SIZE * C1_OUTPUT_SIZE * sizeof(float));
    float* S1_data = (float*)malloc(NUM_KERNELS * S1_OUTPUT_SIZE * S1_OUTPUT_SIZE * sizeof(float));
    float* C1_kernel = (float*)malloc(NUM_KERNELS * C1_KERNEL_SIZE * C1_KERNEL_SIZE * sizeof(float));

    // Initialisation des matrices
    initializeRandomMatrix(raw_data, INPUT_SIZE * INPUT_SIZE);  // Valeurs aléatoires entre 0 et 1 pour raw_data
    initializeZeroMatrix(C1_data, NUM_KERNELS * C1_OUTPUT_SIZE * C1_OUTPUT_SIZE);  // Initialisation à zéro pour C1_data
    initializeZeroMatrix(S1_data, NUM_KERNELS * S1_OUTPUT_SIZE * S1_OUTPUT_SIZE);  // Initialisation à zéro pour S1_data
    initializeRandomMatrix(C1_kernel, NUM_KERNELS * C1_KERNEL_SIZE * C1_KERNEL_SIZE);  // Valeurs aléatoires pour C1_kernel

    // Pointeurs GPU
    float *d_raw_data, *d_C1_data, *d_S1_data, *d_C1_kernel;

    // Allocation mémoire sur le GPU
    cudaMalloc((void**)&d_raw_data, INPUT_SIZE * INPUT_SIZE * sizeof(float));
    cudaMalloc((void**)&d_C1_data, NUM_KERNELS * C1_OUTPUT_SIZE * C1_OUTPUT_SIZE * sizeof(float));
    cudaMalloc((void**)&d_S1_data, NUM_KERNELS * S1_OUTPUT_SIZE * S1_OUTPUT_SIZE * sizeof(float));
    cudaMalloc((void**)&d_C1_kernel, NUM_KERNELS * C1_KERNEL_SIZE * C1_KERNEL_SIZE * sizeof(float));

    // Copie des données vers le GPU
    cudaMemcpy(d_raw_data, raw_data, INPUT_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C1_kernel, C1_kernel, NUM_KERNELS * C1_KERNEL_SIZE * C1_KERNEL_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Définition des dimensions des blocs et des grilles
    dim3 blockDim(16, 16);
    dim3 gridDim((C1_OUTPUT_SIZE + blockDim.x - 1) / blockDim.x, (C1_OUTPUT_SIZE + blockDim.y - 1) / blockDim.y, NUM_KERNELS);

    // Lancer la convolution avec activation
    cudaConvolution2DWithActivation<<<gridDim, blockDim>>>(d_raw_data, d_C1_data, d_C1_kernel, INPUT_SIZE, C1_KERNEL_SIZE, C1_OUTPUT_SIZE, NUM_KERNELS);
    checkCudaError(cudaGetLastError(), "cudaConvolution2DWithActivation");

    // Copier les résultats de la convolution
    cudaMemcpy(C1_data, d_C1_data, NUM_KERNELS * C1_OUTPUT_SIZE * C1_OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Affichage des résultats après convolution et activation
    printf("\nRésultat de la convolution avec activation tanh (C1_data) :\n");
    MatrixPrint(C1_data, NUM_KERNELS, C1_OUTPUT_SIZE, C1_OUTPUT_SIZE);

    // Lancer le pooling (average pooling)
    dim3 poolGridDim((S1_OUTPUT_SIZE + blockDim.x - 1) / blockDim.x, (S1_OUTPUT_SIZE + blockDim.y - 1) / blockDim.y, NUM_KERNELS);
    cudaAveragePooling<<<poolGridDim, blockDim>>>(d_C1_data, d_S1_data, C1_OUTPUT_SIZE, S1_OUTPUT_SIZE, NUM_KERNELS, POOL_SIZE, STRIDE);
    checkCudaError(cudaGetLastError(), "cudaAveragePooling");

    // Copier les résultats du pooling
    cudaMemcpy(S1_data, d_S1_data, NUM_KERNELS * S1_OUTPUT_SIZE * S1_OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Affichage des résultats après pooling
    printf("\nRésultat après pooling (S1_data) :\n");
    MatrixPrint(S1_data, NUM_KERNELS, S1_OUTPUT_SIZE, S1_OUTPUT_SIZE);

    // Libération de la mémoire
    cudaFree(d_raw_data);
    cudaFree(d_C1_data);
    cudaFree(d_S1_data);
    cudaFree(d_C1_kernel);

    free(raw_data);
    free(C1_data);
    free(S1_data);
    free(C1_kernel);

    return 0;
}