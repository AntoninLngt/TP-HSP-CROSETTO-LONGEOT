# TP-HSP-CROSETTO-LONGEOT
CNN Implemetation Hardware for signal processing

**The ultimate goal of these 4 sessions is to implement the inference of a very clear CNN: LeNet-5 proposed by Yann LeCun et al.
by Yann LeCun et al. in 1998 for handwritten digit recognition**

![image](https://github.com/user-attachments/assets/a04826d8-fcad-47c6-a7a1-c68daecdf5d3)

## 29/11 : First Practical
In this session we worked on a knowing CUDAwith Matrix multiplication :
- Design multiplication function in Cuda
- Understanding the difference between the CPU (sequential) & the GPU (parallelization)
- Compare the time for running a programm under GPU & CPU

For do that we defined and worked on the functions below : 
````
// Matrix  Initialization
void MatrixInit(float *M, int n, int p);

// Print Matrix 
void MatrixPrint(float *M, int n, int p);

// Addition (CPU)
void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p);

// Addition (GPU)
void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p);

// Multiplication (CPU)
void MatrixMult(float *M1, float *M2, float *Mout, int n);

// Multiplication (GPU)
__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n);
````
In Cuda, the indexes of our matrix are accessed in a different way, with a call to Grid, Block, which is represented below:





![time_CPU (s) and time_GPU (s) in logarithmic scale](https://github.com/user-attachments/assets/5f195eac-7eba-4fd0-95b3-e63b0740030c)
This is a schema representing the comparison between CPU and GPU multiplication of 2 N_DIM Matrix. 


## 03/12 : Second Practical
In this session we worked on the first LeNet-5 layers.
- Layer 1 - Test data generation, we realized 1D tabular initialization where each case is corresponding to one element

**N=32x32, 6x28x28, 6x14x14 et 6x5x5**
