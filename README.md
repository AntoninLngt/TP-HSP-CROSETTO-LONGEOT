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

### Design multiplication function in Cuda

We started by define and work on the functions below : 

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

**Some results (Print Matrix) :**

<div align="center">
  
  ![image](https://github.com/user-attachments/assets/64c5fdc6-c2a2-4f79-92e4-f07bcfea493b)
  
</div>
### Understanding the difference between the CPU (sequential) & the GPU (parallelization)

In Cuda, the indexes of our matrix elements are accessed in a different way, with a call to Grid, Block, Threads which are represented below:
<div align="center">
  
  ![image](https://github.com/user-attachments/assets/e03e9463-8ca8-45a9-a953-e673d9e26f82)
  
</div>

This is the concept of the parallelism offered by CUDA. 

1. **Thread**: The smallest unit of execution, responsible for performing computations. Each thread has a unique identifier within its block.
2. **Block**: A group of threads. Threads in a block can share memory (shared memory) and synchronize with each other using barriers.
3. **Grid**: A collection of blocks. Grids organize how blocks are distributed across the GPU.

Each thread is identified by a combination of its indices: `threadIdx`, `blockIdx`, and the block and grid dimensions (`blockDim`, `gridDim`). This allows threads to collaborate and divide workloads dynamically.

> [!TIP]  
>A CPU (Central Processing Unit) is optimized for sequential processing with a few powerful cores handling tasks one at a time or in small batches. In contrast, a GPU (Graphics Processing Unit) excels at parallelization, using thousands of smaller, less powerful cores to process many tasks simultaneously.

Using this theory will show us the difference between CPU and GPU. 

### Compare the time for running a programm under GPU & CPU

We have made several measurements of multiplication times. Each measurement was performed with **different matrix sizes** : 10, 100, 500, 1000, 2000. 

To count the execution time under CPU we used a `clock()` during multiplication. 

![Uploading image.png…]()


For the GPU, the `nvprof` command gives us program information, including execution time. 

We obtain the following values:

<div align="center">
  
  ![time_CPU (s) and time_GPU (s) in logarithmic scale](https://github.com/user-attachments/assets/5f195eac-7eba-4fd0-95b3-e63b0740030c)
  
  Schema representing the comparison between CPU and GPU multiplication of 2 N_DIM Matrix
  
</div>

Results show that for a small size of data, we don't need to use a GPU for computational functions. Moreover passed a certain size above, GPU show drastically that the time of calculation is lower than the time of CPU calculation. 


## 03/12 : Second Practical
In this session we worked on the first LeNet-5 layers.
- Layer 1 - Test data generation, we realized 1D tabular initialization where each case is corresponding to one element

**N=32x32, 6x28x28, 6x14x14 et 6x5x5**
