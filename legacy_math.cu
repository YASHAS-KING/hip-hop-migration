#include <iostream>

// CUDA Kernel function to add the elements of two arrays
__global__ void addVectors(int* A, int* B, int* C, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int size = 1000;
    int *A, *B, *C;

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&A, size * sizeof(int));
    cudaMallocManaged(&B, size * sizeof(int));
    cudaMallocManaged(&C, size * sizeof(int));

    // Initialize the vectors on the host
    for (int i = 0; i < size; i++) {
        A[i] = i;
        B[i] = i * 2;
    }

    // Launch the kernel on the GPU with 10 blocks of 100 threads
    addVectors<<<10, 100>>>(A, B, C, size);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    std::cout << "Vector addition complete!" << std::endl;

    // Free the memory
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}
