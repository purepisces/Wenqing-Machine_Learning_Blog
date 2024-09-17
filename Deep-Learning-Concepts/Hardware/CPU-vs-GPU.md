#### Compare CPU vs GPU

-   **CPU**: A CPU has a small number of cores (often 4 to 16 in consumer devices, more in high-performance systems), each capable of handling multiple threads using techniques like **hyper-threading**. CPUs are optimized for sequential tasks and complex operations, where each core handles a few threads with high individual performance. They can do parallel work but are primarily designed for versatility and handling different types of tasks efficiently.
    
-   **GPU**: A GPU, on the other hand, has **thousands** of smaller, simpler cores designed for massive parallelism. GPUs excel at handling many threads simultaneously, making them ideal for tasks where a large number of simple operations need to be performed in parallel, such as processing large datasets or running machine learning models.

Let’s consider a simple task: **adding two arrays element-wise**. You have two arrays, `A` and `B`, each with 8 elements, and you want to compute a new array `C` where `C[i] = A[i] + B[i]`.

##### Example 1: CPU (Sequential Execution)

On a CPU, the task is typically performed **sequentially**. This means that each addition is performed one after another, using a single processing core or thread.

```cpp
// CPU version: Sequential execution
void add_arrays_cpu(const int* A, const int* B, int* C, size_t size) {
    for (size_t i = 0; i < size; i++) {
        C[i] = A[i] + B[i];  // Perform one addition at a time
    }
}
```
Here’s what happens:

1.  The CPU picks the first element (`A[0] + B[0]`) and stores the result in `C[0]`.
2.  Then it moves to the second element (`A[1] + B[1]`) and stores the result in `C[1]`.
3.  This continues until all 8 additions are performed **sequentially**.

So, if you have 8 elements, the CPU does 8 steps, one at a time.

### Example 2: GPU (Parallel Execution)

On a GPU, tasks can be performed **in parallel** by using multiple threads to handle multiple operations at once. In this case, each thread can be responsible for adding one element of `A` and `B` and storing the result in `C`.
```cpp
// GPU version: Parallel execution
__global__ void add_arrays_gpu(const int* A, const int* B, int* C, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] + B[idx];  // Each thread handles one addition
    }
}
```
Here’s what happens on the GPU:

1.  The GPU launches **8 threads** in parallel, each one responsible for adding a single pair of elements.
2.  Each thread independently computes `C[i] = A[i] + B[i]` for a different index `i` at the same time.

So, if you have 8 elements and launch 8 threads, the GPU can compute all 8 additions **simultaneously** in a single step. This is parallelism.

 **Visualization:**

##### CPU (Sequential):
```css
Time 1: A[0] + B[0] -> C[0]
Time 2: A[1] + B[1] -> C[1]
Time 3: A[2] + B[2] -> C[2]
...
Time 8: A[7] + B[7] -> C[7]
```
##### GPU (Parallel):
```css
Time 1: A[0] + B[0] -> C[0]
         A[1] + B[1] -> C[1]
         A[2] + B[2] -> C[2]
         ...
         A[7] + B[7] -> C[7] (all happen at the same time)
```
On a **CPU**, tasks are typically performed one at a time in sequence, while on a **GPU**, tasks can be broken down into smaller units of work (such as array element addition) and performed in parallel across many threads, significantly speeding up the process. This parallelism is what gives GPUs their massive performance advantage in handling large datasets.

## Reference
