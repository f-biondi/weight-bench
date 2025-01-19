#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda_fp16.h>
#include <cuda.h>
#include <chrono>
#include <cusparse.h>

#define THREAD_N 256

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

void checkCUDAError(const char*);
int* read_file_graph(int* edge_n, int* node_n, int* max_node_w);
int read_file_int(FILE *file);
__half* gen_ones(int n);

__global__ void compute_weight_mask(int *node_n, __half *weight_mask, int *node_blocks, int *splitters, int *splitters_mask, int *current_splitter_index) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int node_number = (*node_n);
    if(i < node_number) {
        int csi = *current_splitter_index;
        int splitter = splitters[csi];
        int block = node_blocks[i];
        weight_mask[i] = block == splitter ? 1.0 : 0.0;
        splitters_mask[splitter] = 0;
    }
}

int main(void) {
    int node_n = 0;
    int edge_n = 0;
    int max_node_w = 0;
    int* edge_index = read_file_graph(&edge_n, &node_n, &max_node_w);
    __half* values = gen_ones(edge_n);
    const int BLOCK_N = (node_n+(THREAD_N-1)) / THREAD_N;
    size_t node_size = node_n * sizeof(int);
    size_t edge_size = edge_n * sizeof(int);
    int current_splitter_index = 0;

    int *d_node_n, *d_new_node_blocks, *d_node_blocks, *d_current_splitter_index,
        *d_max_node_w, *d_splitters, *d_splitters_mask, *d_weight_adv, *d_swap,
        *d_rows, *d_columns;

    __half *d_weights, *d_weight_mask, *d_values;

    int* rows = (int*)malloc((node_n+1) * sizeof(int));
    int last = 0;
    int c = 0;
    for(c=0; c< edge_n; ++c) {
        if(!c || edge_index[c] != edge_index[c-1]) {
            for(int k=edge_index[c-1]; k < (edge_index[c]-1); ++k) {
                rows[last] = c;
                ++last;
            }
            rows[last] = c;
            ++last;
        }
    }
    rows[last] = c;

    CHECK_CUDA( cudaMalloc((void **)&d_weights, node_n * sizeof(__half)) );
    CHECK_CUDA( cudaMalloc((void **)&d_weight_mask, node_n * sizeof(__half)) );
    CHECK_CUDA( cudaMalloc((void **)&d_weight_adv, node_size * max_node_w) );
    CHECK_CUDA( cudaMalloc((void **)&d_node_n, sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void **)&d_new_node_blocks, node_size) );
    CHECK_CUDA( cudaMalloc((void **)&d_node_blocks, node_size) );
    CHECK_CUDA( cudaMalloc((void **)&d_splitters, node_size) );
    CHECK_CUDA( cudaMalloc((void **)&d_splitters_mask, node_size) );
    CHECK_CUDA( cudaMalloc((void **)&d_current_splitter_index, sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void **)&d_max_node_w, sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void **)&d_rows, (node_n+1) * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void **)&d_columns, edge_size) );
    CHECK_CUDA( cudaMalloc((void **)&d_values, edge_n * sizeof(__half)) );

    CHECK_CUDA( cudaMemset(d_weights, 0, node_n * sizeof(__half)) );
    CHECK_CUDA( cudaMemset(d_new_node_blocks, 0, node_size) );
    CHECK_CUDA( cudaMemset(d_node_blocks, 0, node_size) );
    CHECK_CUDA( cudaMemset(d_splitters, 0, node_size) );
    CHECK_CUDA( cudaMemset(d_splitters_mask, 0, node_size) );
    CHECK_CUDA( cudaMemset(d_current_splitter_index, 0, sizeof(int)) );

    CHECK_CUDA( cudaMemcpy(d_node_n, &node_n, sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(d_max_node_w, &max_node_w, sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(d_rows, rows, (node_n+1) * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(d_columns, edge_index + edge_n, edge_size, cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(d_values, values, edge_n * sizeof(__half), cudaMemcpyHostToDevice) );

    cusparseHandle_t handle = NULL;
    cusparseSpMatDescr_t adj_mat;
    cusparseDnVecDescr_t vecX, vecY;
    void* dBuffer = NULL;
    float alpha = 1.0;
    float beta = 0;
    size_t bufferSize = 0;

    CHECK_CUSPARSE( cusparseCreate(&handle) );

    CHECK_CUSPARSE( cusparseCreateCsr(&adj_mat, node_n, node_n, edge_n,
                                  d_rows, d_columns, d_values,  CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                  CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F) );
    // Create dense vector X
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecX, node_n, d_weight_mask, CUDA_R_16F) );
    // Create dense vector y
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecY, node_n, d_weights, CUDA_R_16F) );
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMV_bufferSize(
                                 handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, adj_mat, vecX, &beta, vecY, CUDA_R_32F,
                                 CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize) );

    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )
    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();

    compute_weight_mask<<<BLOCK_N, THREAD_N>>>(d_node_n, d_weight_mask, d_node_blocks, d_splitters, d_splitters_mask, d_current_splitter_index);

    CHECK_CUSPARSE( cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, adj_mat, vecX, &beta, vecY, CUDA_R_32F,
                                 CUSPARSE_SPMV_ALG_DEFAULT, dBuffer) );

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    printf("%f\n",std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000000.0);
    /*__half *weights = (__half*)malloc(node_n * sizeof(__half));
    CHECK_CUDA( cudaMemcpy(weights, d_weights, node_n * sizeof(__half), cudaMemcpyDeviceToHost) );
    for(int i=0; i<node_n; ++i) {
        printf("%f ", ((float) weights[i]));
    }
    printf("\n");*/


    return 0;
}

int* read_file_graph(int* edge_n, int* node_n, int* max_node_w) {
    FILE *file = fopen("graph.txt", "r");
    *node_n = read_file_int(file);
    *edge_n = read_file_int(file);
    int* weights = (int*)calloc(*node_n, sizeof(int));
    size_t index_size = (*edge_n) * 2 * sizeof(int);
    int *edge_index = (int*)malloc(index_size);
    for(int i=0; i<(*edge_n); ++i) {
        edge_index[i] = read_file_int(file);
        edge_index[(*edge_n) + i] = read_file_int(file);
        weights[edge_index[i]]++;
        if(weights[edge_index[i]] > *max_node_w) {
            *max_node_w = weights[edge_index[i]];
        }
    }
    free(weights);
    return edge_index;
}

int read_file_int(FILE *file) {
    char ch = fgetc(file);
    int n = 0;
    int c = 0;
    while(ch != ' ' && ch != '\n') {
        c = ch - '0';
        n = (n*10) + c;
        ch = fgetc(file);
    }
    return n;
}

__half* gen_ones(int n) {
    __half* values = (__half*)malloc(n * sizeof(__half));
    for(int i=0; i<n; ++i) values[i] = 1.0;
    return values;
}
