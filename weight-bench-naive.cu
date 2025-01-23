#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <chrono>

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

typedef struct {
    int* index;
    int* edge_offset;
} edge_index_t;

void checkCUDAError(const char*);
edge_index_t read_file_graph(int* edge_n, int* node_n, int* max_node_w);
int read_file_int(FILE *file);

__global__ void compute_edge_mask(int* edge_n, int* edge_end, unsigned char* edge_mask, int* node_blocks, int* splitters, int* current_splitter_index) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < *edge_n) {
        int csi = *current_splitter_index;
        int splitter = splitters[csi];
        int node = edge_end[i];
        int block = node_blocks[node];
        edge_mask[i] = (block == splitter);
    }
}

__global__ void compute_weights(int* edge_n, int* edge_start,int* edge_end, unsigned char* edge_mask, int* node_blocks, int* splitters, int* current_splitter_index,int* weights) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < *edge_n) {
        int csi = *current_splitter_index;
        int splitter = splitters[csi];
        int node = edge_end[i];
        int block = node_blocks[node];
        int s = edge_start[i];

        if(block == splitter) {
            atomicAdd(weights + s, 1);
        }
    }
}

int main(void) {
    int node_n = 0;
    int edge_n = 0;
    int max_node_w = 0;
    edge_index_t edge_index = read_file_graph(&edge_n, &node_n, &max_node_w);
    const int BLOCK_N = (node_n+(THREAD_N-1)) / THREAD_N;
    int current_splitter_index = 0;

    int *d_node_n, *d_edge_n, *d_node_blocks, *d_current_splitter_index,
        *d_max_node_w, *d_splitters, *d_weight_adv, *d_edge_offset, *d_edge_start, *d_edge_end;

    int *d_weights;

    unsigned char *d_edge_mask;

    CHECK_CUDA( cudaMalloc((void **)&d_weights, node_n * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void **)&d_edge_mask, edge_n * sizeof(unsigned char)) );
    CHECK_CUDA( cudaMalloc((void **)&d_edge_start, edge_n * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void **)&d_edge_end, edge_n * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void **)&d_edge_offset, node_n * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void **)&d_weight_adv, node_n * sizeof(int) * max_node_w) );
    CHECK_CUDA( cudaMalloc((void **)&d_node_n, sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void **)&d_edge_n, sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void **)&d_node_blocks, node_n * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void **)&d_splitters, node_n * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void **)&d_current_splitter_index, sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void **)&d_max_node_w, sizeof(int)) );

    CHECK_CUDA( cudaMemset(d_weights, 0, node_n * sizeof(int)) );
    CHECK_CUDA( cudaMemset(d_node_blocks, 0, node_n * sizeof(int)) );
    CHECK_CUDA( cudaMemset(d_splitters, 0, node_n * sizeof(int)) );
    CHECK_CUDA( cudaMemset(d_current_splitter_index, 0, sizeof(int)) );

    CHECK_CUDA( cudaMemcpy(d_node_n, &node_n, sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(d_edge_n, &edge_n, sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(d_edge_start, edge_index.index, sizeof(int) * edge_n, cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(d_edge_end, edge_index.index + edge_n, sizeof(int) * edge_n, cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(d_edge_offset, edge_index.edge_offset, sizeof(int) * node_n, cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(d_max_node_w, &max_node_w, sizeof(int), cudaMemcpyHostToDevice) );

    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();

    //compute_edge_mask<<<(edge_n+(THREAD_N-1)) / THREAD_N, THREAD_N>>>(d_edge_n, d_edge_end, d_edge_mask, d_node_blocks, d_splitters, d_current_splitter_index);
    //compute_weights<<<(edge_n+(THREAD_N-1)) / THREAD_N, THREAD_N>>>(d_edge_n, d_edge_start, d_edge_offset, d_edge_mask, d_weights);
    compute_weights<<<(edge_n+(THREAD_N-1)) / THREAD_N, THREAD_N>>>(d_edge_n, d_edge_start, d_edge_end, d_edge_mask, d_node_blocks, d_splitters, d_current_splitter_index, d_weights);

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();

    printf("%f\n",std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000000.0);



    return 0;
}

edge_index_t read_file_graph(int* edge_n, int* node_n, int* max_node_w) {
    FILE *file = fopen("graph.txt", "r");
    *node_n = read_file_int(file);
    *edge_n = read_file_int(file);
    edge_index_t res;
    int* weights = (int*)calloc(*node_n, sizeof(int));
    res.index = (int*)malloc(((size_t)*edge_n) * 2 * sizeof(int));
    res.edge_offset = (int*)malloc((*node_n) * sizeof(int));
    for(int i=0; i<(*edge_n); ++i) {
        res.index[i] = read_file_int(file);
        res.index[(*edge_n) + i] = read_file_int(file);
        weights[res.index[i]]++;
        if(weights[res.index[i]] > *max_node_w) {
            *max_node_w = weights[res.index[i]];
        }
    }
    unsigned long long tot = 0;
    for(int i=0; i<(*node_n); ++i) {
        tot+=max(weights[i], 128);
        res.edge_offset[i] = i ? res.edge_offset[i-1] + weights[i] : weights[i];
    }
    free(weights);
    return res;
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
