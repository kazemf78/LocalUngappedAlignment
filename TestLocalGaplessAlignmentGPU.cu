// remember!
#include <bits/stdc++.h>
#include "Utils.h"
#include "ScoreMatrix.h"

using namespace std;

#define MAX_LEN 1024
#define debug(A) cout << #A << ": " << A << endl

__constant__ int _aa2num[(int)'Z' - 'A'];
__constant__ int _score_matrix[ALPH_SIZE * ALPH_SIZE];

__global__ void test_kernel_flattened(char* flat_str, int* ids, int num_strs) {
    int index = threadIdx.x;
    int cur_str_len = ids[index+1] - ids[index];
    char* cur_str = (char *)malloc((cur_str_len + 1) * sizeof(char));
    memcpy(cur_str, flat_str + ids[index], cur_str_len);
    cur_str[cur_str_len] = '\0';
    printf("length: %d, string: %s\n", cur_str_len, cur_str);
}

__global__ void test_kernel_ptr2ptr(char** str_ptrs, int num_strings) {

    int index = threadIdx.x;
    char* cur_str = str_ptrs[index];
    if (index < num_strings)
        printf("string: %s\n", cur_str);
}

void allcoate_device_flattened(vector<string> strings, char ** d_fstr_addr, int ** d_fids_addr, int num_strs) {
    string flat_temp; int* flat_ids; char * flat_str;
    flat_ids = (int *) malloc((num_strs + 1) * sizeof(int));
    int cur_ptr = 0;
    // todo: maybe implementation of flattening can be more efficient but it's more readable now, improve it later on?
    for (int i = 0; i < strings.size(); i++) {
        flat_ids[i] = cur_ptr;
        flat_temp += strings[i];
        cur_ptr += strings[i].size();
    }
    flat_ids[num_strs] = cur_ptr;
    int total_chars_num = cur_ptr;

    flat_str = (char *) flat_temp.c_str();
    cudaMalloc(d_fstr_addr, total_chars_num * sizeof(char));
    cudaMalloc(d_fids_addr, (num_strs + 1) * sizeof(int));
    cudaMemcpy(*d_fstr_addr, flat_str, total_chars_num * sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(*d_fids_addr, flat_ids, (num_strs + 1) * sizeof(int), cudaMemcpyHostToDevice);
}

// maybe a redundant function!
void free_device_flattened(char* d_fstr, int* d_fids) {
    cudaFree(d_fstr);
    cudaFree(d_fids);
}

char ** allocate_device_ptr2ptr_without_free(vector<string> strings, int num_strs) {
    char ** d_str_ptrs;
    cudaMalloc(&d_str_ptrs, num_strs * sizeof(char *));
    char * d_temp_strs[num_strs];
    for (int i = 0; i < num_strs; i++) {
        int q_i_len = strings[i].size();
        cudaMalloc(&d_temp_strs[i],  q_i_len * sizeof(char));
        cudaMemcpy(d_temp_strs[i], strings[i].c_str(), q_i_len * sizeof(char), cudaMemcpyHostToDevice);
        cudaMemcpy(d_str_ptrs + i, &d_temp_strs[i], sizeof(char *), cudaMemcpyHostToDevice);
    }
    return d_str_ptrs;
}

void allocate_device_ptr2ptr(vector<string> strings, char *** d_str_ptrs_addr, char *** d_temp_strs_addr, int num_strs) {
    cudaMalloc(d_str_ptrs_addr, num_strs * sizeof(char *));
    *d_temp_strs_addr = (char **) malloc(num_strs * sizeof(char*));
    char** d_str_ptrs = *d_str_ptrs_addr;
    char** d_temp_strs = *d_temp_strs_addr;
    for (int i = 0; i < num_strs; i++) {
        int q_i_len = strings[i].size();
        cudaMalloc(&d_temp_strs[i],  q_i_len * sizeof(char));
        cudaMemcpy(d_temp_strs[i], strings[i].c_str(), q_i_len * sizeof(char), cudaMemcpyHostToDevice);
        cudaMemcpy(d_str_ptrs + i, &d_temp_strs[i], sizeof(char *), cudaMemcpyHostToDevice);
    }
}

void free_device_ptr2ptr(char** d_str_ptrs, char** d_temp_strs, int num_strs) {
    cudaFree(d_str_ptrs);
    for (int i = 0; i < num_strs; i++) {
        cudaFree(d_temp_strs[i]);
    }
}

__global__ void test() {
    for (int i = 0; i < ALPH_SIZE; i++) {
        for (int j = 0; j < ALPH_SIZE; j++)
            printf("%3d ", _score_matrix[i*ALPH_SIZE+j]);
        printf("\n");
    }
    for (char c = 'A'; c < 'Z'; c++)
        printf("%c:%d, ", c, _aa2num[(int) c - 'A']);
    printf("\n");
}

int main() {

    init_score_matrix();
    for (int i = 0; i < ALPH_SIZE; i++) {
        for (int j = 0; j < ALPH_SIZE; j++)
            printf("%3d ", score_matrix_flattened[i*ALPH_SIZE+j]);
        printf("\n");
    }
    printf("\n");
    cudaMemcpyToSymbol(_score_matrix, score_matrix_flattened, SCORE_MATRIX_SIZE * sizeof(int));
    cudaMemcpyToSymbol(_aa2num, aa2num, int('Z' - 'A') * sizeof(int));
    test<<<1, 1>>>();
    cudaDeviceSynchronize();
    
    vector<string> queries, targets;
    init_input_from_file("TestSamples/queries.txt", queries, false);
    init_input_from_file("TestSamples/targets.txt", targets, true);
    for (auto &q: queries)
        debug(q);

    int num_strs = queries.size();

    // flatten method (with c++ string appending)
    int * d_fids; char * d_fstr;
    allcoate_device_flattened(queries, &d_fstr, &d_fids, num_strs);
    test_kernel_flattened<<<1, num_strs>>> (d_fstr, d_fids, num_strs);
    cudaDeviceSynchronize();
    free_device_flattened(d_fstr, d_fids);

    // array of pointers to string method
    char ** d_str_ptrs;
    char ** d_temp_strs;
    allocate_device_ptr2ptr(queries, &d_str_ptrs, &d_temp_strs, num_strs);
    test_kernel_ptr2ptr<<<1, num_strs>>> (d_str_ptrs, num_strs);
    cudaDeviceSynchronize();
    free_device_ptr2ptr(d_str_ptrs, d_temp_strs, num_strs);
    
    return 0;
}