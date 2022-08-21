// remember!
#include <bits/stdc++.h>
#include "Utils.h"
#include "ScoreMatrix.h"

using namespace std;

#define MAX_LEN 1024
#define debug(A) cout << #A << ": " << A << endl

// #define DEBUG

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
    // char* cur_str = str_ptrs[index];
    if (index < num_strings)
        printf("index: %d, num_strings: %d, string: %s\n", index, num_strings, str_ptrs[index]);
}

void allcoate_strings_on_device_flattened(vector<string> strings, char ** d_fstr_addr, int ** d_fids_addr, int num_strs) {
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
void free_strings_on_device_flattened(char* d_fstr, int* d_fids) {
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

void allocate_strings_on_device_ptr2ptr(vector<string> strings, char *** d_str_ptrs_addr, char *** d_temp_strs_addr, int num_strs) {
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

void free_strings_on_device_ptr2ptr(char** d_str_ptrs, char** d_temp_strs, int num_strs) {
    cudaFree(d_str_ptrs);
    for (int i = 0; i < num_strs; i++) {
        cudaFree(d_temp_strs[i]);
    }
}

__global__ void local_ungapped_alignemnt(
    int aa2num_len,
    int score_matrix_len,
    int rows_memory_len,
    int q_tmax_len,
    int max_diagonal_size,
    char* target_flat_str,
    int* flat_ids,
    char* query,
    int q_len
) {

    // assumptions:
    // - the 3rd parameter of kernel call is provided (for dynamic allocation of shared memory)
    // - the block size (or number of threads) equals query size
    // - the target is spreaded on the row (axis=0) and the query on columns (axis=1)

    // __shared__ int best_score;
    // __shared__ int best_diag_idx;
    // the type doesn't actually matter!
    // todo: make it global maybe
    extern __shared__ int shared_memory[];

    // dynamically divide the whole shared memory for the shared memory variables
    int* aa2num = (int *)shared_memory;
    int* score_matrix_flat = (int *)&aa2num[aa2num_len];
    // rows_memory_len = 2 * (q_len + 1)
    int* last_row = (int *)&score_matrix_flat[score_matrix_len];
    int* current_row = (int *)&last_row[q_len + 1];
    char* q_tmax = (char *)&current_row[q_len + 1];
    char* q_cache = q_tmax;
    char* t_cache = (char *)&q_tmax[q_len];
    opt_cell* best_cells = (opt_cell*)&t_cache[q_tmax_len];

    int bid = blockIdx.x; int tid =  threadIdx.x; int bsize = blockDim.x; int tnum = bsize;
    
    // todo: extract a function for this repeated allocations if possible
    // initialize and retrieve content of shared memory variables
    for (int i = 0; i < (aa2num_len + tnum - 1) / tnum; i++) {
        int idx = i * tnum + tid;
        if (idx < aa2num_len) {
            aa2num[idx] = _aa2num[idx];
        }
    }

    for (int i = 0; i < (score_matrix_len + tnum - 1) / tnum; i++) {
        int idx = i * tnum + tid;
        if (idx < score_matrix_len) {
            score_matrix_flat[idx] = _score_matrix[idx];
        }
    }

    // sth to make it branchless :(
    // maybe we shouldn't store (q_len + 1) cells of rows but in that case we would probably have to have a branch
    last_row[0] = 0;
    // Note: thread with index i works on column i+1 of alignment matrix
    last_row[tid + 1] = 0;
    // current row does not need initialization!
    
    q_cache[tid] = query[tid];

    int t_start_idx = flat_ids[bid];
    int t_len = flat_ids[bid + 1] - t_start_idx;
    for (int i = 0; i < (t_len + tnum - 1) / tnum; i++) {
        int idx = i * tnum + tid;
        if (idx < t_len) {
            t_cache[idx] = target_flat_str[idx + t_start_idx];
        }
    }

    int diagonal_size = t_len + q_len - 1;
    for (int i = 0; i < (diagonal_size + tnum - 1) / tnum; i++) {
        int idx = i * tnum + tid;
        if (idx < diagonal_size) {
            best_cells[idx].row = -1, best_cells[idx].score = -1, best_cells[idx].diagonal_idx = idx;
        }
    }

    __syncthreads();

    // now actual alignment algorithm can begin :D

    // ignore these lines!
#ifdef DEBUG    
    int cur_str_len = t_len;
    char* cur_str = (char *)malloc((cur_str_len + 1) * sizeof(char));
    memcpy(cur_str, t_cache, cur_str_len);
    // printf("%d\n", t_len);
    // for(int i = 0; i < t_len; i++)
    //     printf("%d:%c| ", i, t_cache[i]);
    // printf("\n");
    // cur_str[cur_str_len] = '\0';
    printf("id: %d, length: %d, query: %s, target: %s\n", tid, cur_str_len, query, cur_str);
#endif
} 

int main() {
    vector<string> queries, targets;
    init_input_from_file("TestSamples/queries.txt", queries, false);
    init_input_from_file("TestSamples/targets2.txt", targets, false);
#ifdef DEBUG
    for (auto &q: queries)
        debug(q);
    for (auto &t: targets)
        cout << t.size() << ": " << t << endl;
#endif

/* some todos for more efficiency:
    - merge minor memory transactions
    - use pinned memory
    - clean up this mess!
*/
    int max_target_len = 0;
    for (auto &t: targets)
        max_target_len = max(max_target_len, (int)t.size());
    char * q_str = (char *) queries[0].c_str();
    int q_len = queries[0].size();
    char * d_query_str;
    cudaMalloc(&d_query_str, q_len * sizeof(char));
    cudaMemcpy(d_query_str, q_str, q_len * sizeof(char), cudaMemcpyHostToDevice);
    // flatten method (with c++ string appending)
    int num_target_strs = targets.size();
    char * d_targets_fstr; int * d_target_indices;
    allcoate_strings_on_device_flattened(targets, &d_targets_fstr, &d_target_indices, num_target_strs);
    // test_kernel_flattened<<<1, num_target_strs>>> (d_targets_fstr, d_target_indices, num_target_strs);

    cudaMemcpyToSymbol(_score_matrix, score_matrix_flattened, SCORE_MATRIX_SIZE * sizeof(int));
    cudaMemcpyToSymbol(_aa2num, aa2num, int('Z' - 'A') * sizeof(int));
    int aa2num_len = int('Z' - 'A');
    int score_matrix_len = SCORE_MATRIX_SIZE;

    int rows_memory_len = 2 * (q_len + 1); // current row + last row
    int q_tmax_len = (max_target_len + q_len);
    int max_diagonal_size = (max_target_len + q_len - 1);
    // calculating shared memory size in bytes
    int shared_memory_size = aa2num_len * sizeof(int)
     + score_matrix_len  * sizeof(int) 
     + rows_memory_len * sizeof(int)
     + q_tmax_len  * sizeof(char) 
     + max_diagonal_size * sizeof(opt_cell)
     ;

    local_ungapped_alignemnt<<<num_target_strs, q_len, shared_memory_size>>> (
        aa2num_len,
        score_matrix_len,
        rows_memory_len,
        q_tmax_len,
        max_diagonal_size,
        d_targets_fstr,
        d_target_indices,
        d_query_str,
        q_len
    );
    cudaDeviceSynchronize();
    free_strings_on_device_flattened(d_targets_fstr, d_target_indices);
    cudaFree(d_query_str);
    
    // array of pointers to string method
#ifdef DEBUG
    int num_strs = queries.size();
    char ** d_str_ptrs;
    char ** d_temp_strs;
    allocate_strings_on_device_ptr2ptr(queries, &d_str_ptrs, &d_temp_strs, num_strs);
    test_kernel_ptr2ptr<<<1, num_strs>>> (d_str_ptrs, num_strs);
    cudaDeviceSynchronize();
    free_strings_on_device_ptr2ptr(d_str_ptrs, d_temp_strs, num_strs);
#endif
    return 0;
}