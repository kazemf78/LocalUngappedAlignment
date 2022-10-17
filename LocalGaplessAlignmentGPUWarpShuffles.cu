// remember!
#include <bits/stdc++.h>
#include <cooperative_groups.h>
#include "Utils.h"
#include "ScoreMatrix.h"

namespace cg = cooperative_groups;

#define MAX_LEN 1024
#define FULL_MASK 0xffffffff
#define WARP_SIZE 32
#define debug(A) cout << #A << ": " << A << endl
#define ull unsigned long long
#define BENCH_CAP_VAL 200

// #define DEBUG_REDUCE
// #define DEBUG_KERNEL
// #define DEBUG
#define SHOW_KERNEL_CONF
#ifdef DEBUG
    #define SHOW_ALIGNMENT_SCORES
#endif

// #define HANDLE_LONG_QUERY
// #define REDUCE_ALIGNMENT_RESULT
// #define BENCHMARK
// #define DEBUG_BENCH
// #define SORT_RESULTS
// #define SORT_RESULTS_LIMITED

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// the type doesn't actually matter!
extern __shared__ int shared_memory[];

__constant__ byte_type _score_matrix[ALPH_SIZE * ALPH_SIZE];

void allcoate_strings_on_device_flattened(const vector<std::string>& strings, char ** d_fstr_addr, int ** d_fids_addr, int num_strs) {
    std::string flat_temp; int* flat_ids; char * flat_str;
    flat_ids = (int *) malloc((num_strs + 1) * sizeof(int));
    int cur_ptr = 0;
    // todo: maybe implementation of flattening can be more efficient but it's more readable now, improve it later on?
    for (int i = 0; i < strings.size(); i++) {
        flat_ids[i] = cur_ptr;
        flat_temp += strings[i];
        cur_ptr += strings[i].size();
    }
    // convert chars to indexes on aa2num
    for (int i = 0; i < cur_ptr; i++) {
        flat_temp[i] = (char)aa2num[flat_temp[i] - 'A'];
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

void allocate_strings_on_device_ptr2ptr(const vector<std::string>& strings, char *** d_str_ptrs_addr, char *** d_temp_strs_addr, int num_strs) {
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

__device__ void initial_all_reduce(cg::thread_block block, opt_cell * best_cells, unsigned int tid, int& cur_step, int array_size) {
    for (; cur_step > WARP_SIZE; cur_step = (cur_step + 1) / 2) {
        int idx = tid + cur_step;
        if (tid < cur_step && idx < array_size) {
    #ifdef DEBUG_REDUCE
            if (!tid) printf("(tid:%d, idx:%d| cur_size:%d)\n", tid, idx, cur_step);
    #endif
            if (best_cells[tid].score < best_cells[idx].score) {
                best_cells[tid].score = best_cells[idx].score;
                best_cells[tid].diagonal_idx = best_cells[idx].diagonal_idx;
            }
        }
        block.sync();
    #ifdef DEBUG_REDUCE
        if (!tid) for (int j = 0; j < cur_step; j++) {
            printf("score:%d, idx:%d\n", best_cells[j].score, best_cells[j].diagonal_idx);
        }
    #endif
    }
}

__device__ void initial_warp_reduce(opt_cell * best_cells, unsigned int tid, int best_score_param, int best_diag_idx_param) {
    int score_tmp, diag_idx_tmp;
    int best_score = best_score_param;
    int best_diag_idx = best_diag_idx_param;
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        // todo: handle undefined value !!!!!!!!!!!!! or force the threads to be multiple of 32
        score_tmp = __shfl_down_sync(FULL_MASK, best_score, offset);
        diag_idx_tmp = __shfl_down_sync(FULL_MASK, best_diag_idx, offset);
        if (score_tmp > best_score) {
            best_score = score_tmp;
            best_diag_idx = diag_idx_tmp;
        }
    }
    if (tid % WARP_SIZE == 0) {
        best_cells[tid / WARP_SIZE].score = best_score;
        best_cells[tid / WARP_SIZE].diagonal_idx = best_diag_idx;
    }
}

__device__ void final_warp_reduce(volatile opt_cell * best_cells, unsigned int tid, int cur_step, int array_size) {
    for (; cur_step > 0; cur_step = (cur_step + 1) / 2) {
        int idx = tid + cur_step;
        if (tid < cur_step && idx < array_size) {
            if (best_cells[tid].score < best_cells[idx].score) {
                best_cells[tid].score = best_cells[idx].score;
                best_cells[tid].diagonal_idx = best_cells[idx].diagonal_idx;
            }
        }
#ifdef DEBUG_REDUCE
        if (!tid) for (int j = 0; j < cur_step; j++) {
            printf("score:%d, idx:%d\n", best_cells[j].score, best_cells[j].diagonal_idx);
        }
#endif
        if (cur_step == 1)
            break;
    }
}

__global__ void local_ungapped_alignment(
    int score_matrix_len,
    int rows_memory_len,
    int q_tmax_len,
    int opt_cells_size,
    char* target_flat_str,
    int* flat_ids,
    char* query_idx,
    int q_len,
    int_type* best_scores
) {

    // assumptions:
    // - the 3rd parameter of kernel call is provided (for dynamic allocation of shared memory)
    // - the block size (or number of threads) equals query size ([tnum = q_len = bsize] in code)
    // - the target is spreaded on the row (axis=0) and the query on columns (axis=1)

    __shared__ int best_overall_score; best_overall_score = 0;
    // some preprocessing of variables (these would go on registers probably)
    int bid = blockIdx.x; int tid =  threadIdx.x; int bsize = blockDim.x; int tnum = bsize;
    int t_start_idx = flat_ids[bid];
    int t_len = flat_ids[bid + 1] - t_start_idx;

    // dynamically divide the whole shared memory for the shared memory variables
    /* Note1: the order of these allocations actually matters and if not considered, the "misaligned address" error could occur
        for example cuda needs that int* pointers, point to addresses aligned with 32 bytes and if we allocate the char* before int* it's probable this error occurs
       Note2: the size of variables being used here is their actual size not the max size we had to specify for the kernel call
       todo: would be nice if we could free or release the additional space previously reserved but not used here */
    int_type* last_row = (int_type *)shared_memory;
    int_type* current_row = (int_type *)&last_row[q_len + 1];
    opt_cell* best_cells = (opt_cell*)&current_row[q_len + 1];
    char* q_cache = (char *)&best_cells[opt_cells_size];
    char* t_cache = (char *)&q_cache[q_len];
    byte_type* score_matrix_flat = (byte_type *)&t_cache[t_len];
    
    // initialize and retrieve content of shared memory variables
    for (int i = 0; i < (score_matrix_len + tnum - 1) / tnum; i++) {
        int idx = i * tnum + tid;
        if (idx < score_matrix_len) {
            score_matrix_flat[idx] = _score_matrix[idx];
        }
    }

    for (int i = 0; i < (q_len + tnum - 1) / tnum; i++) {
        int idx = i * tnum + tid;
        if (idx < q_len) {
            last_row[idx] = 0;
        }
    }
    // current row does not need initialization!
    
    for (int i = 0; i < (q_len + tnum - 1) / tnum; i++) {
        int idx = i * tnum + tid;
        if (idx < q_len) {
            q_cache[idx] = query_idx[idx];
        }
    }

    for (int i = 0; i < (t_len + tnum - 1) / tnum; i++) {
        int idx = i * tnum + tid;
        if (idx < t_len) {
            t_cache[idx] = target_flat_str[idx + t_start_idx];
        }
    }

    __syncthreads();
    int best_score = 0;
    int best_diag_idx = 0;
    // now actual alignment algorithm can begin :D
    if (tnum >= q_len) {
    for (int row = 1; row <= t_len; row++) {
        int col = tid + 1;
        int mat_idx = q_cache[col-1] * ALPH_SIZE + t_cache[row-1];
        byte_type substitution_score = score_matrix_flat[mat_idx];
        int_type current_score = max(0, last_row[col-1] + substitution_score);
        int_type current_diagonal = row - col + q_len - 1;
        current_row[col] = current_score; // there is no race condition here!

        // each thread works on different diagonal or column (considering columns of threads differ and rows are the same)
        // so there is no race condition here yet either
        if (current_score > best_score) {
            best_score = current_score;
            best_diag_idx = current_diagonal;
            // here we have a race condition to update the best_overall_score and best_overall_diagonal
#if !defined REDUCE_ALIGNMENT_RESULT
            atomicMax(&best_overall_score, current_score);
#endif
        }

        // to reassure that all current_row cells are updated before using them
        __syncthreads();
        // last_row[0] is already zero and there is no need to update the value (btw same can be said for last_row[q_len] considering it will not be used but anyway!)
        last_row[col] = current_row[col];
        __syncthreads();
    }
    } else {
        for (int row = 1; row <= t_len; row++) {
            for (int col = tid + 1; col <= q_len; col += tnum) {
            int mat_idx = q_cache[col-1] * ALPH_SIZE + t_cache[row-1];
            byte_type substitution_score = score_matrix_flat[mat_idx];
            int_type current_score = max(0, last_row[col-1] + substitution_score);
            int_type current_diagonal = row - col + q_len - 1;
            current_row[col] = current_score;
            if (current_score > best_score) {
                best_score = current_score;
                best_diag_idx = current_diagonal;
#if !defined REDUCE_ALIGNMENT_RESULT
                atomicMax(&best_overall_score, current_score);
#endif
            }
            __syncthreads();
            last_row[col] = current_row[col];
            __syncthreads();
            }
        }
    }

#ifdef REDUCE_ALIGNMENT_RESULT
    initial_warp_reduce(best_cells, tid, best_score, best_diag_idx);
    __syncthreads();
    int array_size = (tnum + WARP_SIZE - 1) / WARP_SIZE;
    int cur_step = (array_size + 1) / 2;
    initial_all_reduce(cg::this_thread_block(), best_cells, tid, cur_step, array_size);
    if (tid < WARP_SIZE) {
        final_warp_reduce(best_cells, tid, cur_step, array_size);
    }
    // now best_cells[0] holds the maximum best score
    if (!tid) best_overall_score = best_cells[0].score;
#endif

    if (!tid) {
#if defined REDUCE_ALIGNMENT_RESULT && defined DEBUG_REDUCE
        printf("%d, bid:%d, best_overall_score:%d, best_diag: %d\n", t_len, bid, best_overall_score, best_cells[0].diagonal_idx);
#endif
        best_scores[bid] = best_overall_score;
    }
}

__global__ void local_ungapped_alignment_on_diagonal(
    int score_matrix_len,
    int q_tmax_len,
#ifdef REDUCE_ALIGNMENT_RESULT
    int opt_cells_size,
#endif
    char* target_flat_str,
    int* flat_ids,
    char* query,
    int q_len,
    int_type* best_scores
) {

    // assumptions:
    // - the 3rd parameter of kernel call is provided (for dynamic allocation of shared memory)
    // - the block size (or number of threads) equals mx=max(Lq, max{Lt})
    // - the target is spreaded on the row (axis=0) and the query on columns (axis=1)
    // - thread with "tid" works on diagonal_idx of tid and maybe mx + tid
    // - starting cell of diagonal_idx "did" is (row=max(1, did - Lq + 2), col=max(1, Lq - did)) [reminder: row-> [1, Lt], col-> [1, Lq], diag_idx-> [0, Lt+Lq-1)]
    __shared__ int best_overall_score; best_overall_score = 0;
    // some preprocessing of variables (these would go on registers probably)
    int bid = blockIdx.x; int tid =  threadIdx.x; int bsize = blockDim.x; int tnum = bsize;
    int t_start_idx = flat_ids[bid];
    int t_len = flat_ids[bid + 1] - t_start_idx;
    int diagonal_size = t_len + q_len - 1;
    int actual_diag_size = max(t_len, q_len);
    int actual_tnum = min(actual_diag_size, tnum);

    // dynamically divide the whole shared memory for the shared memory variables
#ifdef REDUCE_ALIGNMENT_RESULT
    opt_cell* best_cells = (opt_cell*)shared_memory;
    char* q_cache = (char *)&best_cells[opt_cells_size];
    char* t_cache = (char *)&q_cache[q_len];
#else
    char* q_cache = (char *)shared_memory;
    char* t_cache = (char *)&q_cache[q_len];
#endif
    byte_type* score_matrix_flat = (byte_type *)&t_cache[t_len];

    // initialize and retrieve content of shared memory variables
    for (int i = 0; i < (score_matrix_len + tnum - 1) / tnum; i++) {
        int idx = i * tnum + tid;
        if (idx < score_matrix_len) {
            score_matrix_flat[idx] = _score_matrix[idx];
        }
    }

    for (int i = 0; i < (q_len + tnum - 1) / tnum; i++) {
        int idx = i * tnum + tid;
        if (idx < q_len) {
            q_cache[idx] = query[idx];
        }
    }

    for (int i = 0; i < (t_len + tnum - 1) / tnum; i++) {
        int idx = i * tnum + tid;
        if (idx < t_len) {
            t_cache[idx] = target_flat_str[idx + t_start_idx];
        }
    }

    __syncthreads();

    // now actual alignment algorithm can begin :D
    int_type best_score = 0;
    int best_diag_idx = tid;
    int_type current_score;
    int diag_idx = tid;

    if (tnum >= actual_diag_size) {
        current_score = 0;
        if (diag_idx < actual_diag_size) {
            for (int row = max(1, diag_idx - q_len + 2), col = max(1, q_len - diag_idx); row <= t_len && col <= q_len; row++, col++) {
                int mat_idx = q_cache[col-1] * ALPH_SIZE + t_cache[row-1];
                byte_type substitution_score = score_matrix_flat[mat_idx];
                current_score = max(0, current_score + substitution_score);
                if (current_score > best_score) {
                    best_score = current_score;
                }
            }
        if (diag_idx + actual_diag_size < diagonal_size) {
            diag_idx = diag_idx + actual_diag_size;
            current_score = 0;
            for (int row = max(1, diag_idx - q_len + 2), col = max(1, q_len - diag_idx); row <= t_len && col <= q_len; row++, col++) {
                int mat_idx = q_cache[col-1] * ALPH_SIZE + t_cache[row-1];
                byte_type substitution_score = score_matrix_flat[mat_idx];
                current_score = max(0, current_score + substitution_score);
                if (current_score > best_score) {
                    best_score = current_score;
                    best_diag_idx = diag_idx;
                }
            }
        }
        }
    } else {
        for (int i = 0; i < (actual_diag_size + tnum - 1) / tnum; i++) {
            diag_idx = i * tnum + tid;
            current_score = 0;
            if (diag_idx < actual_diag_size) {
                for (int row = max(1, diag_idx - q_len + 2), col = max(1, q_len - diag_idx); row <= t_len && col <= q_len; row++, col++) {
                    int mat_idx = q_cache[col-1] * ALPH_SIZE + t_cache[row-1];
                    byte_type substitution_score = score_matrix_flat[mat_idx];
                    current_score = max(0, current_score + substitution_score);
                    if (current_score > best_score) {
                        best_score = current_score;
                        best_diag_idx = diag_idx;
                    }
                }
            if (diag_idx + actual_diag_size < diagonal_size) {
                diag_idx = diag_idx + actual_diag_size;
                current_score = 0;
                for (int row = max(1, diag_idx - q_len + 2), col = max(1, q_len - diag_idx); row <= t_len && col <= q_len; row++, col++) {
                    int mat_idx = q_cache[col-1] * ALPH_SIZE + t_cache[row-1];
                    byte_type substitution_score = score_matrix_flat[mat_idx];
                    current_score = max(0, current_score + substitution_score);
                    if (current_score > best_score) {
                        best_score = current_score;
                        best_diag_idx = diag_idx;
                    }
                }
            }
            }
        }
    }

#ifndef REDUCE_ALIGNMENT_RESULT
    atomicMax(&best_overall_score, best_score);
    __syncthreads();
#else
    // here we first reduce results on each warp seperately with using warp_shuffle operations
    // this can divide the length of best_cells by 32 and save shared_memory
    if (tid < actual_tnum) {
        initial_warp_reduce(best_cells, tid, best_score, best_diag_idx);
    }
    __syncthreads();
    #if defined DEBUG_REDUCE
    if (!tid) {
        for (int i = 0; i < (actual_tnum + WARP_SIZE - 1) / WARP_SIZE; i++) {
            printf("i:%d, score:%d, idx:%d\n", i, best_cells[i].score, best_cells[i].diagonal_idx);
        }
    }
    #endif
    // what we're doing here is that we divide the last 5 iterations of this loop to avoid stalling the majority of the threads that are idle
    // instead only the first warp will resume reducing seperately
    int array_size = (actual_tnum + WARP_SIZE - 1) / WARP_SIZE;
    int cur_step = (array_size + 1) / 2;
    initial_all_reduce(cg::this_thread_block(), best_cells, tid, cur_step, array_size);
    if (tid < WARP_SIZE) {
        final_warp_reduce(best_cells, tid, cur_step, array_size);
    }
    // now best_cells[0] holds the maximum best score
    if (!tid) best_overall_score = best_cells[0].score;

#endif

    if (!tid) {
#if defined REDUCE_ALIGNMENT_RESULT && defined DEBUG_REDUCE
        printf("%d, bid:%d, best_score:%d, best_diag: %d\n", t_len, bid, best_overall_score, best_cells[0].diagonal_idx);
#endif
        best_scores[bid] = best_overall_score;
    }
}

ull total_global_memory(int dev = 0) {
    int devCount = 0;
    gpuErrchk(cudaGetDeviceCount(&devCount));
    if (dev < devCount) {
        cudaSetDevice(dev);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        return deviceProp.totalGlobalMem;
    } else {
        return 0;
    }
}

void call_kernel(const std::string query, const vector<std::string>& targets, bool on_columns=true, int_type** scores_to_return_addr=NULL) {

/* some todos for more efficiency:
    - merge minor memory transactions
    - use pinned memory
    - clean up more if possible!
*/
    int max_target_len = 0;
    int sum_target_len = 0;
    for (auto &t: targets) {
        int temp_len = t.size();
        max_target_len = max(max_target_len, temp_len);
        sum_target_len += temp_len;
    }
    char * q_str = (char *) query.c_str();
    int q_len = query.size();
    // convert query to its indexes on aa2num
    char * q_str_idx = (char *)malloc(q_len * sizeof(char));
    for (int i = 0; i < q_len; i++) {
        q_str_idx[i] = (char)aa2num[q_str[i] - 'A'];
    }
    int num_target_strs = targets.size();

    // allocate and initialize some necessary variables on device for kernel call
    char * d_query_str_idx; char * d_targets_fstr; int * d_target_indices; int_type *best_scores;
    cudaMalloc(&d_query_str_idx, q_len * sizeof(char));
    cudaMemcpy(d_query_str_idx, q_str_idx, q_len * sizeof(char), cudaMemcpyHostToDevice);
    allcoate_strings_on_device_flattened(targets, &d_targets_fstr, &d_target_indices, num_target_strs);
    cudaMallocManaged(&best_scores, num_target_strs * sizeof(int_type));
    int global_memory_size = q_len + sizeof(char)
    + sum_target_len * sizeof(char)
    + (num_target_strs + 1) * sizeof(int)
    + num_target_strs * sizeof(int_type)
    ;
    // todo: do sth if memory_size exceeds global_memory and improve this next log!
    printf("total_global_memory: %llu\n", total_global_memory(0));
    // be sure to have called init_score_matrix() before using score matrix
    // todo: do sth about this init_score_matrix necessity
    cudaMemcpyToSymbol(_score_matrix, score_matrix_flattened, SCORE_MATRIX_SIZE * sizeof(byte_type));

    // calculating the (maximum) size of different variables which will go on shared-memory
    // also a future todo: remove this score_matrix_len from argument list of kernel functions
    // int score_matrix_len = SCORE_MATRIX_SIZE;
    int rows_memory_len = 2 * (q_len + 1); // current row + last row
    int q_tmax_len = (max_target_len + q_len);
    int opt_cells_size;
    // calculating the maximum needed shared-memory size in bytes
    int shared_memory_size;
    int block_size;
    if (on_columns) {
        block_size = min(q_len, MAX_LEN);
        opt_cells_size = (block_size + WARP_SIZE - 1) / WARP_SIZE;
        shared_memory_size = SCORE_MATRIX_SIZE  * sizeof(byte_type)
        + rows_memory_len * sizeof(int_type)
        + q_tmax_len  * sizeof(char)
        + opt_cells_size * sizeof(opt_cell)
        ;

        local_ungapped_alignment<<<num_target_strs, block_size, shared_memory_size>>> (
            SCORE_MATRIX_SIZE,
            rows_memory_len,
            q_tmax_len,
            opt_cells_size,
            d_targets_fstr,
            d_target_indices,
            d_query_str_idx,
            q_len,
            best_scores
        );
    } else {
        block_size = min(max(max_target_len, q_len), MAX_LEN);
        opt_cells_size = (block_size + WARP_SIZE - 1) / WARP_SIZE;

        shared_memory_size = SCORE_MATRIX_SIZE  * sizeof(byte_type)
        + q_tmax_len  * sizeof(char)
#ifdef REDUCE_ALIGNMENT_RESULT
        + opt_cells_size * sizeof(opt_cell)
#endif
        ;
        local_ungapped_alignment_on_diagonal<<<num_target_strs, block_size, shared_memory_size>>> (
            SCORE_MATRIX_SIZE,
            q_tmax_len,
#ifdef REDUCE_ALIGNMENT_RESULT
            opt_cells_size,
#endif
            d_targets_fstr,
            d_target_indices,
            d_query_str_idx,
            q_len,
            best_scores
        );
    }
#if defined SHOW_KERNEL_CONF // && !defined BENCHMARK
    printf("global_memory_size: %d, shared_memory_size: %d, num_targets=grid_size: %d, block_size: %d, q_len: %d, max_t_len: %d, mode: %s\n", global_memory_size, shared_memory_size, num_target_strs, block_size, q_len, max_target_len, on_columns?"on_columns":"on_diagonals");
#endif
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#if defined BENCHMARK
    *scores_to_return_addr = best_scores;
    #if defined DEBUG_BENCH
    for (int i = 0; i < num_target_strs; i++) {
        cout << query << " " << targets[i] << " " << best_scores[i] << endl;
    }
    #endif
#elif defined SHOW_ALIGNMENT_SCORES
    cout << "final scores:" << endl;
    for (int i = 0; i < num_target_strs; i++) {
        cout << query.substr(0, 10) << "," << targets[i].substr(0, 10);
        printf("(%4d,%4d): %4d\n", q_len, (int)targets[i].size(), best_scores[i]);
    }
#endif
    // freeing device memory
    free_strings_on_device_flattened(d_targets_fstr, d_target_indices);
    cudaFree(d_query_str_idx);
#if !defined BENCHMARK
    cudaFree(best_scores);
#endif
}

void measure_kernel_time(const std::string query, const vector<std::string>& targets, bool on_columns=true, bool verbose=true, int number_of_calls=20) {
    clock_t start_clock, end_clock;
    long maximum_clocks = 0, minimum_clocks = LLONG_MAX, sum_clocks = 0;

    for (int i = 0; i < number_of_calls; i++) {
        start_clock = clock();

        call_kernel(query, targets, on_columns);

        end_clock = clock();

        long execution_clocks = end_clock - start_clock;
        maximum_clocks = max(maximum_clocks, execution_clocks), minimum_clocks = min(minimum_clocks, execution_clocks);
        sum_clocks += execution_clocks;
        if (verbose) cout << "execution clocks: " << execution_clocks << endl; // divide by CLOCKS_PER_SEC if actual time is needed
    }
    debug(maximum_clocks); debug(minimum_clocks); cout << "avg_clocks: " << sum_clocks / number_of_calls << endl;
}

bool sort_by_score(const std::tuple<std::string, std::string, int>& a, const std::tuple<std::string, std::string, int>& b) {
    return (get<2>(a) > get<2>(b));
}

int main(int argc, char** argv) {
    vector<string> queries, targets;
    init_score_matrix();
    // arguments are optional and the default ones are targets=TestSamples/targets.txt, is_fasta=true, on_columns=true, query=TestSamples/queries.txt[0]
    // the 1s argument tells the file path, the 2nd one tells if target file is fasta, the 3rd determines alignment method (on-columns or on-diagonals) and the 4th tell query path
    // only string "0" can change 2nd and 3rd arguments to "not fasta" and "not on_column" (i.e "on_diagonal")
    std::string target_path = "TestSamples/targets.txt";
    std::string query_path = "TestSamples/queries.txt";
    bool is_target_fasta = false;
    bool is_query_fasta = false;
    bool on_columns = true;
    if (argc > 1) {
        target_path = argv[1];
        if (argc > 2) {
            if(std::string(argv[2]) == "1") is_target_fasta = true;
            if (argc > 3) {
                if (std::string(argv[3]) == "0") on_columns = false;
                if (argc > 4) {
                    query_path = argv[4];
                    if (argc > 5) {
                        if(std::string(argv[2]) == "1") is_query_fasta = true;
                    }
                }
            }
        }
    }
#if defined BENCHMARK
    vector<string> query_ids, target_ids;
    // target_path = "TestSamples/targets.fasta";
    // query_path = "TestSamples/queries.fasta";
    init_input_from_fasta_file_with_id(query_path, queries, query_ids);
    init_input_from_fasta_file_with_id(target_path, targets, target_ids);
    printf("number of queries: %d, number of targets: %d\n", (int)queries.size(), (int)targets.size());
#else
    init_input_from_file(query_path, queries, is_query_fasta);
    init_input_from_file(target_path, targets, is_target_fasta);
#endif

#if defined BENCHMARK
    int_type* scores_ret;
    int num_target_strs = targets.size();
    int num_query_strs = queries.size();
    clock_t start_clock, end_clock;
    long execution_clocks;
    vector<std::tuple<std::string, std::string, int>> results;
    for (int i = 0; i < num_query_strs; i++) {
        start_clock = clock();
        call_kernel(queries[i], targets, on_columns, &scores_ret);
        end_clock = clock();
        execution_clocks = end_clock - start_clock;
    #if defined DEBUG && ( defined SORT_RESULTS || defined SORT_RESULTS_LIMITED )
        cout << "kernel execution clocks: " << execution_clocks << endl;
        start_clock = clock();
        #if defined SORT_RESULTS
        for (int j = 0; j < num_target_strs; j++) {
            results.push_back(std::make_tuple(query_ids[i], target_ids[j], scores_ret[j]));
        }
        #else
        vector<std::tuple<std::string, std::string, int>> results_tmp;
        for (int j = 0; j < num_target_strs; j++) {
            results_tmp.push_back(std::make_tuple(query_ids[i], target_ids[j], scores_ret[j]));
        }
        std::sort(results_tmp.begin(), results_tmp.end(), sort_by_score);
        for (int j = 0; j < min(num_target_strs, BENCH_CAP_VAL); j++) {
            results.push_back(results_tmp[j]);
        }
        #endif
        end_clock = clock();
        execution_clocks = end_clock - start_clock;
        cout << "vector contruction execution clocks: " << execution_clocks << endl;
    #endif
    }
    #if defined DEBUG && ( defined SORT_RESULTS || defined SORT_RESULTS_LIMITED )
    start_clock = clock();
    std::sort(results.begin(), results.end(), sort_by_score);
    end_clock = clock();
    execution_clocks = end_clock - start_clock;
    cout << "sort execution clocks: " << execution_clocks << endl;
    int results_size = results.size(); // in SHOW_RESULT MODE it equals num_target_strs * num_query_strs
    for (int i = 0; i < min(results_size, BENCH_CAP_VAL); i++) {
        std::tuple<std::string, std::string, int> t = results[i];
        cout << get<0>(t) << " " << get<1>(t) << " " << get<2>(t) << endl;
    }
    // todo: this is hardcode! (chrono and ctime are needed and are included in bits/stc++)
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    std::ostringstream oss_time;
    oss_time << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S_");
    string out_name = "PlayGround_/full_output_" + oss_time.str() + std::to_string(num_query_strs) + "_" + std::to_string(num_target_strs);
    debug(out_name);
    ofstream out_file(out_name);
    for (int i = 0; i < results_size; i++) {
        std::tuple<std::string, std::string, int> tuple = results[i];
        std::string sep = "|";
        out_file << split(get<0>(tuple), sep)[2] << " " << split(get<1>(tuple), sep)[2] << " " << get<2>(tuple) << endl;
    }
    #endif
    cudaFree(scores_ret);
#elif defined DEBUG
    call_kernel(queries[0], targets, on_columns);
#else
    measure_kernel_time(queries[0], targets, on_columns, true, 10);
#endif
    
    return 0;
}