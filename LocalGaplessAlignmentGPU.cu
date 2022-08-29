// remember!
#include <bits/stdc++.h>
#include "Utils.h"
#include "ScoreMatrix.h"

using namespace std;

#define MAX_LEN 1024
#define debug(A) cout << #A << ": " << A << endl

// #define DEBUG_REDUCE
// #define DEBUG_KERNEL
// #define DEBUG
#define SHOW_KERNEL_CONF
#ifdef DEBUG
    #define SHOW_ALIGNMENT_SCORES
#endif

// #define REDUCE_ON_COLUMNS
#define REDUCE_ALIGNMENT_RESULT
// #define USE_LOCK

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

__constant__ byte_type _aa2num['Z' - 'A'];
__constant__ byte_type _score_matrix[ALPH_SIZE * ALPH_SIZE];

__device__ void lock(int* mutex) {
    while (atomicCAS(mutex, 0, 1) != 0) {}
}

__device__ void unlock(int* mutex) {
    atomicExch(mutex, 0);
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

__global__ void local_ungapped_alignment(
    int aa2num_len,
    int score_matrix_len,
    int rows_memory_len,
    int q_tmax_len,
    int opt_cells_size,
    char* target_flat_str,
    int* flat_ids,
    char* query,
    // char* query_idx,
    int q_len,
    int_type* best_scores
) {

    // assumptions:
    // - the 3rd parameter of kernel call is provided (for dynamic allocation of shared memory)
    // - the block size (or number of threads) equals query size ([tnum = q_len = bsize] in code)
    // - the target is spreaded on the row (axis=0) and the query on columns (axis=1)

    __shared__ int best_overall_score; best_overall_score = 0;
#if !defined REDUCE_ALIGNMENT_RESULT && defined USE_LOCK
    __shared__ int mutex; mutex = 0;
#endif
    // some preprocessing of variables (these would go on registers probably)
    int bid = blockIdx.x; int tid =  threadIdx.x; int bsize = blockDim.x; int tnum = bsize;
    int t_start_idx = flat_ids[bid];
    int t_len = flat_ids[bid + 1] - t_start_idx;

    // dynamically divide the whole shared memory for the shared memory variables
    byte_type* aa2num = (byte_type *)shared_memory;
    byte_type* score_matrix_flat = (byte_type *)&aa2num[aa2num_len];
    int_type* last_row = (int_type *)&score_matrix_flat[score_matrix_len];
    int_type* current_row = (int_type *)&last_row[q_len + 1];
    /* Note1: the order of these allocations actually matters and if not considered, the "misaligned address" error could occur
        for example cuda needs that int* pointers, point to addresses aligned with 32 bytes and if we allocate the char* before int* it's probable this error occurs
       Note2: the size of variables being used here is their actual size not the max size we had to specify for the kernel call
       todo: would be nice if we could free or release the additional space previously reserved but not used here */
    opt_cell* best_cells = (opt_cell*)&current_row[q_len + 1];
    char* q_cache = (char *)&best_cells[opt_cells_size];
    char* t_cache = (char *)&q_cache[q_len];
    
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

    // Note: thread with index i works on column i+1 of alignment matrix and need column i to compute it
    last_row[tid] = 0;
    // current row does not need initialization!
    
    q_cache[tid] = query[tid];
    // q_cache[tid] = query_idx[tid];

    for (int i = 0; i < (t_len + tnum - 1) / tnum; i++) {
        int idx = i * tnum + tid;
        if (idx < t_len) {
            t_cache[idx] = target_flat_str[idx + t_start_idx];
        }
    }

#ifdef REDUCE_ON_COLUMNS
    best_cells[tid].score = 0;
    best_cells[tid].diagonal_idx = USHRT_MAX;
#else
    int diagonal_size = t_len + q_len - 1;
    for (int i = 0; i < (diagonal_size + tnum - 1) / tnum; i++) {
        int idx = i * tnum + tid;
        if (idx < diagonal_size) {
            best_cells[idx].score = 0;
            best_cells[idx].diagonal_idx = idx;
        }
    }
#endif

    __syncthreads();

    // now actual alignment algorithm can begin :D
    // todo: test the thrust and cuBLAS library and the reduce method for acquiring extermum in an array
    for (int row = 1; row <= t_len; row++) {
        // this thread works on column = tid + 1 and q[tid] char
        // int char_idx = (int) q_cache[tid];
        // int mat_idx = char_idx * ALPH_SIZE + aa2num[int(t_cache[row-1] - 'A')];
        int mat_idx = aa2num[(q_cache[tid] - 'A')] * ALPH_SIZE + aa2num[(t_cache[row-1] - 'A')];
        byte_type substitution_score = score_matrix_flat[mat_idx];
#ifdef DEBUG_KERNEL
        printf("(r%d%c, c%2d%c)%2d\n", row, t_cache[row-1], tid+1, q_cache[tid], substitution_score);
#endif
        int_type current_score = max(0, last_row[tid] + substitution_score);
        int_type current_diagonal = row - tid + q_len - 2; // row - col + q_len - 1
        current_row[tid+1] = current_score; // there is no race condition here!

        // each thread works on different diagonal or column (considering columns of threads differ and rows are the same)
        // so there is no race condition here yet either
#ifdef REDUCE_ON_COLUMNS
        if (current_score > best_cells[tid].score) {
            best_cells[tid].score = current_score;
            best_cells[tid].diagonal_idx = current_diagonal;
#else
        if (current_score > best_cells[current_diagonal].score) {
            best_cells[current_diagonal].score = current_score;
#endif
            // here we have a race condition to update the best_overall_score and best_diagonal
#if !defined REDUCE_ALIGNMENT_RESULT && defined USE_LOCK
            while (atomicCAS(&mutex, 0, 1) != 0); // lock
            if (current_score > best_overall_score) {
                best_overall_score = current_score;
                // best_diag_idx = current_diagonal;
            }
            atomicExch(&mutex, 0); // unlock
#elif !defined REDUCE_ALIGNMENT_RESULT && !defined USE_LOCK
            atomicMax(&best_overall_score, current_score);
#endif
        }
        // to reassure that all current_row cells are updated before using them
        __syncthreads();
#ifdef DEBUG_KERNEL
        printf("diag%2d(r%d, c%2d)%2d\n", current_diagonal, row, tid+1, current_score);
#endif
        // last_row[0] is already zero and there is no need to update the value (btw same can be said for last_row[q_len] considering it will not be used but anyway!)
        last_row[tid+1] = current_row[tid+1];
        __syncthreads();
    }

#ifdef REDUCE_ALIGNMENT_RESULT
    #if defined DEBUG_REDUCE & !defined REDUCE_ON_COLUMNS
    if (!tid) {
        printf("before reduction:\n");
        for (int i = 0; i < diagonal_size; i++) {
            printf("score:%d, idx:%d\n", best_cells[i].score, best_cells[i].diagonal_idx);
        }
    }
    #endif

    #if !defined REDUCE_ON_COLUMNS
    // this whole reduction on diagonals complexity time is not o(logn) but it's o(Lt/Lq) + o(logn)
    for (int i = 0; i < (diagonal_size + tnum - 1) / tnum; i++) {
        int idx = i * tnum + tid;
        if (idx < diagonal_size && best_cells[tid].score < best_cells[idx].score) {
            best_cells[tid].score = best_cells[idx].score;
            best_cells[tid].diagonal_idx = best_cells[idx].diagonal_idx;
        }
    }
    #endif

    for (int i = (tnum + 1) / 2; i > 0; i = (i+1) / 2) {
        int idx = tid + i;
        if (tid < i && idx < tnum) {
    #ifdef DEBUG_REDUCE
            printf("(tid:%d, idx:%d| cur_size:%d block_size:%d)\n", tid, idx, i, tnum);
    #endif
            if (best_cells[tid].score < best_cells[idx].score) {
                best_cells[tid].score = best_cells[idx].score;
                best_cells[tid].diagonal_idx = best_cells[idx].diagonal_idx;
            }
        }
        __syncthreads();
    #ifdef DEBUG_REDUCE
        if (!tid) for (int j = 0; j < i; j++) {
            printf("score:%d, idx:%d\n", best_cells[j].score, best_cells[j].diagonal_idx);
        }
    #endif
        if (i == 1)
            break;
    }
    // now best_cells[0] holds the maximum best score
    best_overall_score = best_cells[0].score;
    #ifdef DEBUG_REDUCE
        printf("end of reducing from thread %d --> best_diagonal:%d, best_socore:%d\n", tid, best_cells[0].score, best_cells[0].diagonal_idx);
    #endif
#else
    #ifdef DEBUG_KERNEL
    printf("end of thread %d, best_overall_score: %d\n", tid, best_overall_score);
    #endif
#endif

    if (!tid) {
#if defined REDUCE_ALIGNMENT_RESULT && defined DEBUG_REDUCE
        printf("%d, bid:%d, best_overall_score:%d, best_diag: %d\n", t_len, bid, best_overall_score, best_cells[0].diagonal_idx);
#endif
        best_scores[bid] = best_overall_score;
    }
}

__global__ void local_ungapped_alignment_on_diagonal(
    int aa2num_len,
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
    int actual_tnum = max(t_len, q_len);

    // dynamically divide the whole shared memory for the shared memory variables
    byte_type* aa2num = (byte_type *)shared_memory;
    byte_type* score_matrix_flat = (byte_type *)&aa2num[aa2num_len];
#ifdef REDUCE_ALIGNMENT_RESULT
    opt_cell* best_cells = (opt_cell*)&score_matrix_flat[score_matrix_len];
    char* q_cache = (char *)&best_cells[opt_cells_size];
    char* t_cache = (char *)&q_cache[q_len];
#else
    char* q_cache = (char *)&score_matrix_flat[score_matrix_len];
    char* t_cache = (char *)&q_cache[q_len];
#endif

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

    q_cache[min(tid, q_len-1)] = query[min(tid, q_len-1)];
    t_cache[min(tid, t_len-1)] = target_flat_str[min(tid, t_len-1) + t_start_idx];

    __syncthreads();

    // now actual alignment algorithm can begin :D
    int diag_idx = tid;
    bool best_diag_loc_first = true;
    int_type best_score = 0;
    int_type current_score = 0;
    if (tid < actual_tnum) {
        for (int row = max(1, diag_idx - q_len + 2), col = max(1, q_len - diag_idx); row <= t_len && col <= q_len; row++, col++) {
            int mat_idx = aa2num[(q_cache[col-1] - 'A')] * ALPH_SIZE + aa2num[(t_cache[row-1] - 'A')];
            byte_type substitution_score = score_matrix_flat[mat_idx];
            current_score = max(0, current_score + substitution_score);
            if (current_score > best_score) {
                best_score = current_score;
            }
        }
    }
    if (tid + actual_tnum < diagonal_size) {
        diag_idx = tid + actual_tnum;
        current_score = 0;
        for (int row = max(1, diag_idx - q_len + 2), col = max(1, q_len - diag_idx); row <= t_len && col <= q_len; row++, col++) {
            int mat_idx = aa2num[(q_cache[col-1] - 'A')] * ALPH_SIZE + aa2num[(t_cache[row-1] - 'A')];
            byte_type substitution_score = score_matrix_flat[mat_idx];
            current_score = max(0, current_score + substitution_score);
            if (current_score > best_score) {
                best_score = current_score;
                best_diag_loc_first = false;
            }
        }
    }

#ifndef REDUCE_ALIGNMENT_RESULT
    atomicMax(&best_overall_score, best_score);
    __syncthreads();
#else
    if (tid < actual_tnum) {
        best_cells[tid].score = best_score;
        best_cells[tid].diagonal_idx = best_diag_loc_first * tid + !best_diag_loc_first * (tid + actual_tnum);
    }
    __syncthreads();
    #if defined DEBUG_REDUCE
    if (!tid) {
        for (int i = 0; i < actual_tnum; i++) {
            printf("i:%d, score:%d, idx:%d\n", i, best_cells[i].score, best_cells[i].diagonal_idx);
        }
    }
    #endif

    for (int i = (actual_tnum + 1) / 2; i > 0; i = (i+1) / 2) {
        int idx = tid + i;
        if (tid < i && idx < actual_tnum) {
    #ifdef DEBUG_REDUCE
            printf("(tid:%d, idx:%d| cur_size:%d block_size:%d)\n", tid, idx, i, tnum);
    #endif
            if (best_cells[tid].score < best_cells[idx].score) {
                best_cells[tid].score = best_cells[idx].score;
                best_cells[tid].diagonal_idx = best_cells[idx].diagonal_idx;
            }
        }
        __syncthreads();
    #ifdef DEBUG_REDUCE
        if (!tid) for (int j = 0; j < i; j++) {
            printf("score:%d, idx:%d\n", best_cells[j].score, best_cells[j].diagonal_idx);
        }
    #endif
        if (i == 1)
            break;
    }

    // now best_cells[0] holds the maximum best score
    best_overall_score = best_cells[0].score;

#endif

    if (!tid) {
#if defined REDUCE_ALIGNMENT_RESULT && defined DEBUG_REDUCE
        printf("%d, bid:%d, best_score:%d, best_diag: %d\n", t_len, bid, best_score, best_cells[0].diagonal_idx);
#endif
        best_scores[bid] = best_overall_score;
    }
}

void call_kernel(string query, vector<string>& targets, bool on_columns=true) {

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
    // char * q_str_idx = (char *)malloc(q_len * sizeof(char));
    // for (int i = 0; i < q_len; i++) {
    //     q_str_idx[i] = (char)aa2num[q_str[i] - 'A'];
    // }
    int num_target_strs = targets.size();

    // allocate and initialize some necessary variables on device for kernel call
    char * d_query_str; char * d_targets_fstr; int * d_target_indices; int_type *best_scores;
    cudaMalloc(&d_query_str, q_len * sizeof(char));
    cudaMemcpy(d_query_str, q_str, q_len * sizeof(char), cudaMemcpyHostToDevice);

    // char * d_query_str_idx; char * d_targets_fstr; int * d_target_indices; int *best_scores;
    // cudaMalloc(&d_query_str_idx, q_len * sizeof(char));
    // cudaMemcpy(d_query_str_idx, q_str_idx, q_len * sizeof(char), cudaMemcpyHostToDevice);
    allcoate_strings_on_device_flattened(targets, &d_targets_fstr, &d_target_indices, num_target_strs);
    cudaMallocManaged(&best_scores, num_target_strs * sizeof(int_type));
    // be sure to have called init_score_matrix() before using score matrix
    // todo: do sth about this init_score_matrix necessity
    cudaMemcpyToSymbol(_score_matrix, score_matrix_flattened, SCORE_MATRIX_SIZE * sizeof(byte_type));
    cudaMemcpyToSymbol(_aa2num, aa2num, ('Z' - 'A') * sizeof(byte_type));

    // calculating the (maximum) size of different variables which will go on shared-memory
    int aa2num_len = 'Z' - 'A';
    int score_matrix_len = SCORE_MATRIX_SIZE;
    int rows_memory_len = 2 * (q_len + 1); // current row + last row
    int q_tmax_len = (max_target_len + q_len);
#ifdef REDUCE_ON_COLUMNS
    int opt_cells_size = q_len; // column_size
#else
    int max_diagonal_size = (max_target_len + q_len - 1);
    int opt_cells_size = max_diagonal_size;
#endif
    // calculating the maximum needed shared-memory size in bytes
    int shared_memory_size;
    if (on_columns) {
        shared_memory_size = aa2num_len * sizeof(byte_type)
        + score_matrix_len  * sizeof(byte_type)
        + rows_memory_len * sizeof(int_type)
        + q_tmax_len  * sizeof(char)
        + opt_cells_size * sizeof(opt_cell)
        ;

        local_ungapped_alignment<<<num_target_strs, q_len, shared_memory_size>>> (
            aa2num_len,
            score_matrix_len,
            rows_memory_len,
            q_tmax_len,
            opt_cells_size,
            d_targets_fstr,
            d_target_indices,
            d_query_str,
            // d_query_str_idx,
            q_len,
            best_scores
        );
    } else {
        opt_cells_size = max(max_target_len, q_len);

        shared_memory_size = aa2num_len * sizeof(byte_type)
        + score_matrix_len  * sizeof(byte_type)
        + q_tmax_len  * sizeof(char)
#ifdef REDUCE_ALIGNMENT_RESULT
        + opt_cells_size * sizeof(opt_cell)
#endif
        ;
        local_ungapped_alignment_on_diagonal<<<num_target_strs, opt_cells_size, shared_memory_size>>> (
            aa2num_len,
            score_matrix_len,
            q_tmax_len,
#ifdef REDUCE_ALIGNMENT_RESULT
            opt_cells_size,
#endif
            d_targets_fstr,
            d_target_indices,
            d_query_str,
            q_len,
            best_scores
        );
    }
#ifdef SHOW_KERNEL_CONF
    printf("shared_memory_size: %d, num_targets=grid_size: %d, q_len: %d, max_t_len: %d, mode: %s\n", shared_memory_size, num_target_strs, q_len, max_target_len, on_columns?"on_columns":"on_diagonals");
#endif
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#ifdef SHOW_ALIGNMENT_SCORES
    cout << "final scores:" << endl;
    for (int i = 0; i < num_target_strs; i++) {
        cout << query.substr(0, 10) << "," << targets[i].substr(0, 10);
        printf("(%4d,%4d): %4d\n", q_len, (int)targets[i].size(), best_scores[i]);
    }
#endif
    // freeing device memory
    free_strings_on_device_flattened(d_targets_fstr, d_target_indices);
    cudaFree(d_query_str);
    // cudaFree(d_query_str_idx);
    cudaFree(best_scores);
}

void measure_kernel_time(string query, vector<string> targets, bool on_columns=true, bool verbose=true, int number_of_calls=20) {
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

int main(int argc, char** argv) {
    vector<string> queries, targets;
    init_score_matrix();
    // arguments are optional and the default ones are targets=TestSamples/targets.txt, is_fasta=true, on_columns=true, query=TestSamples/queries.txt[0]
    // the 1s argument tells the file path, the 2nd one tells if target file is fasta, the 3rd determines alignment method (on-columns or on-diagonals) and the 4th tell query path
    // only string "0" can change 2nd and 3rd arguments to "not fasta" and "not on_column" (i.e "on_diagonal")
    string target_path = "TestSamples/targets.txt";
    string query_path = "TestSamples/queries.txt";
    bool is_target_fasta = true;
    bool on_columns = true;
    if (argc > 1) {
        target_path = argv[1];
        if (argc > 2) {
            if(string(argv[2]) == "0") is_target_fasta = false;
            if (argc > 3) {
                if (string(argv[3]) == "0") on_columns = false;
                if (argc > 4) {
                    query_path = argv[4];
                }
            }
        }
    }
    init_input_from_file(query_path, queries, false);
    init_input_from_file(target_path, targets, is_target_fasta);

#ifdef DEBUG
    call_kernel(queries[0], targets, on_columns);
#else
    measure_kernel_time(queries[0], targets, on_columns, true, 10);
#endif
    
    return 0;
}