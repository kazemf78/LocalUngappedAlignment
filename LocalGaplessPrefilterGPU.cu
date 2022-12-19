// remember!
#include <bits/stdc++.h>
#include <cooperative_groups.h>
#include "Utils.h"
#include "ScoreMatrix.h"

namespace cg = cooperative_groups;

#define MAX_LEN 1024
#define FULL_MASK 0xffffffff
#define WARP_SIZE 32
#define debug(A) std::cout << #A << ": " << A << std::endl
#define ull unsigned long long
#define BENCH_CAP_VAL 2000
#define PREFILTER_SCORE_THRESHOLD 60

// #define DEBUG_REDUCE
// #define DEBUG
#define SHOW_KERNEL_CONF
// #define SHOW_ALL_ALIGNMENT_SCORES
// #define REDUCE_ALIGNMENT_RESULT

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

bool sort_by_score(const std::tuple<int, int, int>& a, const std::tuple<int, int, int>& b) {
    return (std::get<2>(a) > std::get<2>(b));
}


void allcoate_strings_on_device_flattened(const std::vector<std::string>& strings, char ** d_fstr_addr, int ** d_fids_addr, int num_strs) {
    std::string flat_temp; int* flat_ids; char * flat_str;
    flat_ids = (int *) malloc((num_strs + 1) * sizeof(int));
    int cur_ptr = 0;
    // todo: can implementation of flattening be more efficient maybe?
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
        // todo: handle undefined value or force the threads to be multiple of 32
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


void call_kernel_multiple_queries(const std::vector<std::string>& queries, const std::vector<std::string>& targets, const std::vector<std::string>& query_ids, const std::vector<std::string>& target_ids, bool on_columns=true) {

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
    int num_target_strs = targets.size();
    int num_query_strs = queries.size();

    // allocate and initialize some necessary variables on device for kernel call
    clock_t start_clock, end_clock, s1, s2, s3, s4, ss, ee;
    long execution_clocks;
    start_clock = clock();
    ss = clock();
    char * d_targets_fstr; int * d_target_indices; int_type *best_scores;
    allcoate_strings_on_device_flattened(targets, &d_targets_fstr, &d_target_indices, num_target_strs);
    cudaMallocManaged(&best_scores, num_target_strs * sizeof(int_type));
    end_clock = clock();
    execution_clocks = end_clock - start_clock;
    #if defined DEBUG
        std::cout << "global memory transaction clocks: " << execution_clocks << std::endl;
    #endif
    // int global_memory_size = q_len * sizeof(char)
    int global_memory_size = 0
    + sum_target_len * sizeof(char)
    + (num_target_strs + 1) * sizeof(int)
    + num_target_strs * sizeof(int_type)
    ;
    // todo: do sth if memory_size exceeds global_memory and improve this next log!
    printf("total_global_memory: %llu\n", total_global_memory(0));
    // be sure to have called init_score_matrix() before using score matrix
    // todo: do sth about this init_score_matrix necessity
    cudaMemcpyToSymbol(_score_matrix, score_matrix_flattened, SCORE_MATRIX_SIZE * sizeof(byte_type));

    // std::vector<std::tuple<std::string, std::string, int>> results;
    std::vector<std::tuple<int, int, int>> results;
    for (int k = 0; k < queries.size(); k++) {
        std::string query = queries[k];
        char * q_str = (char *) query.c_str();
        int q_len = query.size();
        // convert query to its indexes on aa2num
        char * q_str_idx = (char *)malloc(q_len * sizeof(char));
        for (int i = 0; i < q_len; i++) {
            q_str_idx[i] = (char)aa2num[q_str[i] - 'A'];
        }
        char * d_query_str_idx;
        cudaMalloc(&d_query_str_idx, q_len * sizeof(char));
        cudaMemcpy(d_query_str_idx, q_str_idx, q_len * sizeof(char), cudaMemcpyHostToDevice);

        // calculating the (maximum) size of different variables which will go on shared-memory
        // also a future todo: remove this score_matrix_len from argument list of kernel functions
        int q_tmax_len = (max_target_len + q_len);
        int opt_cells_size;
        // calculating the maximum needed shared-memory size in bytes
        int shared_memory_size;
        int block_size;
        s1 = clock();

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

        s2 = clock();
        #if defined SHOW_KERNEL_CONF
        printf("global_memory_size: %d, shared_memory_size: %d, num_targets=grid_size: %d, block_size: %d, q_len: %d, max_t_len: %d, mode: %s\n", global_memory_size, shared_memory_size, num_target_strs, block_size, q_len, max_target_len, on_columns?"on_columns":"on_diagonals");
        #endif
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        s3 = clock();
        #if defined DEBUG && defined SHOW_ALL_ALIGNMENT_SCORES
        std::cout << "final scores:" << std::endl;
        for (int i = 0; i < num_target_strs; i++) {
            std::cout << query.substr(0, 10) << "," << targets[i].substr(0, 10);
            printf("(%4d,%4d): %4d\n", q_len, (int)targets[i].size(), best_scores[i]);
        }
        #endif
        cudaFree(d_query_str_idx);
        s4 = clock();
        #if defined DEBUG
        printf("kernel steps clocks; gpu_kernel: %ld, cuda_sync: %ld, log+free: %ld\n", (s2 - s1), (s3 - s2), (s4 - s3));
        #endif
        start_clock = clock();
        int idx[num_target_strs];
        int ptr = 0;
        clock_t s11, s22, s33, s44;
        // clock_t s11, s22, s33, s44, s55;
        s11 = clock();
        for (int j = 0; j < num_target_strs; j++) {
            if (best_scores[j] > PREFILTER_SCORE_THRESHOLD) {
                idx[ptr++] = j;
            }
        }
        s22 = clock();
        // std::sort(idx, idx+ptr, [&](const int& a, const int& b) -> bool
        std::sort(idx, idx+ptr, [=](const int& a, const int& b) -> bool
            {return best_scores[a] > best_scores[b]; }
        );
        s33 = clock();
        for (int j = 0, lim = min(ptr, BENCH_CAP_VAL); j < lim; j++) {
            results.push_back(std::forward_as_tuple(k, idx[j], best_scores[idx[j]]));
        }
        s44 = clock();
#if defined DEBUG
        // std::sort(best_scores, best_scores+ptr, std::greater<int>());
        // s55 = clock();
        printf("filter and idx construction clocks: %ld, its sorting: %ld, grabbing the fixed amount: %ld\n", (s22 - s11), (s33 - s22), (s44 - s33));
        // printf("filter and idx construction clocks: %ld, its sorting: %ld, grabbing the fixed amount: %ld, original sorting: %ld\n", (s22 - s11), (s33 - s22), (s44 - s33), (s55 - s44));
        end_clock = clock();
        execution_clocks = end_clock - start_clock;
        std::cout << "vector contruction execution clocks: " << execution_clocks << ", elements_num: " << ptr << std::endl;
#endif
    }

    start_clock = clock();
    std::sort(results.begin(), results.end(), sort_by_score);
    end_clock = clock();
#if defined DEBUG
    std::cout << "sort execution clocks: " << end_clock - start_clock << std::endl;
    ee = clock();
    printf("total runtime of gpu search: %ld\n", (ee - ss));
#endif
    ss = clock();
    int results_size = results.size();
    #if defined DEBUG
    for (int i = 0; i < min(results_size, BENCH_CAP_VAL); i++) {
        std::tuple<int, int, int> t = results[i];
        std::cout << query_ids[std::get<0>(t)] << " " << target_ids[std::get<1>(t)] << " " << std::get<2>(t) << std::endl;
    }
    #endif
    // todo: this is hardcode! (chrono and ctime are needed and are included in bits/stc++)
    auto t = std::time(nullptr); auto tm = *std::localtime(&t); std::ostringstream oss_time;
    oss_time << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S_");
    std::string out_name = "PlayGround_/full_output_" + oss_time.str() + std::to_string(num_query_strs) + "_" + std::to_string(num_target_strs);
    debug(out_name);
    std::ofstream out_file(out_name);
    for (int i = 0; i < results_size; i++) {
        std::tuple<int, int, int> tuple = results[i];
        out_file << query_ids[std::get<0>(tuple)] << " " << target_ids[std::get<1>(tuple)] << " " << std::get<2>(tuple) << std::endl;
    }
    // freeing device memory
    free_strings_on_device_flattened(d_targets_fstr, d_target_indices);
    cudaFree(best_scores);
    ee = clock();
#if defined DEBUG
    printf("clocks for outputting results: %ld\n", (ee - ss));
#endif
}


void measure_kernel_time(const std::vector<std::string>& queries, const std::vector<std::string>& targets, const std::vector<std::string>& query_ids, const std::vector<std::string>& target_ids, bool on_columns=true, bool verbose=true, int number_of_calls=20) {
    clock_t start_clock, end_clock;
    long maximum_clocks = 0, minimum_clocks = LLONG_MAX, sum_clocks = 0;

    for (int i = 0; i < number_of_calls; i++) {
        start_clock = clock();

        call_kernel_multiple_queries(queries, targets, query_ids, target_ids, on_columns);

        end_clock = clock();

        long execution_clocks = end_clock - start_clock;
        maximum_clocks = max(maximum_clocks, execution_clocks), minimum_clocks = min(minimum_clocks, execution_clocks);
        sum_clocks += execution_clocks;
        if (verbose) std::cout << "execution clocks: " << execution_clocks << std::endl; // divide by CLOCKS_PER_SEC if actual time is needed
    }
    debug(maximum_clocks); debug(minimum_clocks); std::cout << "avg_clocks: " << sum_clocks / number_of_calls << std::endl;
}


int main(int argc, char** argv) {
    std::vector<std::string> queries, targets;
    init_score_matrix();
    // arguments are optional and the default ones are targets=TestSamples/targets.txt, is_fasta=true, on_columns=true, query=TestSamples/queries.txt[0]
    // the 1s argument tells the file path, the 2nd one tells if target file is fasta, the 3rd determines alignment method (on-columns or on-diagonals) and the 4th tell query path
    // only string "0" can change 2nd and 3rd arguments to "not fasta" and "not on_column" (i.e "on_diagonal")
    std::string target_path = "TestSamples/targets.fasta";
    std::string query_path = "TestSamples/queries.fasta";
    // bool is_target_fasta = false;
    // bool is_query_fasta = false;
    bool on_columns = true;
    if (argc > 1) {
        target_path = argv[1];
        if (argc > 2) {
            // if(std::string(argv[2]) == "1") is_target_fasta = true;
            if (argc > 3) {
                if (std::string(argv[3]) == "0") on_columns = false;
                if (argc > 4) {
                    query_path = argv[4];
                    if (argc > 5) {
                        // if(std::string(argv[2]) == "1") is_query_fasta = true;
                    }
                }
            }
        }
    }

    std::vector<std::string> query_ids, target_ids;
    init_input_from_fasta_file_with_id(query_path, queries, query_ids);
    init_input_from_fasta_file_with_id(target_path, targets, target_ids);
    printf("number of queries: %d, number of targets: %d\n", (int)queries.size(), (int)targets.size());

    call_kernel_multiple_queries(queries, targets, query_ids, target_ids, on_columns);
    // measure_kernel_time(queries, targets, query_ids, target_ids, on_columns);
    return 0;
}