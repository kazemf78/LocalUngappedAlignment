// remember!
#include <bits/stdc++.h>
#include "ScoreMatrix.h"
#include "LocalGaplessAlignmentCPU.h"


#define MAX_LEN 1024
#define debug(A) cout << #A << ": " << A << endl


// #define DEBUG
// #define SHOW_RESULTS


alignment_result * local_ungapped_alignment(std::string query, std::string target) {
    int q_len = query.size(), t_len = target.size();
    #ifdef DEBUG
    cout << "query_length = " << q_len << ", target_length = " << t_len << endl;
    #endif 
    
    int matrix[t_len+1][q_len+1];
    opt_cell best_cells[q_len + t_len - 1];

    // initializations
    for (int i = 0; i <= q_len; i++)
        matrix[0][i] = 0;
    for (int i = 0; i <= t_len; i++)
        matrix[i][0] = 0;
    for (int i = 0; i < q_len + t_len - 1; i++) {
        // best_cells[i].row = 0,
        best_cells[i].score = 0, best_cells[i].diagonal_idx = i;
    }
    
    int best_score = 0;
    int best_diag_idx = -1;
    for (int row = 1; row <= t_len; row++) {
        for (int col = 1; col <= q_len; col++) {
            // note that score_matrix must have been initialized somewhere before calling get_score
            // todo: maybe we can do sth more clear and still efficient about this preprocessing for get_score
            int current_score = max(0, matrix[row-1][col-1] + get_score(target[row-1], query[col-1]));
            int current_diagonal = row - col + q_len - 1;
            matrix[row][col] = current_score;

            // holding the best_score overall and for each diagonal
            if (current_score > best_cells[current_diagonal].score) {
                best_cells[current_diagonal].score = current_score;
                // best_cells[current_diagonal].row = row;
                if (current_score > best_score) {
                    best_score = current_score;
                    best_diag_idx = current_diagonal;
                }
            }
        }
    }
    #ifdef DEBUG
    for (int row = 1; row <= t_len; row++) {
        for (int col = 1; col <= q_len; col++) {
            printf("%2d ", matrix[row][col]);
        }
        cout << endl;
    }
    // cout << "best_diagonal_idx: " << best_diag_idx << ", best_score:" << best_score << endl;
    debug(best_diag_idx);
    debug(best_score);
    #endif
    // todo: find a better solution for handling local variable
    // todo: find a better way to return results?
    alignment_result* res = (alignment_result *)malloc(sizeof(alignment_result));
    res->best_score = best_score, res->best_diagonal = best_diag_idx, res->best_cells = best_cells;
    return res;
}

// this function and local_ungapped_alignment are the same function but only memory usage differs
alignment_result * local_ungapped_alignment_less_memory(std::string query, std::string target) {
    int q_len = query.size(), t_len = target.size();
    #ifdef DEBUG
    cout << "query_length = " << q_len << ", target_length = " << t_len << endl;
    #endif 
    
    int last_row_scores[q_len+1];
    int current_row_scores[q_len+1];
    // usage of malloc because we will use this array outside of the function stack
    opt_cell* best_cells = (opt_cell *)malloc(sizeof(opt_cell) * (q_len + t_len - 1));

    // initializations
    for (int i = 0; i <= q_len; i++)
        last_row_scores[i] = 0;
    // redundant operation
    // current_row_scores[0] = 0;
    for (int i = 0; i < q_len + t_len - 1; i++) {
        // best_cells[i].row = 0,
        best_cells[i].score = 0, best_cells[i].diagonal_idx = i;
    }
    
    int best_score = 0;
    int best_diag_idx = -1;
    for (int row = 1; row <= t_len; row++) {
        for (int col = 1; col <= q_len; col++) {
            // note that score_matrix must have been initialized somewhere before calling get_score
            // todo: maybe we can do sth more clear and still efficient about this preprocessing for get_score
            int current_score = max(0, last_row_scores[col-1] + get_score(target[row-1], query[col-1]));
            int current_diagonal = row - col + q_len - 1;
            current_row_scores[col] = current_score;

            // holding the best_score overall and for each diagonal
            if (current_score > best_cells[current_diagonal].score) {
                best_cells[current_diagonal].score = current_score;
                // best_cells[current_diagonal].row = row;
                if (current_score > best_score) {
                    best_score = current_score;
                    best_diag_idx = current_diagonal;
                }
            }
        }
        // this can be done more efficiently(?)
        for (int col = 1; col <= q_len; col++) {
            last_row_scores[col] = current_row_scores[col];
        }
#ifdef DEBUG
        for (int col = 1; col <= q_len; col++) {
            printf("%2d ", current_row_scores[col]);
        }
        cout << endl;
#endif
    }
    
    
    // todo: find a better solution for handling these variables and results that will be returned?
    alignment_result* res = (alignment_result *)malloc(sizeof(alignment_result)); 
    res->best_score = best_score, res->best_diagonal = best_diag_idx, res->best_cells = best_cells;
    return res;
}

void measure_time(std::string query, vector<std::string> targets, function<alignment_result* (std::string, std::string)> func, bool verbose=true, int number_of_calls=10) {
    clock_t start_clock, end_clock;
    long maximum_clocks = 0, minimum_clocks = LLONG_MAX, sum_clocks = 0;

    for (int i = 0; i < number_of_calls; i++) {
        start_clock = clock();
        for (auto &t: targets) {
            alignment_result* res = func(query, t);
        }
        end_clock = clock();
        long execution_clocks = end_clock - start_clock;
        maximum_clocks = max(maximum_clocks, execution_clocks), minimum_clocks = min(minimum_clocks, execution_clocks);
        sum_clocks += execution_clocks;
        if (verbose) {
            printf("num_targets: %d, q_len: %d\n", (int)targets.size(), (int)query.size());
            cout << "execution clocks: " << execution_clocks  << ", TIME: " <<  double(execution_clocks) / CLOCKS_PER_SEC << endl;
        }
    }
    debug(maximum_clocks); debug(minimum_clocks); cout << "avg_clocks: " << sum_clocks / number_of_calls << endl;
}

int main(int argc, char** argv) {
    vector<std::string> queries, targets;

    std::string target_path = "TestSamples/targets.txt";
    std::string query_path = "TestSamples/queries.txt";
    bool is_target_fasta = false;

    if (argc > 1) {
        target_path = argv[1];
        if (argc > 2) {
            if(std::string(argv[2]) == "0") is_target_fasta = false;
            if (argc > 3) {
                query_path = argv[3];
            }
        }
    }
    init_input_from_file(query_path, queries, false);
    init_input_from_file(target_path, targets, is_target_fasta);
#ifdef DEBUG
    for (auto &query: queries) {
        cout << query << endl;
    }
    cout << "-----" << endl;
    for (auto &target: targets) {
        cout << target << endl;
    }
#endif
    init_score_matrix();

#ifndef SHOW_RESULTS
    measure_time(queries[0], targets, &local_ungapped_alignment_less_memory, true, 10);
#else
    std::string q = queries[0];
    for (auto &t: targets) {
        alignment_result *res = local_ungapped_alignment_less_memory(q, t);
        int lq = q.size(), lt = t.size();
        cout << q.substr(0, 10) << "," << t.substr(0, 10);
        printf("(%4d,%4d): %4d\n", lq, lt, res->best_score);
        // printf("(%4d,%4d): %4d --> %d\n", lq, lt, res->best_score, res->best_diagonal);
     }
#endif
    return 0;
}