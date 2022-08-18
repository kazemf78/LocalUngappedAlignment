// remember!
#include <bits/stdc++.h>
#include "ScoreMatrix.h"

using namespace std;

#define MAX_LEN 1024
#define debug(A) cout << #A << ": " << A << endl


// #define DEBUG


typedef struct {
    int row = 0;
    // diagonal_idx = row - col + L_q (1 <= row <= L_t & 1 <= col <= L_q)
    // ranges from 1 to L_t + L_q - 1
    int diagonal_idx; 
    int score = 0;
} opt_cell;

typedef struct {
    int best_score;
    int best_diagonal;
    opt_cell* best_cells;
} alignment_result;

// convert each line of file to a string and adds it to the list
// if the file "is_fasta", then only even lines would be added
void init_input_from_file(string filename, vector<string>& results, bool is_fasta_format) {
    string cur_line;
    ifstream read_file(filename);
    
    if (!is_fasta_format) {
        while (getline(read_file, cur_line)) {
            results.push_back(cur_line);
        }
    }
    else {
        int line_num = 1;
        while (getline(read_file, cur_line)) {
            if ((line_num++) % 2 == 0)
                results.push_back(cur_line);
        }
    }
}

alignment_result * local_ungapped_alignment(string query, string target) {
    int q_len = query.size(), t_len = target.size();
    #ifdef DEBUG
    cout << "query_length = " << q_len << ", target_length = " << t_len << endl;
    #endif 
    
    int matrix[t_len+1][q_len+1];
    opt_cell best_cells[q_len + t_len];

    // initializations
    for (int i = 0; i <= q_len; i++)
        matrix[0][i] = 0;
    for (int i = 0; i <= t_len; i++)
        matrix[i][0] = 0;
    for (int i = 0; i < q_len + t_len; i++) {
        // invalid diagonal_idx for i = 0, but for more readability it was ignored
        best_cells[i].row = -1, best_cells[i].score = -1, best_cells[i].diagonal_idx = i;
    }
    
    int best_score = 0;
    int best_diag_idx = -1;
    for (int row = 1; row <= t_len; row++) {
        for (int col = 1; col <= q_len; col++) {
            // note that score_matrix must have been initialized somewhere before calling get_score
            // todo: maybe we can do sth more clear and still efficient about this preprocessing for get_score
            int current_score = max(0, matrix[row-1][col-1] + get_score(target[row-1], query[col-1]));
            int current_diagonal = row - col + q_len;
            matrix[row][col] = current_score;

            // holding the best_score overall and for each diagonal
            if (current_score > best_cells[current_diagonal].score) {
                best_cells[current_diagonal].score = current_score;
                best_cells[current_diagonal].row = row;
                if (current_score > best_score) {
                    best_score = current_score;
                    best_diag_idx = current_diagonal;
                }
            }
        }
    }
    #ifdef DEBUG
    for (int row = 0; row <= t_len; row++) {
        for (int col = 0; col <= q_len; col++) {
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
    static alignment_result res = {.best_score = best_score, .best_diagonal = best_diag_idx, .best_cells = best_cells};
    return &res;
}

// this function and local_ungapped_alignment are the same function but only memory usage differs
alignment_result * local_ungapped_alignment_less_memory(string query, string target) {
    int q_len = query.size(), t_len = target.size();
    #ifdef DEBUG
    cout << "query_length = " << q_len << ", target_length = " << t_len << endl;
    #endif 
    
    int last_row_scores[q_len+1];
    int current_row_scores[q_len+1];
    // usage of malloc because we will use this array outside of the function stack
    opt_cell* best_cells = (opt_cell *)malloc(sizeof(opt_cell) * (q_len + t_len));

    // initializations
    for (int i = 0; i <= q_len; i++)
        last_row_scores[i] = 0;
    current_row_scores[0] = 0;
    for (int i = 0; i < q_len + t_len; i++) {
        // invalid diagonal_idx for i = 0, but for more readability it was ignored
        best_cells[i].row = -1, best_cells[i].score = -1, best_cells[i].diagonal_idx = i;
    }
    
    int best_score = -1;
    int best_diag_idx = -1;
    for (int row = 1; row <= t_len; row++) {
        for (int col = 1; col <= q_len; col++) {
            // note that score_matrix must have been initialized somewhere before calling get_score
            // todo: maybe we can do sth more clear and still efficient about this preprocessing for get_score
            int current_score = max(0, last_row_scores[col-1] + get_score(target[row-1], query[col-1]));
            int current_diagonal = row - col + q_len;
            current_row_scores[col] = current_score;

            // holding the best_score overall and for each diagonal
            if (current_score > best_cells[current_diagonal].score) {
                best_cells[current_diagonal].score = current_score;
                best_cells[current_diagonal].row = row;
                if (current_score > best_score) {
                    best_score = current_score;
                    best_diag_idx = current_diagonal;
                }
            }
        }
        // this can be done more efficiently
        for (int col = 1; col <= q_len; col++) {
            last_row_scores[col] = current_row_scores[col];
        }
#ifdef DEBUG
        for (int col = 0; col <= q_len; col++) {
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



int main() {
    vector<string> queries, targets;
    init_input_from_file("TestSamples/queries.txt", queries, false);
    init_input_from_file("TestSamples/targets2.txt", targets, false);
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

#ifdef DEBUG
    // doing alignment only for one pair of qurey and target
    alignment_result *ret = local_ungapped_alignment_less_memory(queries[0], targets[0]);
    int lq = queries[0].size(), lt = targets[0].size();
    for (int i = 1; i < lq + lt; i++) {
        cout << ret->best_cells[i].diagonal_idx << " " << ret->best_cells[i].row << " " << ret->best_cells[i].score << endl;
    }
#endif
    return 0;
}