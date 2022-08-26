// #ifndef UTILS_H_
// #define UTILS_H_
// #endif

using namespace std;

typedef struct {
    int row = 0;
    // diagonal_idx = row - col + L_q - 1 (1 <= row <= L_t & 1 <= col <= L_q)
    // ranges in [0, L-t + L_q - 1)
    int diagonal_idx; 
    int score = 0;
} opt_cell;

typedef struct {
    int best_score;
    int best_diagonal;
    opt_cell* best_cells;
} alignment_result;

void init_input_from_file(string, vector<string>&, bool);