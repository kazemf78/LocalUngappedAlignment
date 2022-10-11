// #ifndef UTILS_H_
// #define UTILS_H_
// #endif

#define int_type unsigned short
#define byte_type char

// using namespace std;

typedef struct {
    // int_type row = 0;
    // diagonal_idx = row - col + L_q - 1 (1 <= row <= L_t & 1 <= col <= L_q)
    // ranges in [0, L-t + L_q - 1)
    int_type diagonal_idx;
    int_type score = 0;
} opt_cell;

typedef struct {
    int best_score;
    int best_diagonal;
    opt_cell* best_cells;
} alignment_result;

void init_input_from_file(std::string, std::vector<std::string>&, bool, bool=true);
// void init_input_from_fasta_file_with_id(std::string, std::vector<std::tuple<std::string, std::string>>&, bool=true);
void init_input_from_fasta_file_with_id(std::string, std::vector<std::string>&, std::vector<std::string>&, bool=true);
