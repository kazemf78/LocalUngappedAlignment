// #ifndef SCORE_MATRIX_H_
// #define SCORE_MATRIX_H_

#define int_type unsigned short
#define byte_type char

#define ALPH_SIZE 21
#define SCORE_MATRIX_SIZE 441
using namespace std;

extern byte_type num2aa[ALPH_SIZE];
// todo: do sth better
extern byte_type aa2num['Z' - 'A'];
extern byte_type score_matrix[ALPH_SIZE][ALPH_SIZE];
extern byte_type score_matrix_flattened[ALPH_SIZE * ALPH_SIZE];


void init_score_matrix_from_string(string);
void init_score_matrix();
int get_score(char, char);

// #endif