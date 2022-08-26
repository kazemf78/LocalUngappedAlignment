// #ifndef SCORE_MATRIX_H_
// #define SCORE_MATRIX_H_

#define ALPH_SIZE 21
#define SCORE_MATRIX_SIZE 441
using namespace std;

extern char num2aa[ALPH_SIZE];
// todo: do sth better
extern int aa2num[(int)'Z' - 'A'];
extern int score_matrix[ALPH_SIZE][ALPH_SIZE];
extern int score_matrix_flattened[ALPH_SIZE * ALPH_SIZE];


void init_score_matrix_from_string(string);
void init_score_matrix();
int get_score(char, char);

// #endif