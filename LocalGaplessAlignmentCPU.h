// #ifndef LOCAL_GAPLESS_ALLIGNMENT_CPU_H_
// #define LOCAL_GAPLESS_ALLIGNMENT_CPU_H_
#include "Utils.h"

using namespace std;

alignment_result * local_ungapped_alignment(string, string );
alignment_result * local_ungapped_alignment_less_memory(string, string);
void measure_time(vector<string>, vector<string>, function<alignment_result* (string, string)>, bool, int);
