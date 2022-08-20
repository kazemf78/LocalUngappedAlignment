
#include <bits/stdc++.h>
#include "Utils.h"
using namespace std;

#define MAX_LEN 1024
#define debug(A) cout << #A << ": " << A << endl


// #define DEBUG


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