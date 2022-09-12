
#include <bits/stdc++.h>
#include "Utils.h"
using namespace std;

#define MAX_LEN 1024
#define debug(A) cout << #A << ": " << A << endl


// #define DEBUG

void adjust_length(std::string& str) {
    int mod32 = str.size() % 32;
    if (mod32 > 0) {
        str += std::string(32 - mod32, 'X');
    }
}

// convert each line of file to a string and adds it to the list
// if the file "is_fasta", then only even lines would be added
void init_input_from_file(string filename, vector<string>& results, bool is_fasta_format, bool align) {
    string cur_line;
    ifstream read_file(filename);
    
    if (!is_fasta_format) {
        while (getline(read_file, cur_line)) {
            if (align) {
                adjust_length(cur_line);
            }
            results.push_back(cur_line);
        }
    }
    else {
        int line_num = 1;
        while (getline(read_file, cur_line)) {
            if ((line_num++) % 2 == 0) {
                if (align) {
                    adjust_length(cur_line);
                }
                results.push_back(cur_line);
            }
        }
    }
}