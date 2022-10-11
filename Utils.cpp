
#include <bits/stdc++.h>
#include "Utils.h"
// using namespace std;

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
void init_input_from_file(std::string filename, std::vector<std::string>& results, bool is_fasta_format, bool align) {
    std::string cur_line;
    std::ifstream read_file(filename);
    
    if (!is_fasta_format) {
        while (std::getline(read_file, cur_line)) {
            if (align) {
                adjust_length(cur_line);
            }
            results.push_back(cur_line);
        }
    }
    else {
        int line_num = 1;
        while (std::getline(read_file, cur_line)) {
            if ((line_num++) % 2 == 0) {
                if (align) {
                    adjust_length(cur_line);
                }
                results.push_back(cur_line);
            }
        }
    }
}

// void init_input_from_fasta_file_with_id(string filename, vector<tuple<string, string>>& results, bool align) {
void init_input_from_fasta_file_with_id(std::string filename, std::vector<std::string>& results, std::vector<std::string>& result_ids, bool align) {
    std::string cur_line;
    std::string cur_id;
    std::ifstream read_file(filename);

    int line_num = 1;
    while (std::getline(read_file, cur_line)) {
        if ((line_num++) % 2 == 0) {
            if (align) {
                adjust_length(cur_line);
            }
            // results.push_back(std::make_tuple(cur_id, cur_line));
                results.push_back(cur_line);
                result_ids.push_back(cur_id);
        } else {
            int first_space = cur_line.find(" ");
            if (first_space != -1) {
                cur_id = cur_line.substr(1, first_space);
            } else {
                cur_id = cur_line.substr(1);
            }
        }
    }

}