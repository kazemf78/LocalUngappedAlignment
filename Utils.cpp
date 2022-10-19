
#include <bits/stdc++.h>
#include "Utils.h"
// using namespace std;

#define MAX_LEN 1024
#define debug(A) cout << #A << ": " << A << endl


// #define DEBUG

std::vector<std::string> split(std::string s, std::string delimiter) {
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    std::string token;
    std::vector<std::string> res;

    while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
        token = s.substr (pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back (token);
    }

    res.push_back(s.substr (pos_start));
    return res;
}

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
            size_t first_space = cur_line.find(" ");
            if (first_space != std::string::npos) {
                char sep = '|';
                cur_id = cur_line.substr(1, first_space-1);
                std::string tmp_line = cur_line.substr(first_space+1);
                std::vector<std::string> strs = split(tmp_line, " |");
                std::replace(strs[0].begin(), strs[0].end(), ' ', ',');
                std::replace(strs[2].begin(), strs[2].end(), ' ', ',');
                cur_id += sep + strs[0] + sep + strs[2];
            } else {
                cur_id = cur_line.substr(1);
            }
        }
    }

}