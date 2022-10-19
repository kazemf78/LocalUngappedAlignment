#include <bits/stdc++.h>
#include "ScoreMatrix.h"

// using namespace std;

byte_type num2aa[ALPH_SIZE];
// todo: do sth better
byte_type aa2num['Z' - 'A'];
byte_type score_matrix[ALPH_SIZE][ALPH_SIZE];
byte_type score_matrix_flattened[ALPH_SIZE * ALPH_SIZE];


// todo: put this matrix in a seperate file!
std::string score_matrix_string_3Di =
R"(    A   C   D   E   F   G   H   I   K   L   M   N   P   Q   R   S   T   V   W   Y   X
A   6  -3   1   2   3  -2  -2  -7  -3  -3 -10  -5  -1   1  -4  -7  -5  -6   0  -2   0
C  -3   6  -2  -8  -5  -4  -4 -12 -13   1 -14   0   0   1  -1   0  -8   1  -7  -9   0
D   1  -2   4  -3   0   1   1  -3  -5  -4  -5  -2   1  -1  -1  -4  -2  -3  -2  -2   0
E   2  -8  -3   9  -2  -7  -4 -12 -10  -7 -17  -8  -6  -3  -8 -10 -10 -13  -6  -3   0
F   3  -5   0  -2   7  -3  -3  -5   1  -3  -9  -5  -2   2  -5  -8  -3  -7   4  -4   0
G  -2  -4   1  -7  -3   6   3   0  -7  -7  -1  -2  -2  -4   3  -3   4  -6  -4  -2   0
H  -2  -4   1  -4  -3   3   6  -4  -7  -6  -6   0  -1  -3   1  -3  -1  -5  -5   3   0
I  -7 -12  -3 -12  -5   0  -4   8  -5 -11   7  -7  -6  -6  -3  -9   6 -12  -5  -8   0
K  -3 -13  -5 -10   1  -7  -7  -5   9 -11  -8 -12  -6  -5  -9 -14  -5 -15   5  -8   0
L  -3   1  -4  -7  -3  -7  -6 -11 -11   6 -16  -3  -2   2  -4  -4  -9   0  -8  -9   0
M -10 -14  -5 -17  -9  -1  -6   7  -8 -16  10  -9  -9 -10  -5 -10   3 -16  -6  -9   0
N  -5   0  -2  -8  -5  -2   0  -7 -12  -3  -9   7   0  -2   2   3  -4   0  -8  -5   0
P  -1   0   1  -6  -2  -2  -1  -6  -6  -2  -9   0   4   0   0  -2  -4   0  -4  -5   0
Q   1   1  -1  -3   2  -4  -3  -6  -5   2 -10  -2   0   5  -2  -4  -5  -1  -2  -5   0
R  -4  -1  -1  -8  -5   3   1  -3  -9  -4  -5   2   0  -2   6   2   0  -1  -6  -3   0
S  -7   0  -4 -10  -8  -3  -3  -9 -14  -4 -10   3  -2  -4   2   6  -6   0 -11  -9   0
T  -5  -8  -2 -10  -3   4  -1   6  -5  -9   3  -4  -4  -5   0  -6   8  -9  -5  -5   0
V  -6   1  -3 -13  -7  -6  -5 -12 -15   0 -16   0   0  -1  -1   0  -9   3 -10 -11   0
W   0  -7  -2  -6   4  -4  -5  -5   5  -8  -6  -8  -4  -2  -6 -11  -5 -10   8  -6   0
Y  -2  -9  -2  -3  -4  -2   3  -8  -8  -9  -9  -5  -5  -5  -3  -9  -5 -11  -6   9   0
X   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0)";

std::string score_matrix_string_blosum62 =
R"(   A       C       D       E       F       G       H       I       K       L       M       N       P       Q       R       S       T       V       W       Y       X
A  3.9291 -0.4085 -1.7534 -0.8639 -2.2101  0.1596 -1.6251 -1.3218 -0.7340 -1.4646 -0.9353 -1.5307 -0.8143 -0.8040 -1.4135  1.1158 -0.0454 -0.1894 -2.5269 -1.7640 -1.0000
C -0.4085  8.5821 -3.4600 -3.6125 -2.3755 -2.5004 -2.9878 -1.2277 -3.0363 -1.2775 -1.4198 -2.6598 -2.7952 -2.9019 -3.3892 -0.8750 -0.8667 -0.8077 -2.3041 -2.4071 -1.0000
D -1.7534 -3.4600  5.7742  1.5103 -3.4839 -1.3135 -1.1189 -3.1212 -0.7018 -3.6057 -3.0585  1.2717 -1.4801 -0.3134 -1.6058 -0.2610 -1.0507 -3.1426 -4.2143 -3.0650 -1.0000
E -0.8639 -3.6125  1.5103  4.9028 -3.1924 -2.1102 -0.1177 -3.1944  0.7753 -2.8465 -1.9980 -0.2680 -1.1162  1.8546 -0.1154 -0.1469 -0.8633 -2.4423 -2.8354 -2.0205 -1.0000
F -2.2101 -2.3755 -3.4839 -3.1924  6.0461 -3.1074 -1.2342 -0.1609 -3.0787  0.4148  0.0126 -2.9940 -3.5973 -3.1644 -2.7863 -2.3690 -2.1076 -0.8490  0.9176  2.9391 -1.0000
G  0.1596 -2.5004 -1.3135 -2.1102 -3.1074  5.5633 -2.0409 -3.7249 -1.5280 -3.6270 -2.6766 -0.4228 -2.1335 -1.7852 -2.3041 -0.2925 -1.5754 -3.1387 -2.4915 -3.0398 -1.0000
H -1.6251 -2.9878 -1.1189 -0.1177 -1.2342 -2.0409  7.5111 -3.2316 -0.7210 -2.7867 -1.5513  0.5785 -2.1609  0.4480 -0.2499 -0.8816 -1.6859 -3.1175 -2.3422  1.6926 -1.0000
I -1.3218 -1.2277 -3.1212 -3.1944 -0.1609 -3.7249 -3.2316  3.9985 -2.6701  1.5216  1.1268 -3.2170 -2.7567 -2.7696 -2.9902 -2.3482 -0.7176  2.5470 -2.5805 -1.3314 -1.0000
K -0.7340 -3.0363 -0.7018  0.7753 -3.0787 -1.5280 -0.7210 -2.6701  4.5046 -2.4468 -1.3547 -0.1790 -1.0136  1.2726  2.1087 -0.2034 -0.6696 -2.2624 -2.9564 -1.8200 -1.0000
L -1.4646 -1.2775 -3.6057 -2.8465  0.4148 -3.6270 -2.7867  1.5216 -2.4468  3.8494  1.9918 -3.3789 -2.8601 -2.1339 -2.1546 -2.4426 -1.1975  0.7884 -1.6319 -1.0621 -1.0000
M -0.9353 -1.4198 -3.0585 -1.9980  0.0126 -2.6766 -1.5513  1.1268 -1.3547  1.9918  5.3926 -2.1509 -2.4764 -0.4210 -1.3671 -1.4809 -0.6663  0.6872 -1.4248 -0.9949 -1.0000
N -1.5307 -2.6598  1.2717 -0.2680 -2.9940 -0.4228  0.5785 -3.2170 -0.1790 -3.3789 -2.1509  5.6532 -2.0004  0.0017 -0.4398  0.6009 -0.0461 -2.8763 -3.6959 -2.0818 -1.0000
P -0.8143 -2.7952 -1.4801 -1.1162 -3.5973 -2.1335 -2.1609 -2.7567 -1.0136 -2.8601 -2.4764 -2.0004  7.3646 -1.2819 -2.1086 -0.8090 -1.0753 -2.3487 -3.6542 -2.9198 -1.0000
Q -0.8040 -2.9019 -0.3134  1.8546 -3.1644 -1.7852  0.4480 -2.7696  1.2726 -2.1339 -0.4210  0.0017 -1.2819  5.2851  0.9828 -0.1011 -0.6753 -2.1984 -1.9465 -1.4211 -1.0000
R -1.4135 -3.3892 -1.6058 -0.1154 -2.7863 -2.3041 -0.2499 -2.9902  2.1087 -2.1546 -1.3671 -0.4398 -2.1086  0.9828  5.4735 -0.7648 -1.1223 -2.5026 -2.6794 -1.6939 -1.0000
S  1.1158 -0.8750 -0.2610 -0.1469 -2.3690 -0.2925 -0.8816 -2.3482 -0.2034 -2.4426 -1.4809  0.6009 -0.8090 -0.1011 -0.7648  3.8844  1.3811 -1.6462 -2.7519 -1.6858 -1.0000
T -0.0454 -0.8667 -1.0507 -0.8633 -2.1076 -1.5754 -1.6859 -0.7176 -0.6696 -1.1975 -0.6663 -0.0461 -1.0753 -0.6753 -1.1223  1.3811  4.5453 -0.0555 -2.4289 -1.6060 -1.0000
V -0.1894 -0.8077 -3.1426 -2.4423 -0.8490 -3.1387 -3.1175  2.5470 -2.2624  0.7884  0.6872 -2.8763 -2.3487 -2.1984 -2.5026 -1.6462 -0.0555  3.7689 -2.8343 -1.2075 -1.0000
W -2.5269 -2.3041 -4.2143 -2.8354  0.9176 -2.4915 -2.3422 -2.5805 -2.9564 -1.6319 -1.4248 -3.6959 -3.6542 -1.9465 -2.6794 -2.7519 -2.4289 -2.8343 10.5040  2.1542 -1.0000
Y -1.7640 -2.4071 -3.0650 -2.0205  2.9391 -3.0398  1.6926 -1.3314 -1.8200 -1.0621 -0.9949 -2.0818 -2.9198 -1.4211 -1.6939 -1.6858 -1.6060 -1.2075  2.1542  6.5950 -1.0000
X -1.0000 -1.0000 -1.0000 -1.0000 -1.0000 -1.0000 -1.0000 -1.0000 -1.0000 -1.0000 -1.0000 -1.0000 -1.0000 -1.0000 -1.0000 -1.0000 -1.0000 -1.0000 -1.0000 -1.0000 -1.0000)";


void init_score_matrix_from_string_int(std::string matrix_string) {
    std::stringstream input_string(matrix_string);
    char single_token;
    for (int i = 0; i < ALPH_SIZE; i++) {
        input_string >> single_token;
        num2aa[i] = single_token;
        aa2num[single_token - 'A'] = i;
    }
    for (int i = 0; i < ALPH_SIZE; i++) {
        input_string >> single_token;
        int temp_score;
        for (int j = 0; j < ALPH_SIZE; j++) {
            input_string >> temp_score;
            score_matrix[i][j] = temp_score;
            score_matrix_flattened[i * ALPH_SIZE + j] = temp_score;
        }
    }
}

void init_score_matrix_from_string_double(std::string matrix_string, double bit_factor) {
    std::stringstream input_string(matrix_string);
    char single_token;
    for (int i = 0; i < ALPH_SIZE; i++) {
        input_string >> single_token;
        num2aa[i] = single_token;
        aa2num[single_token - 'A'] = i;
    }
    for (int i = 0; i < ALPH_SIZE; i++) {
        input_string >> single_token;
        double temp_score;
        for (int j = 0; j < ALPH_SIZE; j++) {
            input_string >> temp_score;
            double scaled_score = bit_factor * temp_score;
            char temp_sub_score = (scaled_score < 0) ? scaled_score - 0.5 : scaled_score + 0.5;
            score_matrix[i][j] = temp_sub_score;
            score_matrix_flattened[i * ALPH_SIZE + j] = temp_sub_score;
        }
    }
}

void init_score_matrix() {
    init_score_matrix_from_string_double(score_matrix_string_blosum62);
}

int get_score(char before, char after) {
    // todo: dirty code
    int score = score_matrix[aa2num[before - 'A']][aa2num[after - 'A']];
    return score;
}
