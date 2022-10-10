#!/bin/bash

# NOTE: the last component of the path can't be a symlink and NO "cd" should have happened before this!
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

# ARGUMENTS for this script: 1. main_cuda_file, 2. targets_file, 3. query_file (input files should have sequences in each line and not be fasta formatted!)
mynvcc="nvcc -arch=sm_50"
sup_files="Utils.cpp ScoreMatrix.cpp"
cpp_main_file="LocalGaplessAlignmentCPU.cpp"
cu_main_file="LocalGaplessAlignmentGPUWarpShuffles.cu"
flags="-lineinfo"
obj_file="exe.out"
# exe_args="TestSamples/bulk_tiny.txt 0 0 TestSamples/bulk_query.txt"
targets="TestSamples/targets.txt"
query="TestSamples/queries.txt"
if [ ! -z "$1" ]
then
    cu_main_file="$1"
    if [ ! -z "$2" ]
    then
        targets="$2"
        if [ ! -z "$3" ]
        then
            query="$3"
        fi
    fi
fi


tmp_file1="res1.tmp"
tmp_file2="res2.tmp"
tmp_file3="res3.tmp"

# echo "Time: $(date +"%Y-%m-%d_%H-%M-%S")"
start=$(date +%s.%N)
#### capturing ON_DIAGONAL and ON_COLUMNS methods with REDUCTION outputs
flags="$flags -DDEBUG -DREDUCE_ALIGNMENT_RESULT"
command1="$mynvcc $cu_main_file $sup_files $flags -o $obj_file && ./$obj_file $targets 0 0 $query"
command2="$mynvcc $cu_main_file $sup_files $flags -o $obj_file && ./$obj_file $targets 0 1 $query"
echo "Running on GPU with on_diagonals method..."
eval $command1 > $tmp_file1
echo "Running on GPU with on_columns method..."
eval $command2 > $tmp_file2

flags="-DSHOW_RESULTS"
command3="g++ $cpp_main_file $sup_files $flags -o $obj_file && ./$obj_file $targets 0 $query"
echo "Running on CPU..."
eval $command3 > $tmp_file3

diff13=$(diff $tmp_file3 <(tail -n +3 $tmp_file1) | wc -l)
diff23=$(diff $tmp_file3 <(tail -n +3 $tmp_file2) | wc -l)
end=$(date +%s.%N)
runtime=$( echo "$end - $start" | bc -l )
printf "%.3f seconds was taken!\n" $runtime
if [ $diff13 -eq 0 ] && [ $diff23 -eq 0 ]
then
    echo "The results match and it seems OK :D"
    # diff $tmp_file{1,2} | head
    head $tmp_file{1,2} -n5
else
    echo "Something's wrong :(("
    diff $tmp_file3 <(tail -n +3 $tmp_file1) | head
    diff $tmp_file3 <(tail -n +3 $tmp_file2) | head
fi

rm $tmp_file{1,2,3}