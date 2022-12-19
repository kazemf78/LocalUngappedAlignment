#!/bin/bash

# todo: handle the 3rd argument !!!
# ARGUMENTS for this script: 1. targets_file, 2. query_file, 3. gpu_model, 4. short_sequence_guarantee! (input files should have sequences in each line and not be fasta formatted!)

f () {
nvcc_flags="-arch=sm_50"
# these are for the cases that target_arch is NOT "armv7l aarch64 sbsa" [arch can be known with "uname -m"]
general_arch_nums="35 37 50 52 60 61 70 75 80 86"
for num in $general_arch_nums; do nvcc_flags_general+="-gencode arch=compute_$num,code=sm_$num "; done
mynvcc="nvcc $nvcc_flags_general"
sup_files="Utils.cpp ScoreMatrix.cpp"
cpp_main_file="LocalGaplessAlignmentCPU.cpp"
cu_main_file="LocalGaplessAlignmentGPUWarpShuffles.cu"
cu_main_file="LocalGaplessPrefilterGPU.cu"
# files="LocalGaplessAlignmentGPU.cu Utils.cpp ScoreMatrix.cpp"
flags="-lineinfo -std=c++11 -Wno-deprecated-gpu-targets" # suppress warning for deprecation of 35, 37, 50 versions
obj_file="exe_$gpu_model.out"
out_pipe="tail -5 | sed '2d'"
# exe_args="TestSamples/bulk_tiny.txt 0 0 TestSamples/bulk_query.txt"
targets="TestSamples/targets.fasta"
is_target_fasta="0"
query="TestSamples/queries.fasta"
is_query_fasta="0"

if [ ! -z "$1" ]
  then
    targets="$1"
    if [ ! -z "$2" ]
      then
        query="$2"
    fi
fi

base_command="nvcc {nvcc_flags} $cu_main_file $sup_files $flags {sup_flags} -o $obj_file && ./$obj_file {<target_file> <is_fasta> <IS on_columns> <query_file>} | {filter output}"
echo "BASE_COMMAND: $base_command"

echo "Time: $(date +"%Y-%m-%d_%H-%M-%S")"
sup_flags=("-DBENCHMARK_MULTI -DDEBUG -DSORT_RESULTS_LIMITED")
# sup_flags=("-DHANDLE_LONG_SEQUENCE")
# if [ ! -z "$4" ] && [ "$4" = "0" ]; then sup_flags=("" "-DHANDLE_LONG_SEQUENCE"); fi

for sup_flag in "${sup_flags[@]}"
do
out_pipe="sort -k3 -r"
out_pipe=""
if [ ! -z "$sup_flag" ]; then echo -e "\n---- The current ADDITIONAL FLAG is: $sup_flag ----"; fi

echo -e "\n#### ON_DIAGONAL method with REDUCTION"
echo $targets $query
command2="$mynvcc $cu_main_file $sup_files $flags $sup_flag -DREDUCE_ALIGNMENT_RESULT -o $obj_file && ./$obj_file $targets $is_target_fasta 0 $query $is_query_fasta"
if [ ! -z $out_pipe ]; then command2="$command2 | $out_pipe"; fi
echo $command2
eval $command2 #>> $outfilename

done
rm $obj_file
}

# NOTE: the last component of the path can't be a symlink and NO "cd" should have happened before this!
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR
if [ ! -z "$3" ]
  then
    gpu_model="$3"
fi
outfilename="PlayGround_/benchmark_$(date +"%Y-%m-%d_%H-%M-%S")_$gpu_model"
echo $gpu_model | tee $outfilename

start=$(date +%s.%N)
f "$@" | tee -a $outfilename
end=$(date +%s.%N)
runtime=$( echo "$end - $start" | bc -l )
printf "%.3f seconds was taken in total!\n" $runtime | tee -a $outfilename
