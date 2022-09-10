#!/bin/bash

# ARGUMENTS for this script: 1. targets_file, 2. query_file, 3. short_sequence_guarantee! (input files should have sequences in each line and not be fasta formatted!)

f () {
nvcc_flags="-arch=sm_50"
# these are for the cases that target_arch is NOT "armv7l aarch64 sbsa" [arch can be known with "uname -m"] [35 & 37 are excluded from the list for now because of deprecation!]
general_arch_nums="50 52 60 61 70 75 80 86"
for num in $general_arch_nums; do nvcc_flags_general+="-gencode arch=compute_$num,code=sm_$num "; done
mynvcc="nvcc $nvcc_flags_general"
sup_files="Utils.cpp ScoreMatrix.cpp"
cpp_main_file="LocalGaplessAlignmentCPU.cpp"
cu_main_file="LocalGaplessAlignmentGPUWarpShuffles.cu"
# files="LocalGaplessAlignmentGPU.cu Utils.cpp ScoreMatrix.cpp"
flags="-lineinfo"
obj_file="exe.out"
out_pipe="tail -5 | sed '2d'"
# exe_args="TestSamples/bulk_tiny.txt 0 0 TestSamples/bulk_query.txt"
targets="TestSamples/targets.txt"
query="TestSamples/queries.txt"
if [ ! -z "$1" ]
  then
    targets="$1"
    if [ ! -z "$2" ]
      then
        query="$2"
    fi
fi


base_command="$mynvcc $cu_main_file $sup_files $flags {sup_flags} -o $obj_file && ./$obj_file {<target_file> <is_fasta> <IS on_columns> <query_file>} | {filter output}"
echo $base_command

echo "Time: $(date +"%Y-%m-%d_%H-%M-%S")"
sup_flags=("-DHANDLE_LONG_SEQUENCE")
if [ ! -z "$3" ] && [ "$3" = "0" ]; then sup_flags=("" "-DHANDLE_LONG_SEQUENCE"); fi

for sup_flag in "${sup_flags[@]}"
do
if [ ! -z $sup_flag ]; then echo -e "\n**** The current ADDITIONAL FLAG is: $sup_flag ****"; fi
echo -e "\n#### ON_DIAGONAL method without REDUCTION (with ATOMIC_FUNCTION)"
command1="$mynvcc $cu_main_file $sup_files $flags $sup_flag -o $obj_file && ./$obj_file $targets 0 0 $query | $out_pipe"
eval $command1 #>> $outfilename

echo -e "\n#### ON_DIAGONAL method with REDUCTION"
command2="$mynvcc $cu_main_file $sup_files $flags $sup_flag -DREDUCE_ALIGNMENT_RESULT -o $obj_file && ./$obj_file $targets 0 0 $query | $out_pipe"
eval $command2 #>> $outfilename

echo -e "\n#### ON_COLUMNS method without REDUCTION (with ATOMIC_FUNCTION)"
command3="$mynvcc $cu_main_file $sup_files $flags $sup_flag -o $obj_file && ./$obj_file $targets 0 1 $query | $out_pipe"
eval $command3 #>> $outfilename

echo -e "\n#### ON_COLUMNS method with REDUCTION"
command4="$mynvcc $cu_main_file $sup_files $flags $sup_flag -DREDUCE_ALIGNMENT_RESULT -o $obj_file && ./$obj_file $targets 0 1 $query | $out_pipe"
eval $command4 #>> $outfilename
done
}

# NOTE: the last component of the path can't be a symlink and NO "cd" should have happened before this!
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR
outfilename="PlayGround/benchmark_$(date +"%Y-%m-%d_%H-%M-%S")"
start=$(date +%s.%N)
f "$@" | tee $outfilename
end=$(date +%s.%N)
runtime=$( echo "$end - $start" | bc -l )
printf "%.3f seconds was taken!\n" $runtime | tee -a $outfilename
