#!/bin/bash

f () {
mynvcc="nvcc -arch=sm_50"
files="LocalGaplessAlignmentGPU.cu Utils.cpp ScoreMatrix.cpp"
files2="LocalGaplessAlignmentGPUWarpShuffles.cu Utils.cpp ScoreMatrix.cpp"
flags="-lineinfo"
obj_file="exe.out"
out_pipe="tail -5 | sed '2d'"
# exe_args="TestSamples/bulk_tiny.txt 0 0 TestSamples/bulk_query.txt"
targets="TestSamples/bulk_tiny2.txt"
query="TestSamples/bulk_query.txt"
if [ ! -z "$1" ]
  then
    targets="$1"
    if [ ! -z "$2" ]
      then
        query="$2"
    fi
fi


base_command="$mynvcc $files $flags $default_defines -o $obj_file && ./$obj_file $exe_args | $out_pipe"

echo "Time: $(date +"%Y-%m-%d_%H-%M-%S")"
echo "#### ON_DIAGONAL method without REDUCTION (with ATOMIC_FUNCTION)"
command1="$mynvcc $files $flags -o $obj_file && ./$obj_file $targets 0 0 $query | $out_pipe"
eval $command1 #>> $outfilename

echo -e "\n#### ON_DIAGONAL method with REDUCTION"
command2_1="$mynvcc $files $flags -DREDUCE_ALIGNMENT_RESULT -o $obj_file && ./$obj_file $targets 0 0 $query | $out_pipe"
eval $command2_1 #>> $outfilename

echo -e "\n#### ON_DIAGONAL method with MORE efficient REDUCTION"
command2_2="$mynvcc $files2 $flags -DREDUCE_ALIGNMENT_RESULT -o $obj_file && ./$obj_file $targets 0 0 $query | $out_pipe"
eval $command2_2 #>> $outfilename


echo -e "\n#### ON_COLUMNS method without REDUCTION (with ATOMIC_FUNCTION)"
command3="$mynvcc $files $flags -o $obj_file && ./$obj_file $targets 0 1 $query | $out_pipe"
eval $command3 #>> $outfilename

echo -e "\n#### ON_COLUMNS method with REDUCTION (reduce on COLUMNS)"
command4="$mynvcc $files $flags -DREDUCE_ON_COLUMNS -DREDUCE_ALIGNMENT_RESULT -o $obj_file && ./$obj_file $targets 0 1 $query | $out_pipe"
eval $command4 #>> $outfilename

echo -e "\n#### ON_COLUMNS method with REDUCTION (reduce on DIAGONALS)"
command5="$mynvcc $files $flags -DREDUCE_ALIGNMENT_RESULT -o $obj_file && ./$obj_file $targets 0 1 $query | $out_pipe"
eval $command5 #>> $outfilename
}

outfilename="PlayGround/benchmark_$(date +"%Y-%m-%d_%H-%M-%S")"
f "$@" | tee $outfilename

