# LocalUngappedAlignment
## Prerequesties
ONLY [CUDA](https://docs.nvidia.com/cuda/index.html) :-"

TBH it can be a bit tricky to install CUDA and run your CUDA code properly. Aside from using tutorials for installing and etc, here are some NOTES worth mentioning in my opinion.
- Try to run the deviceQuery program on [cuda-samples](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/1_Utilities). The results are important specs of your GPU and some of them are crucial to know.
- Your GPU CUDA compatibility version (can be known with previous program or from [Wiki](https://en.wikipedia.org/wiki/CUDA), [Nvidia](https://developer.nvidia.com/cuda-gpus) or etc) matters a lot! It determines the features and technical specifications of your GPU ([compute-capabilities](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities)).
When compiling your CUDA program using `nvcc`, sometimes you need to specify the compatiblity version (with -arch or -gencode options) to prevent some unseen errors without logs!
## How to run!
### Makefile
to be done!
### Command Line
```
# compiling
nvcc <flags-required-for-nvcc> -o gpu_exe LocalGaplessAlignmentGPU.cu Utils.cpp ScoreMatrix.cpp -D{REDUCE_ALIGNMENT_RESULT,REDUCE_ON_COLUMNS}
# running (refer to the code for the options!)
./gpu_exe <targets-file> <is-fasta> <alignment-method> <query-file>
```
