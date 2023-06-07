#ifndef __CONVDATA_H__
#define __CONVDATA_H__

// the first convolutional layer size
#define CONV1_FILTERS       64

// the second convolutional layer size
#define CONV2_FILTERS       32

typedef float KernelMat99[9][9];
typedef float ConvKernel64_99[CONV1_FILTERS][9][9];
typedef float ConvKernel32x64[CONV2_FILTERS][CONV1_FILTERS];
typedef float ConvKernel32_55[CONV2_FILTERS][5][5];
typedef float ConvKernel1[CONV1_FILTERS];
typedef float ConvKernel2[CONV2_FILTERS];
typedef float ConvKernel21[CONV2_FILTERS][CONV1_FILTERS];

/* The 64 cell bias in the first layer */
extern __constant__ const ConvKernel1 biases_conv1_cuda;
extern __constant__ const ConvKernel64_99 weights_conv1_data_cuda;
extern __constant__ const ConvKernel2 biases_conv2_cuda;
extern __constant__ const ConvKernel32x64 weights_conv2_data_cuda;
extern __constant__ const float biases_conv3_cuda;
extern __constant__ const ConvKernel32_55 weights_conv3_data_cuda;
#endif /// of __CONVDATA_H__
