
#define AP_INT_MAX_W 8

#include "bnn-library.h"

// includes for network parameters
#include "weights.hpp"
#include "activations.hpp"
#include "thresh.h"

// defines for network parameters
#define Channels1 512
 #define InnerProdDim 75

            #define SIMD1 1
 #define PE1 2
 #define numReps 128

void Vector_Vector_Activate_Batch_70(hls::stream<ap_uint<8>> &in0,
            hls::stream<ap_uint<8>> &out
            )
{
#pragma HLS INTERFACE axis port=in0
#pragma HLS INTERFACE axis port=out
#pragma HLS stream depth=2 variable=in0
#pragma HLS stream depth=2 variable=out
#pragma HLS INTERFACE ap_ctrl_none port=return
#include "params.h"
#pragma HLS ARRAY_PARTITION variable=weights.m_weights complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs.m_thresholds complete dim=1
#pragma HLS ARRAY_PARTITION variable=threshs.m_thresholds complete dim=3
Vector_Vector_Activate_Batch<Channels1, InnerProdDim, SIMD1, PE1, 1, Slice<ap_uint<4>>, Slice<ap_int<4>>, Identity>
            (in0, out, weights, threshs, numReps, ap_resource_dsp());
}
