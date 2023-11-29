/******************************************************************************
 * Copyright (C) 2022, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  1. Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *  2. Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *
 *  3. Neither the name of the copyright holder nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * @brief	Testbench for DWC.
 *****************************************************************************/

 module dwc_parallelwindow_tb();

//-------------------- Simulation parameters --------------------\\
    localparam int unsigned  IN_FOLD = 10;
    localparam int unsigned  OUT_FOLD = 2;
    localparam int unsigned  ACTIVATION_WIDTH = 4;
    localparam int unsigned  IN_WIDTH = ACTIVATION_WIDTH * IN_FOLD;
    localparam int unsigned  OUT_WIDTH = ACTIVATION_WIDTH * OUT_FOLD;

    localparam int unsigned  NUM_ITERS_IN = IN_FOLD > OUT_FOLD ? 1 : OUT_FOLD / IN_FOLD;
    localparam int unsigned  NUM_ITERS_OUT = OUT_FOLD > IN_FOLD ? 1 : IN_FOLD / OUT_FOLD;
    localparam int unsigned  INPUT_STREAM_WIDTH_BA = (IN_WIDTH+7)/8*8;
    localparam int unsigned  OUTPUT_STREAM_WIDTH_BA = (OUT_WIDTH+7)/8*8;
    localparam int unsigned  INPUT_STREAM_WIDTH_BA_DELTA = INPUT_STREAM_WIDTH_BA - IN_WIDTH;
    localparam int unsigned  OUTPUT_STREAM_WIDTH_BA_DELTA = OUTPUT_STREAM_WIDTH_BA - OUT_WIDTH;
    
    localparam int unsigned  IMG_H = 1;
    localparam int unsigned  IMG_W = 1;

    // Generate clk and reset signal
    logic clk = 0;
    always #5ns clk = !clk;

    logic ap_rst_n = 0;
    initial begin
        repeat(16) @(posedge clk);
        ap_rst_n <= 1;
    end

    uwire ap_clk = clk;

    // Generate input stimuli
    typedef logic [IN_FOLD-1:0][ACTIVATION_WIDTH-1:0] input_t; // [IN_FOLD][ACTIVATION_WIDTH-1:0]
    typedef input_t input_vector_t [NUM_ITERS_IN]; // [NUM_ITERS_IN][IN_FOLD][ACTIVATION_WIDTH-1:0]
    
    function input_vector_t init_INPUT;
        automatic input_vector_t res;
        std::randomize(res);
        return res;
    endfunction : init_INPUT
    
    input_vector_t  GOLDEN_INPUT = init_INPUT();

    // initial begin
    //     for (int i = 0; i<KERNEL_PROD*CHANNELS; i++) begin
    //         $display("IN[%0d][%0d] = %0d", i/CHANNELS, i%CHANNELS, GOLDEN_INPUT[0][0][i/CHANNELS][i%CHANNELS]);
    //     end
    // end

    struct {
        input_t dat;
        logic vld;
        logic rdy;
    } dut_input;

    initial begin
        dut_input.vld = 0;
        dut_input.dat = '{ default : 0};
        @(posedge clk iff ap_rst_n);

        for (int i = 0; i < NUM_ITERS_IN; i++) begin
            dut_input.dat <= GOLDEN_INPUT[i];
            do begin
                dut_input.vld <= $urandom()%7 >= 0;
                @(posedge clk);
            end while (!(dut_input.vld === 1 && dut_input.rdy === 1));
        end

        dut_input.vld <= 0;
        dut_input.dat <= '{ default : 1};
    end

    // Generate / compare against golden output
    typedef logic [OUT_FOLD-1:0][ACTIVATION_WIDTH-1:0] output_t; // [OUT_FOLD][ACTIVATION_WIDTH-1:0]
    typedef output_t output_matrix [NUM_ITERS_OUT]; // [NUM_ITERS_OUT][OUT_FOLD][ACTIVATION_WIDTH-1:0]
    
    struct {
        output_t dat;
        logic vld;
        logic rdy;
    } dut_output;

    // Input: [IN_FOLD][ACTIVATION_WIDTH-1:0]
    // Output: [NUM_ITERS_OUT][OUT_FOLD][ACTIVATION_WIDTH-1:0]
    function output_matrix compute_output(input_t a);
        automatic output_matrix res;
            for (int i = 0; i < NUM_ITERS_OUT; i++) begin
                for (int j = 0; j < OUT_FOLD; j++) begin
                    res[i][j] = a[i*OUT_FOLD + j];
                    //$display("OUT[%0d][%0d][%0d][%0d] = %0d", i, j, k, l, res[i][j][k][l]);
                end
            end
        return res;
    endfunction : compute_output;

// TODO: CONTINUE HERE
    // output_t = [OUT_FOLD][ACTIVATION_WIDTH-1:0]
    // Input: [OUT_FOLD] golden_output, [OUTPUT_STREAM_WIDTH_BA-1:0] dut_simulated_output
    function void check_output(output_t golden_output, output_t dut_simulated_output, int sf, int nf, int h, int w);
        for (int i = 0; i < OUT_FOLD; i++) begin
            automatic int g_o = golden_output[i][j];
            automatic int c_o = dut_simulated_output[i][j]; 
            //dut_simulated_output[(i*PE+j)*ACTIVATION_WIDTH +: ACTIVATION_WIDTH];
            assert (g_o == c_o) $display(">>> [t=%0t] Test succeeded (H, W) = (%0d, %0d), (NF, SF) = (%0d, %0d), (PE, SIMD) = (%0d, %0d), (Computed, Golden) = (%0d, %0d)", $time, h, w, nf, sf, j, i, c_o, g_o);
            else begin
                //$error(">>> [t=%0t] Test failed (SF, NF) = (%0d, %0d), (SIMD, PE) = (%0d, %0d), (Computed, Golden) = (%0d, %0d)", $time, SF, NF, i, j, g_o, c_o);
                //$stop;
                $display(">>> [t=%0t] Test failed (H, W) = (%0d, %0d), (NF, SF) = (%0d, %0d), (PE, SIMD) = (%0d, %0d), (Computed, Golden) = (%0d, %0d)", $time, h, w, nf, sf, j, i, c_o, g_o);
                $stop;
            end
        end
    endfunction : check_output;

    initial begin
        if (IN_FOLD > OUT_FOLD) begin
            dut_output.rdy = 0;
            @(posedge clk iff ap_rst_n);
            for (int i = 0; i < NUM_ITERS_OUT; i++) begin
                automatic output_matrix GOLDEN_OUTPUT = compute_output(GOLDEN_INPUT[i]);
                for (int j = 0; j < NUM_ITERS_OUT; j++) begin
                    do begin
                        dut_output.rdy <= $urandom()%7 >= 3;
                        @(posedge clk);
                    end while (!(dut_output.rdy === 1 && dut_output.vld === 1));
                    check_output(GOLDEN_OUTPUT[j], dut_output.dat, i, j);
                end
            end
            $finish;
        end
    end


    // Instantiate DUT
    dwc_parallelwindow #(
        .IN_WIDTH(IN_WIDTH), .OUT_WIDTH(OUT_WIDTH), .SIMD(SIMD), .PE(PE), .CHANNELS(CHANNELS), .KERNEL_PROD(KERNEL_PROD),
        .ACTIVATION_WIDTH(ACTIVATION_WIDTH)
    )
    dut (
        .ap_clk(ap_clk), .ap_rst_n(ap_rst_n), .s_axis_input_tdata(dut_input.dat), .s_axis_input_tvalid(dut_input.vld), .s_axis_input_tready(dut_input.rdy),
        .m_axis_output_tdata(dut_output.dat), .m_axis_output_tvalid(dut_output.vld), .m_axis_output_tready(dut_output.rdy)
    );

 endmodule : dwc_parallelwindow_tb