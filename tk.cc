/*
 * Copyright (c) 2017-2018 ARM Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include <iostream>
#include <fstream>
#include "arm_compute/graph/Graph.h"
#include "arm_compute/graph/Nodes.h"
#include "arm_compute/graph/SubGraph.h"
#include "support/ToolchainSupport.h"
#include "utils/GraphUtils.h"
#include "utils/Utils.h"
#include "arm_compute/runtime/CL/CLScheduler.h"

using namespace arm_compute::utils; 		//utils/Utils.h
using namespace arm_compute::graph; 		//graph/Graph.h
using namespace arm_compute::graph_utils; 	//graph/GraphUtils.h
using namespace arm_compute::logging;		//

namespace
{
} // namespace

/** Example demonstrating how to implement Squeezenet's network using the Compute Library's graph API
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments ( [optional] Target (0 = NEON, 1 = OpenCL), [optional] Path to the weights folder, [optional] image, [optional] labels )
 */

std::unique_ptr<ITensorAccessor> dummy() {
    return arm_compute::support::cpp14::make_unique<DummyAccessor>(1);
}

class GraphSqueezeDetExample : public Example
{
public:
    void do_setup(int argc, char **argv) override
    {
        std::string data_path; /* Path to the trainable data */
        std::string image;     /* Image data */
        std::string label;     /* Label data */
	std::ofstream output_path("./tk.txt", std::ios::out);

        ConvolutionMethodHint convolution_hint;
        arm_compute::DataType type;

	std::cout << "start\n";

        // Create a preprocessor object
        const std::array<float, 3> mean_rgb{ { 122.68f, 116.67f, 104.01f } };
        std::unique_ptr<IPreprocessor> preprocessor = arm_compute::support::cpp14::make_unique<CaffePreproccessor>(mean_rgb);

	// 1. backend
        // Set target. 0 (NEON), 1 (OpenCL), 2 (OpenCL with Tuner). By default it is NEON
        const int  int_target_hint = argc > 1 ? std::strtol(argv[1], nullptr, 10) : 0;
        TargetHint target_hint     = set_target_hint(int_target_hint);

        // Parse arguments
        if(argc < 2)
        {
            // Print help
    	    convolution_hint = ConvolutionMethodHint::GEMM;
    	    type = DataType::F32;
            std::cout << "Usage: " << argv[0] << " [target] [path_to_data] [image] [labels] [conv_method] [dtype]\n\n";
            std::cout << "No data folder provided: using random values\n\n";
        }
        else if(argc == 2)
        {
    	    convolution_hint = ConvolutionMethodHint::GEMM;
    	    type = DataType::F32;
            std::cout << "Usage: " << argv[0] << " " << argv[1] << " [path_to_data] [image] [labels] [conv_method] [dtype]\n\n";
            std::cout << "No data folder provided: using random values\n\n";
        }
        else if(argc == 3)
        {
    	    convolution_hint = ConvolutionMethodHint::GEMM;
    	    type = DataType::F32;
            //data_path = argv[2];
            std::cout << "Usage: " << argv[0] << " " << argv[1] << " " << argv[2] << " [image] [labels] [conv_method] [dtype]\n\n";
            std::cout << "No image provided: using random values\n\n";
        }
        else if(argc == 4)
        {
    	    convolution_hint = ConvolutionMethodHint::GEMM;
    	    type = DataType::F32;
            //data_path = argv[2];
            //image     = argv[3];
            std::cout << "Usage: " << argv[0] << " " << argv[1] << " " << argv[2] << " " << argv[3] << " [labels] [conv_method] [dtype]\n\n";
            std::cout << "No text file with labels provided: skipping output accessor\n\n";
        }
        else if(argc == 5)
        {
    	    convolution_hint = ConvolutionMethodHint::GEMM;
    	    type = DataType::F32;
            //data_path = argv[2];
            //image     = argv[3];
	    //label     = argv[4];    

            std::cout << "Usage: " << argv[0] << " " << argv[1] << " " << argv[2] << " " << argv[3] << " " << argv[4] << " [conv_method] [dtype]\n\n";
            std::cout << "No convolution method provided: using gemm or direct\n\n";
        }
        else if(argc == 6)
        {
            //data_path = argv[2];
            //image     = argv[3];
	    //label     = argv[4];
    	    std::string conv_method = argv[5];
    	    if (conv_method == "gemm") {
    	        convolution_hint = ConvolutionMethodHint::GEMM;
    	    } else {
    	        convolution_hint = ConvolutionMethodHint::DIRECT;
    	    }
    	    type = DataType::F32;
            std::cout << "Usage: " << argv[0] << " " << argv[1] << " " << argv[2] << " " << argv[3] << " " << argv[4] << " " << argv[5] << " [dtype]\n\n";
            std::cout << "No data type provided: using float32 or float16\n\n";
        }
        else
        {
            //data_path = argv[2];
            //image     = argv[3];
            //label     = argv[4];
	    std::cout << "1\n";
    	    std::string conv_method = argv[5];
    	    if (conv_method == "gemm") {
    	        convolution_hint = ConvolutionMethodHint::GEMM;
    	    } else {
    	        convolution_hint = ConvolutionMethodHint::DIRECT;
    	    }
	    std::cout << "2\n";
    	    std::string dtype = argv[6];

	    //note The tensor data type for the inputs must be U8/QS8/QS16/S16/F16/F32.
    	    if (dtype == "float32") {
    	        type = DataType::F32;
    	    } else if(dtype == "float16"){
		type = DataType::F16;
	    } else if(dtype == "uint8"){
		type = DataType::U8;
	    } else if(dtype == "fixed16"){
		type = DataType::QS16;
	    } else if(dtype == "fixed8"){
		type = DataType::QS8;
	    } else {
    	        type = DataType::S16;
    	    }
        }
   	std::cout << "okay input is okay\n";


	// let's make graph !!! 
        graph << target_hint 
	      //<< convolution_hint
              << Tensor(TensorInfo(TensorShape(1224U, 370U, 3U, 1U), 1, type),
                        get_input_accessor(image, std::move(preprocessor)))
              << ConvolutionLayer(
                  3U, 3U, 64U, dummy(), dummy(),
                  //get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/conv1_w.npy"), //weight
                  //get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/conv1_b.npy"),
                  PadStrideInfo(2, 2, 0, 0)) //stride size, padding size
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
              << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL)))

		  //--------------------------------------------------------------------------
              << ConvolutionLayer(
                  1U, 1U, 16U, dummy(), dummy(),
                  //get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire2_squeeze1x1_w.npy"),
                  //get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire2_squeeze1x1_b.npy"),
                  PadStrideInfo(1, 1, 0, 0))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
              << get_expand_fire_node(data_path, "fire2", 64U, 64U)

              << ConvolutionLayer(
                  1U, 1U, 16U, dummy(), dummy(),
                  //get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire3_squeeze1x1_w.npy"),
                  //get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire3_squeeze1x1_b.npy"),
                  PadStrideInfo(1, 1, 0, 0))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
              << get_expand_fire_node(data_path, "fire3", 64U, 64U)

              << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL)))
		  //--------------------------------------------------------------------------

              << ConvolutionLayer(
                  1U, 1U, 32U, dummy(), dummy(),
                  //get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire4_squeeze1x1_w.npy"),
                  //get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire4_squeeze1x1_b.npy"),
                  PadStrideInfo(1, 1, 0, 0))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
              << get_expand_fire_node(data_path, "fire4", 128U, 128U)

              << ConvolutionLayer(
                  1U, 1U, 32U, dummy(), dummy(),
                  //get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire5_squeeze1x1_w.npy"),
                  //get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire5_squeeze1x1_b.npy"),
                  PadStrideInfo(1, 1, 0, 0))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
              << get_expand_fire_node(data_path, "fire5", 128U, 128U)

              << PoolingLayer(PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0, DimensionRoundingType::CEIL)))
		  //--------------------------------------------------------------------------

              << ConvolutionLayer(
                  1U, 1U, 48U, dummy(), dummy(),
                  //get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire6_squeeze1x1_w.npy"),
                  //get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire6_squeeze1x1_b.npy"),
                  PadStrideInfo(1, 1, 0, 0))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
              << get_expand_fire_node(data_path, "fire6", 192U, 192U)

              << ConvolutionLayer(
                  1U, 1U, 48U, dummy(), dummy(),
                  //get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire7_squeeze1x1_w.npy"),
                  //get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire7_squeeze1x1_b.npy"),
                  PadStrideInfo(1, 1, 0, 0))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
              << get_expand_fire_node(data_path, "fire7", 192U, 192U)

              << ConvolutionLayer(
                  1U, 1U, 64U, dummy(), dummy(),
                  //get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire8_squeeze1x1_w.npy"),
                  //get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire8_squeeze1x1_b.npy"),
                  PadStrideInfo(1, 1, 0, 0))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
              << get_expand_fire_node(data_path, "fire8", 256U, 256U)

              << ConvolutionLayer(
                  1U, 1U, 64U, dummy(), dummy(),
                  //get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire9_squeeze1x1_w.npy"),
                  //get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire9_squeeze1x1_b.npy"),
                  PadStrideInfo(1, 1, 0, 0))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
              << get_expand_fire_node(data_path, "fire9", 256U, 256U)

		  //--------------------------------------------------------------------------
              << ConvolutionLayer(
                  1U, 1U, 96U, dummy(), dummy(),
                  //get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire8_squeeze1x1_w.npy"),
                  //get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire8_squeeze1x1_b.npy"),
                  PadStrideInfo(1, 1, 0, 0))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
              << get_expand_fire_node(data_path, "fire10", 384U, 384U)

              << ConvolutionLayer(
                  1U, 1U, 96U, dummy(), dummy(),
                  //get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire9_squeeze1x1_w.npy"),
                  //get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/fire9_squeeze1x1_b.npy"),
                  PadStrideInfo(1, 1, 0, 0))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))
              << get_expand_fire_node(data_path, "fire11", 384U, 384U)
		
		//dropout??

		  //--------------------------------------------------------------------------

              << ConvolutionLayer(
                  3U, 3U, 72U, dummy(), dummy(),
                  //get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/conv10_w.npy"),
                  //get_weights_accessor(data_path, "/cnn_data/squeezenet_v1.0_model/conv10_b.npy"),
                  PadStrideInfo(1, 1, 1, 1))
              << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU))

              //<< PoolingLayer(PoolingLayerInfo(PoolingType::AVG))
		  //--------------------------------------------------------------------------
              << FlattenLayer()
              << SoftmaxLayer()
              << Tensor(get_output_accessor(label, 5, output_path));
	      //<< ReshapeLayer(TensorShape(72U))
	      //<< Tensor(TensorInfo(TensorShape(72U), 

        // In order to enable the OpenCL tuner, graph_init() has to be called only when all nodes have been instantiated
        graph.graph_init(int_target_hint == 2);

	
	std::cout << "do_setup() finished\n";
    }
    Graph* getGraph() {
        return &(this->graph);
    }
    void do_run() override
    {
        // Run graph
        graph.run();
	std::cout << "do_run() finished\n";
    }

private:
    Graph graph{};

    BranchLayer get_expand_fire_node(const std::string &data_path, std::string &&param_path, unsigned int expand1_filt, unsigned int expand3_filt)
    {
        std::string total_path = "/cnn_data/squeezenet_v1.0_model/" + param_path + "_";
        SubGraph    i_a;
        i_a << ConvolutionLayer(
                1U, 1U, expand1_filt,
                get_weights_accessor(data_path, total_path + "expand1x1_w.npy"),
                get_weights_accessor(data_path, total_path + "expand1x1_b.npy"),
                PadStrideInfo(1, 1, 0, 0))
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        SubGraph i_b;
        i_b << ConvolutionLayer(
                3U, 3U, expand3_filt,
                get_weights_accessor(data_path, total_path + "expand3x3_w.npy"),
                get_weights_accessor(data_path, total_path + "expand3x3_b.npy"),
                PadStrideInfo(1, 1, 1, 1))
            << ActivationLayer(ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        return BranchLayer(BranchMergeMethod::DEPTH_CONCATENATE, std::move(i_a), std::move(i_b));
    }
};
//---------------------------------------------------------------------------------------------------
double measure(Graph *graph, int n_times) {
    arm_compute::CLScheduler::get().default_init();
    graph->run();
    arm_compute::CLScheduler::get().sync();

    auto tbegin = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n_times; i++) {
        graph->run();
	std::cout << "do_run() finished\n";
    }

    arm_compute::CLScheduler::get().sync();
    auto tend = std::chrono::high_resolution_clock::now();

    double cost = std::chrono::duration_cast<std::chrono::duration<double>>(tend - tbegin).count();
    return cost / n_times;
}

double run_case(int argc, char **argv){

    GraphSqueezeDetExample example;
    example.do_setup(argc,argv);
    Graph *graph = example.getGraph();

    int num_warmup, num_test;
    num_warmup = 1; //10
    num_test   = 5; //60
    //num_warmup *= 5;
    //num_test   *= 5;

    // warm up
    measure(graph, num_warmup);

    // test
    double cost = measure(graph, num_test);
    return cost;
}
//---------------------------------------------------------------------------------------------------

/** Main program for Squeezenet v1.0
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments ( [optional] Target (0 = NEON, 1 = OpenCL), [optional] Path to the weights folder, [optional] image, [optional] labels )
 */
int main(int argc, char **argv)
{
    // utils/Utils.cpp
    //return arm_compute::utils::run_example<GraphSqueezenetExample>(argc, argv);
    std::cout << "go\n";

    double cost = run_case(argc, argv);
    std::cout << "cost is " << cost << "\n";
    std::stringstream ss;
    ss << "backend: " << argv[1] << "\tdatapath: " << argv[2]
	   << "\timage: " << argv[3] << "\tlabel: " << argv[4]
           << "\tconv_method: " << argv[5] << "\tdtype: " << argv[6];

    std::cout << ss.str() << std::endl;
    
    return 0;
}
