#ifndef OP
#define OP

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <mpi.h>
#include <mutex>
#include <thread>
#include <dlfcn.h>


/* 
 * Tensorflow Operation which given a tensor T 
 * produces a new Tensor T' which is the 
 * element-wise sum of T and all other Ts in the distributed runtime
 */
using namespace tensorflow;


using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;




REGISTER_OP("AggregateTensorsGetInfo")
      .Output("info: float")
      .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
        ::tensorflow::shape_inference::ShapeHandle output_shape = c->MakeShape({2});
        c->set_output(0, output_shape);

        return Status::OK();
    });



REGISTER_OP("AggregateTensorsInit");


REGISTER_OP("AggregateTensorsFinalize");




REGISTER_OP("AggregateTensors")
    .Attr("T: list({float, int32, double})")
    .Attr("num_elements: int")
    .Input("local_tensors: T")
    .Output("global_tensors: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });

REGISTER_OP("GatherTensors")
    .Attr("N: int")
    .Attr("T: {float32, int32, double}")
    .Attr("num_elements: int")
    .Input("local_tensors: T")
    .Output("global_tensors: N * T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });



#endif
