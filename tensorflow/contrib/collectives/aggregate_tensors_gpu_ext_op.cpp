#include <cuda_runtime.h>
#include "../include/aggregate_tensors_op.h"

/*
 * Tensorflow Op definitions for CPU and GPU
 * 
*/



/*
 * Initializes MPI environment
 * This function should be called once per process or else MPI
 * will return an error. 
 * Note* we must call dlopen since we are initializing MPI from
 * a local scope*
 *
*/

class AggregateTensorsInit : public OpKernel {
private:
  int initialized=0;
public:
  // constructor is thread-safe
  explicit AggregateTensorsInit(OpKernelConstruction *ctx): OpKernel(ctx) {
  }

  void Compute(OpKernelContext *ctx) override {

    int res;
    if(__sync_bool_compare_and_swap(&initialized, false, true)){
        dlopen("libmpi.so", RTLD_NOW | RTLD_GLOBAL);
        int check = MPI_Init_thread(NULL,NULL, MPI_THREAD_MULTIPLE, &res);
        assert(check ==0);
 //       assert (res == MPI_THREAD_MULTIPLE);
        
        initialized=1;
    }
  }


};
REGISTER_KERNEL_BUILDER(Name("AggregateTensorsInit").Device(DEVICE_CPU), AggregateTensorsInit);
REGISTER_KERNEL_BUILDER(Name("AggregateTensorsInit").Device(DEVICE_GPU), AggregateTensorsInit);


/*
 * Returns a 1D tensor with two elements
 * First elements is the number of processes in MPI_COMM_WORLD
 * Second element is the rank of the calling process
*/


class AggregateTensorsGetInfo : public OpKernel {
private:
  int initialized=0;
public:
  // constructor is thread-safe
  explicit AggregateTensorsGetInfo(OpKernelConstruction *ctx): OpKernel(ctx) {
  }

  void Compute(OpKernelContext *ctx) override {
      
        
        tensorflow::TensorShape t_s;
        t_s.AddDim(2);
        Tensor * global_tensor = NULL;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, t_s,
                                              &global_tensor));

        auto output_data = global_tensor->flat<float>();
//        int * output_data = global_tensor->flat<int>().data();
        int size;
        int rank;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        output_data(0) = size;
        output_data(1) = rank;
 //       MPI_Comm_rank(MPI_COMM_WORLD, &output_data[1]);
        
    }

};
REGISTER_KERNEL_BUILDER(Name("AggregateTensorsGetInfo").Device(DEVICE_CPU), AggregateTensorsGetInfo);
REGISTER_KERNEL_BUILDER(Name("AggregateTensorsGetInfo").Device(DEVICE_GPU), AggregateTensorsGetInfo);


/*
 * Finalizes the MPI Environment
 * After this call subsequent MPI calls will be undefined
*/


class AggregateTensorsFinalize : public OpKernel {
private:
  int initialized=0;
public:
  // constructor is thread-safe
  explicit AggregateTensorsFinalize(OpKernelConstruction *ctx): OpKernel(ctx) {
  }

  void Compute(OpKernelContext *ctx) override {

    MPI_Finalize();
        
  }


};
REGISTER_KERNEL_BUILDER(Name("AggregateTensorsFinalize").Device(DEVICE_CPU), AggregateTensorsFinalize);
REGISTER_KERNEL_BUILDER(Name("AggregateTensorsFinalize").Device(DEVICE_GPU), AggregateTensorsFinalize);



/*
 * Given a list of tensors returns the element-wise sum
 * of a tensor at index i with all other tensors at index i in
 * the MPI environment
 *
 * Example:
 *  [T0_0, T1_0, T2_0] on process 0
 *  [T0_1, T1_1, T2_1] on process 1
 *
 *  the returned list of tensors will be
 *  [T0_0+T0_1, T1_0+T1_1, T2_0+T2_1]
 *  Where each sum is the element wise of tensors
 *
 *  In general:
 *  [T0_0+T0_1+...+T0_N, T1_0+T1_1+...T1_N, T2_0+T2_1+...+T2_N]
 *
*/


template <typename T>
class AggregateTensors : public OpKernel {
private:
  int size;
  int rank;
  int BUF_SIZE;
  T * recv_buf;
  T * send_buf;
public:
  // constructor is thread-safe
  explicit AggregateTensors(OpKernelConstruction *ctx): OpKernel(ctx) {

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
// only allocate memory once TODO make size of allocation based
// on an attirbute
    OP_REQUIRES_OK(ctx, 
                   ctx->GetAttr("num_elements", &BUF_SIZE));

    OP_REQUIRES(ctx, BUF_SIZE > 0,
                   errors::InvalidArgument("Must have greater than 0 elements",
                                            BUF_SIZE));

    recv_buf = new T[BUF_SIZE];
    send_buf = new T[BUF_SIZE];



  }
  MPI_Datatype get_mpi_type(int32 type){
    return MPI_INT;
  }
  MPI_Datatype get_mpi_type(float type){
    return MPI_FLOAT;
  }
  MPI_Datatype get_mpi_type(double type){
    return MPI_DOUBLE;
  }
  /* 
   * The function executed on each invocation of
   * aggregate_tensors from python API
   *
   * The function sends, aggregates, and receives tensors in
   * three steps:
   *  
   *  1) Pack each tensor in the input list into a single
   *  contiguous buffer
   *  2) Communicate data using MPI_Allreduce with a SUM
   *  aggregation operation
   *  3) Unpacked a single contiguous buffer into the data
   *  buffers for each output tensor
   */

  void Compute(OpKernelContext *ctx) override {

    // convert tensor bufs into single contiguous buffer
    int num_inputs = ctx->num_inputs();
    int offset = 0;
    for (int i = 0; i < num_inputs; i ++){

      const Tensor& local_tensor = ctx->input(i);

      auto data = local_tensor.flat<T>();

      const T *  buffer = data.data();

      std::copy_n(buffer, data.size(), &send_buf[offset]);

      offset += data.size();

    }

    // do the communication
    MPI_Allreduce(send_buf, recv_buf, offset, get_mpi_type(send_buf[0]), MPI_SUM, MPI_COMM_WORLD);
  
    offset = 0;


    // convert message buffer to output tensors
    for (int i = 0; i < num_inputs; i++){

     const Tensor& local_tensor = ctx->input(i);
     Tensor * global_tensor = NULL;
     OP_REQUIRES_OK(ctx, ctx->allocate_output(i, local_tensor.shape(),
                                              &global_tensor));

     auto output = global_tensor->flat<T>();
     T * dst_buffer = output.data();

     std::copy_n(&recv_buf[offset], output.size(), dst_buffer);

     offset += output.size();

    }



  }
};


/*
 * GPU specialization of Aggregate Tensors
 *
 * Currently must be used with a CUDA-AWARE MPI implementation
 *
 * This differs from CPU implementaation because
 * it uses cudaMemcpy with DevicetoDevice specification instead
 * of standard stl lib function std::copy_n
 *
 * In addition send_buf and recv_buf are allocated on GPU
*/

template <typename T>
class AggregateTensorsGpu : public OpKernel {
private:

  int BUF_SIZE;

  T * recv_buf;
  T * send_buf;
  int size;
  int rank;
public:
  // constructor is thread-safe
  explicit AggregateTensorsGpu(OpKernelConstruction *ctx): OpKernel(ctx) {
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    OP_REQUIRES_OK(ctx, 
                   ctx->GetAttr("num_elements", &BUF_SIZE));

    OP_REQUIRES(ctx, BUF_SIZE > 0,
                   errors::InvalidArgument("Must have greater than 0 elements",
                                            BUF_SIZE));

   
    cudaMalloc(&recv_buf, BUF_SIZE * sizeof(T));
    cudaMalloc(&send_buf, BUF_SIZE * sizeof(T));

  }

  MPI_Datatype get_mpi_type(int32* type){
    return MPI_INT;
  }
  MPI_Datatype get_mpi_type(float* type){
    return MPI_FLOAT;
  }
  MPI_Datatype get_mpi_type(double* type){
    return MPI_DOUBLE;
  }
  /* 
   * The function executed on each invocation of
   * aggregate_tensors from python API
   *
   * The function sends, aggregates, and receives tensors in
   * three steps:
   *  
   *  1) Pack each tensor in the input list into a single
   *  contiguous buffer
   *  2) Communicate data using MPI_Allreduce with a SUM
   *  aggregation operation
   *  3) Unpacked a single contiguous buffer into the data
   *  buffers for each output tensor
   */

  void Compute(OpKernelContext *ctx) override {

    // convert tensor bufs into single contiguous buffer
    // do the communication
    int num_tensors = ctx->num_inputs();


    int offset = 0;
    for (int i = 0; i < num_tensors; i++){
      const Tensor& local_tensor = ctx->input(i);
      auto input_data = local_tensor.flat<T>();
      int size= input_data.size();
      const T * buffer = input_data.data();


      cudaMemcpy(&send_buf[offset], buffer, sizeof(T) *
          size, cudaMemcpyDeviceToDevice);

      offset += size;
    }

    // do the communication
    MPI_Allreduce(send_buf, recv_buf, offset, get_mpi_type(send_buf), MPI_SUM, MPI_COMM_WORLD);
  
    offset = 0;

    // convert message buffer to output tensors
    for (int i = 0; i < num_tensors; i++){
      Tensor * global_tensor = NULL;
      const Tensor& local_tensor = ctx->input(i);
      OP_REQUIRES_OK(ctx, ctx->allocate_output(i, local_tensor.shape(),
                                              &global_tensor));

      T * buffer = global_tensor->flat<T>().data();
      int size = global_tensor->flat<T>().size();

      cudaMemcpy(buffer, &recv_buf[offset], sizeof(T) *
         size, cudaMemcpyDeviceToDevice); 

       offset += size;
    }
  }
};



#define REGISTER_KERNEL_CPU(type)                                       \
    REGISTER_KERNEL_BUILDER(                                          \
      Name("AggregateTensors").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      AggregateTensors<type>)

#define REGISTER_KERNEL_GPU(type)                                       \
    REGISTER_KERNEL_BUILDER(                                          \
      Name("AggregateTensors").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      AggregateTensorsGpu<type>)

REGISTER_KERNEL_CPU(float);
REGISTER_KERNEL_CPU(double);
REGISTER_KERNEL_CPU(int32);

REGISTER_KERNEL_GPU(float);
REGISTER_KERNEL_GPU(double);
REGISTER_KERNEL_GPU(int32);
#undef REGISTER_KERNEL_CPU
#undef REGISTER_KERNEL_GPU

/*
 * Given a tensor returns a list of all other tensor
 * copies in MPI_COMM_WORLD
 *
 * Example:
 *  T0 on process 0
 *  T1 on process 1
 *  ...
 *  TN on process N
 *  
 *  [T0, T1, ..., TN]
 *
 *
 *
*/

template <typename T>
class GatherTensors : public OpKernel {
private:
  int size;
  int rank;
  int BUF_SIZE;
  int check = 0;
  int comp = 0;
  T * recv_buf;
  T * send_buf;
  int mpi_type;
public:
  explicit GatherTensors(OpKernelConstruction *ctx): OpKernel(ctx) {
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    OP_REQUIRES_OK(ctx, 
                   ctx->GetAttr("num_elements", &BUF_SIZE));

    OP_REQUIRES(ctx, BUF_SIZE > 0,
                   errors::InvalidArgument("Must have greater than 0 elements",
                                            BUF_SIZE));

    recv_buf = new T[BUF_SIZE * size];
    send_buf = new T[BUF_SIZE];


  }

  MPI_Datatype get_mpi_type(int32 type){
    return MPI_INT;
  }
  MPI_Datatype get_mpi_type(float type){
    return MPI_FLOAT;
  }
  MPI_Datatype get_mpi_type(double type){
    return MPI_DOUBLE;
  }

  /* 
   * The function executed on each invocation of
   * aggregate_tensors from python API
   *
   * The function sends, aggregates, and receives tensors in
   * three steps:
   *  
   *  1) Pack each tensor in the input list into a single
   *  contiguous buffer
   *  2) Communicate data using MPI_Allreduce with a SUM
   *  aggregation operation
   *  3) Unpacked a single contiguous buffer into the data
   *  
   *  */

  void Compute(OpKernelContext *ctx) override {
    // get the input list of tensors
    const Tensor& local_tensor = ctx->input(0);

    // offset into the send buffer to start next copy

    //buffer fill step of 1) begins; loop through input tensors


    auto data = local_tensor.flat<T>();

    const T *  buffer = data.data();




    MPI_Datatype dt = get_mpi_type(send_buf[0]);
    // do the communication using MPI_Allreduce
    MPI_Allgather(buffer, data.size(), dt, recv_buf, data.size(), dt, MPI_COMM_WORLD);
  
    // reset offset to use for unpacking recv_buf
    int offset = 0;

    // convert message buffer to output tensors
    for (int i = 0; i < size; i++){

     const Tensor& local_tensor = ctx->input(0);
     Tensor * global_tensor = NULL;

     // allocate output with same shape as its corresponding
     // input
     OP_REQUIRES_OK(ctx, ctx->allocate_output(i, local_tensor.shape(),
                                              &global_tensor));

     auto output = global_tensor->flat<T>();
     T * dst_buffer = output.data();

     std::copy_n(&recv_buf[offset], output.size(), dst_buffer);

     offset += output.size();

    }


  }
};


template <typename T>
class GatherTensorsGpu : public OpKernel {
private:
  int size;
  int rank;
  int BUF_SIZE;
  int check = 0;
  int comp = 0;
  T * recv_buf;
  T * send_buf;
  int mpi_type;
public:
  explicit GatherTensorsGpu(OpKernelConstruction *ctx): OpKernel(ctx) {
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    OP_REQUIRES_OK(ctx, 
                   ctx->GetAttr("num_elements", &BUF_SIZE));

    OP_REQUIRES(ctx, BUF_SIZE > 0,
                   errors::InvalidArgument("Must have greater than 0 elements",
                                            BUF_SIZE));

    cudaMalloc(&recv_buf, BUF_SIZE * size * sizeof(T));
    cudaMalloc(&send_buf, BUF_SIZE *  sizeof(T));


  }

  MPI_Datatype get_mpi_type(int32* type){
    return MPI_INT;
  }
  MPI_Datatype get_mpi_type(float* type){
    return MPI_FLOAT;
  }
  MPI_Datatype get_mpi_type(double* type){
    return MPI_DOUBLE;
  }

  /* 
   * The function executed on each invocation of
   * aggregate_tensors from python API
   *
   * The function sends, aggregates, and receives tensors in
   * three steps:
   *  
   *  1) Pack each tensor in the input list into a single
   *  contiguous buffer
   *  2) Communicate data using MPI_Allreduce with a SUM
   *  aggregation operation
   *  3) Unpacked a single contiguous buffer into the data
   *  buffers for each output tensor
   */

  void Compute(OpKernelContext *ctx) override {
    // get the input list of tensors
    const Tensor& local_tensor = ctx->input(0);

    // offset into the send buffer to start next copy

    //buffer fill step of 1) begins; loop through input tensors


    auto data = local_tensor.flat<T>();

    const T *  buffer = data.data();




    MPI_Datatype dt = get_mpi_type(send_buf);
    // do the communication using MPI_Allreduce
    MPI_Allgather(buffer, data.size(), dt, recv_buf, data.size(), dt, MPI_COMM_WORLD);
  
    // reset offset to use for unpacking recv_buf
    int offset = 0;

    // convert message buffer to output tensors
    for (int i = 0; i < size; i++){

     const Tensor& local_tensor = ctx->input(0);
     Tensor * global_tensor = NULL;

     // allocate output with same shape as its corresponding
     // input
     OP_REQUIRES_OK(ctx, ctx->allocate_output(i, local_tensor.shape(),
                                              &global_tensor));

     auto output = global_tensor->flat<T>();
     T * dst_buffer = output.data();
     

     cudaMemcpy(dst_buffer, &recv_buf[offset], sizeof(T) *
         output.size(), cudaMemcpyDeviceToDevice); 


     offset += output.size();

    }


  }
};


#define REGISTER_KERNEL_CPU(type)                                       \
    REGISTER_KERNEL_BUILDER(                                          \
      Name("GatherTensors").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      GatherTensors<type>)
#define REGISTER_KERNEL_GPU(type)                                       \
    REGISTER_KERNEL_BUILDER(                                          \
      Name("GatherTensors").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      GatherTensorsGpu<type>)


REGISTER_KERNEL_CPU(float);
REGISTER_KERNEL_CPU(double);
REGISTER_KERNEL_CPU(int32);

REGISTER_KERNEL_GPU(float);
REGISTER_KERNEL_GPU(double);
REGISTER_KERNEL_GPU(int32);


#undef REGISTER_KERNEL_CPU
#undef REGISTER_KERNEL_GPU
