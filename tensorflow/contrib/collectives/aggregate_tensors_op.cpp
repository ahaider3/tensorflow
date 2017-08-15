#include "aggregate_tensors_op.h"

/*
 * Tensorflow Op definitions for CPU only
 * 
*/

/*
 * Initializes MPI environment
 * This function should be called once per process or else MPI
 * will return an error. 
 * Note* we must call dlopen since we are initializing MPI from
 * a local scope*
 * MPI_THREAD_MULTIPLE is not needed for current implementation
 * TODO specify lower level thread support for performance
 * reasons
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

          initialized=1;
        }
        
        
    }

};
REGISTER_KERNEL_BUILDER(Name("AggregateTensorsInit").Device(DEVICE_CPU), AggregateTensorsInit);

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
        t_s.AddDim(3);
        Tensor * global_tensor = NULL;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, t_s,
                                              &global_tensor));

        auto output_data = global_tensor->flat<float>();
        int size;
        int rank;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm shcomm;

        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,0,
                        MPI_INFO_NULL, &shcomm);

        int shmrank;

        MPI_Comm_rank(shcomm, &shmrank);

        output_data(0) = size;
        output_data(1) = rank;
        output_data(2) = shmrank;

        
    }

};
REGISTER_KERNEL_BUILDER(Name("AggregateTensorsGetInfo").Device(DEVICE_CPU), AggregateTensorsGetInfo);

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
 *
*/
template <typename T>
class AggregateTensors : public OpKernel {
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
  explicit AggregateTensors(OpKernelConstruction *ctx): OpKernel(ctx) {
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
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
    // get the input list of tensors
    int num_inputs = ctx->num_inputs();

    // offset into the send buffer to start next copy
    int offset = 0;

    //buffer fill step of 1) begins; loop through input tensors
    for (int i = 0; i < num_inputs; i ++){

      const Tensor& local_tensor = ctx->input(i);

      auto data = local_tensor.flat<T>();

      const T *  buffer = data.data();

      std::copy_n(buffer, data.size(), &send_buf[offset]);

      // increment offset to next available position in buf
      offset += data.size();

    }

    // do the communication using MPI_Allreduce
    MPI_Allreduce(send_buf, recv_buf, offset, get_mpi_type(send_buf[0]), MPI_SUM, MPI_COMM_WORLD);
  
    // reset offset to use for unpacking recv_buf
    offset = 0;

    // convert message buffer to output tensors
    for (int i = 0; i < num_inputs; i++){

     const Tensor& local_tensor = ctx->input(i);
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

#define REGISTER_KERNEL(type)                                       \
    REGISTER_KERNEL_BUILDER(                                          \
      Name("AggregateTensors").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      AggregateTensors<type>)

REGISTER_KERNEL(float);
REGISTER_KERNEL(double);
REGISTER_KERNEL(int32);

#undef REGISTER_KERNEL


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
   *  buffers for each output tensor
   */

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
#define REGISTER_KERNEL(type)                                       \
    REGISTER_KERNEL_BUILDER(                                          \
      Name("GatherTensors").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      GatherTensors<type>)

REGISTER_KERNEL(float);
REGISTER_KERNEL(double);
REGISTER_KERNEL(int32);

#undef REGISTER_KERNEL

