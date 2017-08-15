licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

package(default_visibility = ["//tensorflow:__subpackages__"])

load("//tensorflow:tensorflow.bzl", "tf_custom_op_library")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_wrapper_py")
load("//tensorflow:tensorflow.bzl", "tf_gen_op_libs")
load("//tensorflow:tensorflow.bzl", "tf_py_test")
load("//tensorflow:tensorflow.bzl", "cuda_py_test")
load("//tensorflow:tensorflow.bzl", "tf_custom_op_py_library")

tf_custom_op_py_library(
  name = "collectives_py",
  dso = [
    ":mpi_coll.so"
  ],
)

tf_custom_op_library(
    name = "mpi_coll.so",
    srcs = [
        "aggregate_tensors_op.cpp",
    ],
)

tf_gen_op_libs([
  "mpi_coll",
])

cc_library(
  name = "all_ops",
  deps = [
    ":mpi_coll_op_lib",
  ],
)
