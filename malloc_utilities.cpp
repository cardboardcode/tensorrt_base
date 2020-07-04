#include "malloc_utilities.hpp"

std::unique_ptr<void, std::function<void(void*)>> cuda_malloc_from_dims(
    nvinfer1::Dims dims) {
  std::cout << "Enter cuda_malloc_from_dims function." << std::endl;
  size_t size = sizeof(float);
  for (int i = 0; i < dims.nbDims; i++) {
    size *= dims.d[i];
  }
  std::cout << "[Passing by]" << std::endl;
  // currently only supports float32
  std::unique_ptr<void, std::function<void(void*)>> ptr(
      device_malloc(size), [](void* ptr) -> void { device_deleter(ptr); });
  std::cout << "Exiting cuda_malloc_from_dims function." << std::endl;
  return ptr;
}

std::unique_ptr<void, std::function<void(void*)>> cuda_host_malloc_from_dims(
    nvinfer1::Dims dims) {
  size_t size = sizeof(float);
  for (int i = 0; i < dims.nbDims; i++) {
    size *= dims.d[i];
  }
  // currently only supports float32
  std::unique_ptr<void, std::function<void(void*)>> ptr(
      host_pinned_malloc(size),
      [](void* ptr) -> void { host_pinned_deleter(ptr); });
  return ptr;
}
