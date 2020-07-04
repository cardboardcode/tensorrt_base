#include "tensorrt_base.hpp"

/* Function Description: Constructor 1 of class TensorRTModule for onnx model files.
Inputs -
model_path -> file directory path to your .onnx model
batch_size ->
max_workspace_size -> The memory limit the user sets for your ML algorithm to use during operation.
-If the limit is breached, do not consider running the algorithm
Outputs -
NIL
*/
// This uses a member initializer list to assign values immediately to a class public or private attributes.
// The use of initializer list is for performance reason.
// https://www.geeksforgeeks.org/when-do-we-use-initializer-list-in-c/
TensorRTModule::TensorRTModule(std::string model_path, int batch_size,
                               int max_workspace_size)
    : model_path_(model_path),
      model_type_(tensorrt_common::getFileType(model_path)),
      batch_size_(batch_size),
      max_workspace_size_(max_workspace_size) {
  // TODO: DLA stuff as mentioned in commons, probably play with fp16

  std::cout << "Inside tensorrt_base.cpp>>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl;
  std::cout << "[model_path] = " << model_path_ << std::endl;
  std::cout << "[model_type] = " << model_type_ << std::endl;
  std::cout << "[batch_size] = " << batch_size_ << std::endl;
  std::cout << "[max_workspace_size] = " << max_workspace_size_ << std::endl;

  // Create the initial objects needed
  trt_builder_ =
      tensorrt_common::infer_object(nvinfer1::createInferBuilder(g_logger_));
  trt_builder_->setMaxBatchSize(batch_size_);
  trt_builder_->setMaxWorkspaceSize(max_workspace_size_);

  const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

  // trt_builder and trt_runtime
  trt_network_ = tensorrt_common::infer_object(trt_builder_->createNetworkV2(explicitBatch));
  trt_parser_ = tensorrt_common::infer_object(
      nvonnxparser::createParser(*trt_network_, g_logger_));
  trt_runtime_ =
      tensorrt_common::infer_object(nvinfer1::createInferRuntime(g_logger_));

  // load the onnx model
  if (strcmp(model_type_.c_str(), "onnx") == 0) {
    if (!trt_parser_->parseFromFile(model_path_.c_str(), verbosity_)){
      std::cout << "failed to parse onnx file" << std::endl;
    }
      std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << trt_parser_->getNbErrors() << std::endl;

      for (int i = 0; i < trt_parser_->getNbErrors(); ++i){
		      std::cout << trt_parser_->getError(i)->desc() << std::endl;
	    }
      // runtime_error_("failed to parse onnx file");

  }

  std::cout << "ONNX model file parsed successfully." << std::endl;

  // get some network details used in inference later
  n_inputs_ = trt_network_->getNbInputs();
  n_outputs_ = trt_network_->getNbOutputs();
  n_layers_ = trt_network_->getNbLayers();
  n_bindings_ = n_inputs_ + n_outputs_;
  input_dims_.reserve(n_inputs_);
  input_vols_.reserve(n_inputs_);
  output_dims_.reserve(n_outputs_);
  output_vols_.reserve(n_outputs_);
  engine_smart_mem_.reserve(n_bindings_);
  engine_mem_.reserve(n_bindings_);
  output_mem_.reserve(n_outputs_);
  output_mem_.clear();

  std::cout << "[n_bindings_] = " << n_bindings_ << std::endl;

  std::cout << "[After getting network details for inference later.]" << std::endl;

  // saves the dimensions for use later, also mallocs on host and device
  // Using n_inputs_, input_dims, trt_network, input_vols.

  for (int input_index = 0; input_index < n_inputs_; input_index++) {
    std::cout << "[1input_index] = " << input_index << std::endl;
    nvinfer1::Dims curr_dim =
        trt_network_->getInput(input_index)->getDimensions();
        std::cout << "[2input_index] = " << input_index << std::endl;
    input_dims_.push_back(curr_dim);
    std::cout << "[3input_index] = " << input_index << std::endl;
    input_vols_.push_back(tensorrt_common::volume(curr_dim));
    std::cout << "[4input_index] = " << input_index << std::endl;
    engine_smart_mem_.push_back(cuda_malloc_from_dims(curr_dim));
    std::cout << "[5input_index] = " << input_index << std::endl;
    engine_mem_.push_back(engine_smart_mem_.back().get());
    std::cout << "[6input_index] = " << input_index << std::endl;
  }

  std::cout << "[After 1st for loop.]" << std::endl;

  for (int output_index = 0; output_index < n_outputs_; output_index++) {
    nvinfer1::Dims curr_dim =
        trt_network_->getOutput(output_index)->getDimensions();
    output_dims_.push_back(curr_dim);
    output_vols_.push_back(tensorrt_common::volume(curr_dim));
    engine_smart_mem_.push_back(cuda_malloc_from_dims(curr_dim));
    engine_mem_.push_back(engine_smart_mem_.back().get());

    std::vector<float> curr_output_mem(
        static_cast<int>(tensorrt_common::volume(curr_dim)));
    output_mem_.push_back(curr_output_mem);
  }

  std::cout << "[After saving dimensions]" << std::endl;

  // get the final engines, inference contexts and cudaStream
  trt_engine_ = tensorrt_common::infer_object(
      trt_builder_->buildCudaEngine(*trt_network_));
  if (n_bindings_ != trt_engine_->getNbBindings())
    runtime_error_("number of bindings inconsistent");
  trt_context_ =
      tensorrt_common::infer_object(trt_engine_->createExecutionContext());
  CHECK(cudaStreamCreate(&stream_));

  std::cout << "[Exiting constructor function]" << std::endl;

}

/*Function Description: Constructor 2 of class TensorRTModule for serialized engine file.
Inputs -
model_path -> file directory path to your .onnx model
batch_size -> Number of sample to propogate through network before updating weights
Outputs -
NIL
*/
TensorRTModule::TensorRTModule(std::string model_path, int batch_size)
    : model_path_(model_path),
      model_type_(tensorrt_common::getFileType(model_path)),
      batch_size_(batch_size) {
  trt_runtime_ =
      tensorrt_common::infer_object(nvinfer1::createInferRuntime(g_logger_));

  std::stringstream gie_model_stream; // Define a string as a stream object
  gie_model_stream.seekg(0, gie_model_stream.beg); // Set input position at the beginning of the stream buffer.
  std::ifstream cache(model_path_);
  gie_model_stream << cache.rdbuf();
  cache.close();

  gie_model_stream.seekg(0, std::ios::end);
  const int model_size = gie_model_stream.tellg();
  gie_model_stream.seekg(0, std::ios::beg);

  std::unique_ptr<char> model_mem(static_cast<char*>(malloc(model_size)));
  gie_model_stream.read(static_cast<char*>(model_mem.get()), model_size);
  trt_engine_ =
      tensorrt_common::infer_object(trt_runtime_->deserializeCudaEngine(
          model_mem.get(), (std::size_t)model_size, NULL));
  trt_context_ =
      tensorrt_common::infer_object(trt_engine_->createExecutionContext());
  CHECK(cudaStreamCreate(&stream_));

  // get some network details used in inference later, using engine
  n_layers_ = trt_engine_->getNbLayers();
  n_bindings_ = trt_engine_->getNbBindings();
  engine_smart_mem_.reserve(n_bindings_);
  engine_mem_.reserve(n_bindings_);

  n_inputs_ = 0;
  n_outputs_ = 0;
  input_dims_.clear();
  input_vols_.clear();
  output_dims_.clear();
  output_vols_.clear();
  output_mem_.clear();

  for (int binding = 0; binding < n_bindings_; binding++) {
    nvinfer1::Dims curr_dim = trt_engine_->getBindingDimensions(binding);
    if (trt_engine_->bindingIsInput(binding)) {
      n_inputs_++;
      input_dims_.push_back(curr_dim);
      input_vols_.push_back(tensorrt_common::volume(curr_dim));
    } else {
      n_outputs_++;
      output_dims_.push_back(curr_dim);
      output_vols_.push_back(tensorrt_common::volume(curr_dim));

      std::vector<float> curr_output_mem(
          static_cast<int>(tensorrt_common::volume(curr_dim)));
      output_mem_.push_back(curr_output_mem);
    }
    engine_smart_mem_.push_back(cuda_malloc_from_dims(curr_dim));
    engine_mem_.push_back(engine_smart_mem_.back().get());
  }
}

//Function Description: Deconstructor of class TensorRTModule
TensorRTModule::~TensorRTModule() { cudaStreamDestroy(stream_); }

/*Function Description: Mutator 1 of class TensorRTModule
Inputs -
input (Takes in a float-type vector of an input image)
Outputs -
Boolean true or false
*/
bool TensorRTModule::inference(const std::vector<std::vector<float>>& input) {

  // safety: must be same number of inputs
  if (static_cast<int>(input.size()) != n_inputs_)
    runtime_error_("number of inputs incorrect");

  // copy from inputs to device asynchronously
  for (int i = 0; i < n_inputs_; i++) {
    if (static_cast<int>(input[i].size() != input_vols_[i]))
      runtime_error_("input [" + std::to_string(i) + "] size incorrect");
    CHECK(cudaMemcpyAsync(engine_mem_[i], &input[i][0],
                          batch_size_ * input_vols_[i] * sizeof(float),
                          cudaMemcpyHostToDevice, stream_));
  }
  // Waits for the copying of input to device to complete.
  // Think of this as a cuda-specific wait function
  cudaStreamSynchronize(stream_);

  // inference
  this->trt_context_->enqueueV2(engine_mem_.data(), stream_,
                                                          nullptr);
  cudaStreamSynchronize(stream_);

  // copy from device to output asynchronously
  // engine memory pointers are concatenated as [input | output]
  for (int i = 0; i < n_outputs_; i++) {
    CHECK(cudaMemcpyAsync(&output_mem_[i][0], engine_mem_[i + n_inputs_],
                          batch_size_ * output_vols_[i] * sizeof(float),
                          cudaMemcpyDeviceToHost, stream_));
  }
  return true;
}

const std::vector<float>& TensorRTModule::get_output(int output_index) {
  if (output_index < 0 || output_index >= n_outputs_)
    runtime_error_("output index out of range, requesting " +
                   std::to_string(output_index) + " when there are " +
                   std::to_string(n_outputs_) + " outputs");
  return output_mem_[output_index];
}

/*Function Description: Accessor - Saves the .engine file generated from a parsed ONNX model.
Inputs -
the ONNX model dimensions

Outputs -
Print to terminal the ONNX dimensions.
*/
void TensorRTModule::save_engine(std::string engine_path) {
  if (!trt_engine_) runtime_error_("Engine not created yet, unable to save.");

  std::shared_ptr<nvinfer1::IHostMemory> serialized_model =
      tensorrt_common::infer_object(trt_engine_->serialize());
  std::ofstream ofs(engine_path, std::ios::out | std::ios::binary);
  ofs.write((char*)(serialized_model->data()), serialized_model->size());
  ofs.close();
}

// UNUSED
/*Function Description: Utility funciton to print out the dimensions of an ONNX model
Inputs -
the ONNX model dimensions

Outputs -
Print to terminal the ONNX dimensions.
*/
void print_dims(nvinfer1::Dims dimensions) {
  for (int i = 0; i < dimensions.nbDims; i++) {
    std::cout << dimensions.d[i] << " ";
  }
  std::cout << std::endl;
}
