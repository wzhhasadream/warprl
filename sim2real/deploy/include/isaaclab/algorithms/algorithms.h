// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "onnxruntime_cxx_api.h"
#include <iostream>
#include <mutex>

namespace isaaclab
{

class Algorithms
{
public:
    virtual std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs) = 0;

    std::vector<float> get_action()
    {
        std::lock_guard<std::mutex> lock(act_mtx_);
        return action;
    }
    
    std::vector<float> action;
protected:
    std::mutex act_mtx_;
};

class OrtRunner : public Algorithms
{
public:
    OrtRunner(std::string model_path)
    {
        // Init Model
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "onnx_model");
        session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);

        session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);

        for (size_t i = 0; i < session->GetInputCount(); ++i) {
            Ort::TypeInfo input_type = session->GetInputTypeInfo(i);
            input_shapes.push_back(input_type.GetTensorTypeAndShapeInfo().GetShape());
            auto input_name = session->GetInputNameAllocated(i, allocator);
            input_names.push_back(input_name.release());
        }

        for (size_t i = 0; i < input_names.size(); ++i) {
            spdlog::info("ONNX input[{}]: {}", i, input_names[i]);
        }

        for (const auto& shape : input_shapes) {
            size_t size = 1;
            for (const auto& dim : shape) {
                // ONNX uses a negative value for a symbolic/dynamic batch
                // dimension. Deployment runs one robot, so resolve it to 1.
                size *= static_cast<size_t>(dim > 0 ? dim : 1);
            }
            input_sizes.push_back(size);
        }

        for (auto& shape : input_shapes) {
            for (auto& dim : shape) {
                if (dim <= 0) dim = 1;
            }
        }

        // Get output shape
        Ort::TypeInfo output_type = session->GetOutputTypeInfo(0);
        output_shape = output_type.GetTensorTypeAndShapeInfo().GetShape();
        auto output_name = session->GetOutputNameAllocated(0, allocator);
        output_names.push_back(output_name.release());

        action.resize(output_shape[1]);
    }

    std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs)
    {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        // Make sure all input names are in obs.  JAX exporters commonly name
        // their sole input var_0 while the deployment observation group is
        // named obs.  For a single-input/single-observation policy these names
        // are aliases; resolve by position without changing tensor contents.
        for (const auto& name : input_names) {
            if (obs.find(name) == obs.end() &&
                !(input_names.size() == 1 && obs.size() == 1)) {
                throw std::runtime_error("Input name " + std::string(name) + " not found in observations.");
            }
        }

        // Create input tensors
        std::vector<Ort::Value> input_tensors;
        for(int i(0); i<input_names.size(); ++i)
        {
            const std::string name_str(input_names[i]);
            auto input_it = obs.find(name_str);
            if (input_it == obs.end() && input_names.size() == 1 && obs.size() == 1) {
                input_it = obs.begin();
            }
            auto& input_data = input_it->second;
            if (input_data.size() != input_sizes[i]) {
                throw std::runtime_error(
                    "ONNX input size mismatch for " + name_str +
                    ": expected " + std::to_string(input_sizes[i]) +
                    ", got " + std::to_string(input_data.size()));
            }
            auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_data.data(), input_sizes[i], input_shapes[i].data(), input_shapes[i].size());
            input_tensors.push_back(std::move(input_tensor));
        }

        // Run the model
        auto output_tensor = session->Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(), input_tensors.size(), output_names.data(), 1);

        // Copy output data
        auto floatarr = output_tensor.front().GetTensorMutableData<float>();
        std::lock_guard<std::mutex> lock(act_mtx_);
        std::memcpy(action.data(), floatarr, output_shape[1] * sizeof(float));
        return action;
    }

private:
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<const char*> input_names;
    std::vector<const char*> output_names;

    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<int64_t> input_sizes;
    std::vector<int64_t> output_shape;
};
};
