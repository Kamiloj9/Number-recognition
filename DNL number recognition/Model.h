#pragma once
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <fstream>
#include <string>

#include "utils.h"
#include "math.h"

class Model
{
public:
	Model(std::size_t inputSize, std::size_t hiddenSize, std::size_t outputSize, double lr = 0.01);

    /*
    
     Forward pass for a single sample
     Returns the output layer (softmax probabilities)
     Also stores intermediate results needed for backprop (z1, hidden)
    
    */
    std::vector<double> forward(const std::vector<double>& input);

    /*
    
    Backprop for a single sample
    input: original input vector
    output: forward pass result (softmax probabilities)
    target: one-hot vector for the correct label

    */
    void backprop(const std::vector<double>& input,
        const std::vector<double>& output,
        const std::vector<double>& target);

    /*
        
    Train loop

    */
    void train(const std::vector<std::vector<double>>& trainInputs,
        const std::vector<int>& trainLabels,
        int epochs = 5);

    /*
    
    Predict a label for a single input
    returns the class index with max probability
    
    */
    int predict(const std::vector<double>& input);

    /*
    
    Save the model to a binary file
    
    */
    void saveModel(const std::string& filename) const;

    /*
    
    Load the model from a binary file (overwrites current)

    */
    void loadModel(const std::string& filename);
private:
    // Dimensions
    std::size_t inputSize_t;
    std::size_t hiddenSize_t;
    std::size_t outputSize_t;

    // Parameters
    std::vector<std::vector<double>> w1_t; // [hiddenSize_][inputSize_]
    std::vector<double> b1_t;              // [hiddenSize_]

    std::vector<std::vector<double>> w2_t; // [outputSize_][hiddenSize_]
    std::vector<double> b2_t;              // [outputSize_]

    // Intermediate results (for backprop)
    std::vector<double> z1_t;     // pre-activation hidden
    std::vector<double> hidden_t; // post-activation hidden
    std::vector<double> z2_t;     // pre-softmax

    double learningRate_t;
};

