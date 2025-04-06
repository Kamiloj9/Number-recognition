#include "Model.h"

Model::Model(std::size_t inputSize, std::size_t hiddenSize, std::size_t outputSize, double lr) {
	inputSize_t = inputSize;
	hiddenSize_t = hiddenSize;
	outputSize_t = outputSize;
	learningRate_t = lr;

	w1_t.resize(hiddenSize_t, std::vector<double>(inputSize_t));
	b1_t.resize(hiddenSize_t, 0.0);

    // Initialize W1, B1
    w1_t.resize(hiddenSize_t, std::vector<double>(inputSize_t));
    b1_t.resize(hiddenSize_t, 0.0);
    for (std::size_t i = 0; i < hiddenSize_t; ++i) {
        for (std::size_t j = 0; j < inputSize_t; ++j) {
            w1_t[i][j] = utils::randomWeight(0.01);
        }
        b1_t[i] = 0.0;
    }

    // Initialize W2, B2
    w2_t.resize(outputSize_t, std::vector<double>(hiddenSize_t));
    b2_t.resize(outputSize_t, 0.0);
    for (std::size_t i = 0; i < outputSize_t; ++i) {
        for (std::size_t j = 0; j < hiddenSize_t; ++j) {
            w2_t[i][j] = utils::randomWeight(0.01);
        }
        b2_t[i] = 0.0;
    }
}

std::vector<double> Model::forward(const std::vector<double>& input)
{
    // 1) hidden pre-activation: z1 = W1 * input + b1
    z1_t = math::matVecMultiply(w1_t, input);
    math::addBias(z1_t, b1_t);

    // 2) hidden activation = ReLU(z1)
    hidden_t = z1_t;  // copy
    math::reluInPlace(hidden_t);

    // 3) output pre-activation: z2 = W2 * hidden + b2
    z2_t = math::matVecMultiply(w2_t, hidden_t);
    math::addBias(z2_t, b2_t);

    // 4) output activation = softmax(z2)
    return math::softmax(z2_t);
}

void Model::backprop(const std::vector<double>& input, const std::vector<double>& output, const std::vector<double>& target)
{
    // We know that for cross-entropy & softmax:
        //   dL/d(z2) = (output - target)
    std::vector<double> dZ2(outputSize_t);
    for (std::size_t i = 0; i < outputSize_t; ++i) {
        dZ2[i] = output[i] - target[i];
    }

    // hidden was ReLU(z1).
    // We need dZ1 = (W2^T * dZ2) * ReLU'(z1).
    std::vector<double> dZ1(hiddenSize_t, 0.0);

    // For each hidden neuron j:
    for (std::size_t j = 0; j < hiddenSize_t; ++j) {
        double grad = 0.0;
        for (std::size_t i = 0; i < outputSize_t; ++i) {
            grad += w2_t[i][j] * dZ2[i];
        }
        // derivative of ReLU
        if (z1_t[j] > 0.0) {
            dZ1[j] = grad;
        }
        else {
            dZ1[j] = 0.0;
        }
    }

    // Now update W2, B2
    // w2_[i][j] -= learningRate * dZ2[i] * hidden_[j]
    // b2_[i]    -= learningRate * dZ2[i]
    for (std::size_t i = 0; i < outputSize_t; ++i) {
        for (std::size_t j = 0; j < hiddenSize_t; ++j) {
            w2_t[i][j] -= learningRate_t * dZ2[i] * hidden_t[j];
        }
        b2_t[i] -= learningRate_t * dZ2[i];
    }

    // Update W1, B1
    // w1_[j][k] -= learningRate * dZ1[j] * input[k]
    // b1_[j]    -= learningRate * dZ1[j]
    for (std::size_t j = 0; j < hiddenSize_t; ++j) {
        for (std::size_t k = 0; k < inputSize_t; ++k) {
            w1_t[j][k] -= learningRate_t * dZ1[j] * input[k];
        }
        b1_t[j] -= learningRate_t * dZ1[j];
    }
}

void Model::train(const std::vector<std::vector<double>>& trainInputs, const std::vector<int>& trainLabels, int epochs)
{
    if (trainInputs.size() != trainLabels.size()) {
        throw std::runtime_error("Mismatch in trainInputs and trainLabels sizes.");
    }

    std::size_t numSamples = trainInputs.size();

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double totalLoss = 0.0;

        for (std::size_t i = 0; i < numSamples; ++i) {
            // Forward
            auto out = forward(trainInputs[i]);

            // Build one-hot target
            std::vector<double> target(outputSize_t, 0.0);
            target[trainLabels[i]] = 1.0;

            // Calculate loss
            double loss = math::crossEntropy(out, target);
            totalLoss += loss;

            // Backprop
            backprop(trainInputs[i], out, target);
        }

        std::cout << "Epoch " << epoch
            << " - avg loss = " << (totalLoss / numSamples)
            << std::endl;
    }
}

int Model::predict(const std::vector<double>& input)
{
    auto out = forward(input);
    return static_cast<int>(
        std::distance(out.begin(), std::max_element(out.begin(), out.end())));
}

void Model::saveModel(const std::string& filename) const
{
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) {
        throw std::runtime_error("Could not open file for writing: " + filename);
    }

    // 1) Write network dimensions so we can check or reconstruct on load
    ofs.write(reinterpret_cast<const char*>(&inputSize_t), sizeof(inputSize_t));
    ofs.write(reinterpret_cast<const char*>(&hiddenSize_t), sizeof(hiddenSize_t));
    ofs.write(reinterpret_cast<const char*>(&outputSize_t), sizeof(outputSize_t));

    // 2) Write w1_ (hiddenSize_ rows, each row has inputSize_ doubles)
    for (std::size_t i = 0; i < hiddenSize_t; ++i) {
        ofs.write(reinterpret_cast<const char*>(w1_t[i].data()),
            inputSize_t * sizeof(double));
    }

    // 3) Write b1_
    ofs.write(reinterpret_cast<const char*>(b1_t.data()),
        hiddenSize_t * sizeof(double));

    // 4) Write w2_ (outputSize_ rows, each row has hiddenSize_ doubles)
    for (std::size_t i = 0; i < outputSize_t; ++i) {
        ofs.write(reinterpret_cast<const char*>(w2_t[i].data()),
            hiddenSize_t * sizeof(double));
    }

    // 5) Write b2_
    ofs.write(reinterpret_cast<const char*>(b2_t.data()),
        outputSize_t * sizeof(double));

    ofs.close();
}

void Model::loadModel(const std::string& filename)
{
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Could not open file for reading: " + filename);
    }

    // 1) Read dimensions (inputSize, hiddenSize, outputSize)
    std::size_t inSize, hidSize, outSize;
    ifs.read(reinterpret_cast<char*>(&inSize), sizeof(inSize));
    ifs.read(reinterpret_cast<char*>(&hidSize), sizeof(hidSize));
    ifs.read(reinterpret_cast<char*>(&outSize), sizeof(outSize));

    // 2) Check if they match our current model
    //    (Alternatively, you could re-allocate if you want the Model
    //     to adopt whatever sizes are stored in the file.)
    if (inSize != inputSize_t || hidSize != hiddenSize_t || outSize != outputSize_t) {
        throw std::runtime_error(
            "Dimension mismatch in loaded model: file("
            + std::to_string(inSize) + ","
            + std::to_string(hidSize) + ","
            + std::to_string(outSize) + ") != current("
            + std::to_string(inputSize_t) + ","
            + std::to_string(hiddenSize_t) + ","
            + std::to_string(outputSize_t) + ")"
        );
    }

    // 3) Read w1_
    for (std::size_t i = 0; i < hiddenSize_t; ++i) {
        ifs.read(reinterpret_cast<char*>(w1_t[i].data()),
            inputSize_t * sizeof(double));
    }

    // 4) Read b1_
    ifs.read(reinterpret_cast<char*>(b1_t.data()),
        hiddenSize_t * sizeof(double));

    // 5) Read w2_
    for (std::size_t i = 0; i < outputSize_t; ++i) {
        ifs.read(reinterpret_cast<char*>(w2_t[i].data()),
            hiddenSize_t * sizeof(double));
    }

    // 6) Read b2_
    ifs.read(reinterpret_cast<char*>(b2_t.data()),
        outputSize_t * sizeof(double));

    ifs.close();
}
