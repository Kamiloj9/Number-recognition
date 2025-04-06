#pragma once
#include <vector>

namespace math {
	std::vector<double> matVecMultiply(const std::vector<std::vector<double>>& M, const std::vector<double>& v);
	void addBias(std::vector<double>& output, const std::vector<double>& bias);
	void reluInPlace(std::vector<double>& v);
	std::vector<double> relu(const std::vector<double>& v);
	std::vector<double> sigmoid(const std::vector<double>& v);
	void sigmoidInPlace(std::vector<double>& v);
	std::vector<double> softmax(const std::vector<double>& logits);
	double crossEntropy(const std::vector<double>& prediction, const std::vector<double>& target);
	double meanSquaredError(const std::vector<double>& prediction, const std::vector<double>& target);
}
