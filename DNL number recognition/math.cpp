#include "math.h"
#include <cmath>
#include <algorithm>

namespace math {

	std::vector<double> matVecMultiply(const std::vector<std::vector<double>>& M, const std::vector<double>& v) {
		const auto rows = M.size();
		const auto cols = (rows > 0) ? M[0].size() : 0;

		std::vector<double> res(rows, 0.0);
		for (std::size_t i = 0; i < rows; i++)
			for (std::size_t j = 0; j < cols; j++)
				res[i] += M[i][j] * v[j];

		return res;
	}

	void addBias(std::vector<double>& output, const std::vector<double>& bias) {
		for (std::size_t i = 0; i < output.size(); i++)
			output[i] += bias[i];
	}

	void reluInPlace(std::vector<double>& v) {
		for (auto& val : v)
			if (val < 0.0)
				val = 0.0;
	}

	std::vector<double> relu(const std::vector<double>& v) {
		std::vector<double> res(v.size());

		for (std::size_t i = 0; i < v.size(); i++)
			res[i] = (v[i] < 0.0) ? 0.0 : v[i];

		return res;
	}

	std::vector<double> sigmoid(const std::vector<double>& v) {
		std::vector<double> res(v.size());

		for (std::size_t i = 0; i < v.size(); i++)
			res[i] = 1.0 / (1.0 + std::exp(-v[i]));

		return res;
	}

	void sigmoidInPlace(std::vector<double>& v) {
		for (auto& val : v)
			val = 1.0 / (1.0 + std::exp(-val));
	}

	std::vector<double> softmax(const std::vector<double>& logits) {
		std::vector<double> result(logits.size());
		double maxVal = *std::max_element(logits.begin(), logits.end());

		double sumExp = 0.0;
		for (auto val : logits)
			sumExp += std::exp(val - maxVal);

		for (std::size_t i = 0; i < logits.size(); ++i)
			result[i] = std::exp(logits[i] - maxVal) / sumExp;

		return result;
	}

	double crossEntropy(const std::vector<double>& prediction, const std::vector<double>& target) {
		double loss = 0.0;
		for (std::size_t i = 0; i < prediction.size(); ++i) {
			double p = std::max(prediction[i], 1e-15); // prevent log(0)
			loss -= target[i] * std::log(p);
		}
		return loss;
	}

	double meanSquaredError(const std::vector<double>& prediction, const std::vector<double>& target) {
		double sum = 0.0;
		for (std::size_t i = 0; i < prediction.size(); ++i) {
			double diff = prediction[i] - target[i];
			sum += diff * diff;
		}
		return sum / prediction.size();
	}
}
