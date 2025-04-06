#pragma once
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace utils {
	double randomWeight(double range = 0.01);
	double getPixel(const std::vector<double>& img, int row, int col);
	void setPixel(std::vector<double>& img, int row, int col, double value);
	double sampleNearest(const std::vector<double>& img, float row, float col);

	/*
	
	 Rotate + scale a 28x28 image.
     angleDegrees: rotation angle in degrees
     scaleFactor:  e.g., 0.8 .. 1.2
     fillValue:    e.g., 0.0 for background
	
	*/
	std::vector<double> augmentImage(const std::vector<double>& input,
		double angleDegrees,
		double scaleFactor,
		int translateX,
		int translateY,
		double fillValue = 0.0);
}