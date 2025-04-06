#include "utils.h"

double utils::randomWeight(double range)
{
    static std::mt19937 rng{ std::random_device{}() };
    std::uniform_real_distribution<double> dist(-range, range);
    return dist(rng);
}

double utils::getPixel(const std::vector<double>& img, int row, int col)
{
    return img[row * 28 + col];
}

void utils::setPixel(std::vector<double>& img, int row, int col, double value)
{
    img[row * 28 + col] = value;
}

double utils::sampleNearest(const std::vector<double>& img, float row, float col)
{
    int r = static_cast<int>(std::round(row));
    int c = static_cast<int>(std::round(col));

    // If out of bounds, return an invalid marker or handle externally
    if (r < 0 || r >= 28 || c < 0 || c >= 28) {
        // We'll handle out-of-bounds in the main loop by fillValue
        return -1.0;
    }

    return getPixel(img, r, c);
}

std::vector<double> utils::augmentImage(const std::vector<double>& input,
    double angleDegrees,
    double scaleFactor,
    int translateX,
    int translateY,
    double fillValue)
{
    // We'll produce a new 28x28
    std::vector<double> output(28 * 28, fillValue);

    // Convert angle to radians, but note for inverse we can just use -angle
    static const auto PI = 3.14159265358979323846;
    double angleRad = angleDegrees * PI / 180.0;
    double cosA = std::cos(-angleRad);
    double sinA = std::sin(-angleRad);

    // Center of image
    double cx = 13.5;
    double cy = 13.5;

    // For each output pixel (r_out, c_out), we do an inverse transform:
    // 1) Translate by (-translateX, -translateY)
    // 2) Move center to (0,0)
    // 3) Scale by (1/scaleFactor)
    // 4) Rotate by (-angle)
    // 5) Move center back
    for (int r_out = 0; r_out < 28; ++r_out) {
        for (int c_out = 0; c_out < 28; ++c_out) {
            // Shift by translateX, translateY
            double x = c_out - translateX;
            double y = r_out - translateY;

            // Move center to 0,0
            x -= cx;
            y -= cy;

            // Apply scale
            x /= scaleFactor;
            y /= scaleFactor;

            // Apply rotation (inverse)
            double x_rot = x * cosA - y * sinA;
            double y_rot = x * sinA + y * cosA;

            // Move center back
            x_rot += cx;
            y_rot += cy;

            // Sample from input
            double val = sampleNearest(input, y_rot, x_rot); // note: row ~ y, col ~ x

            // If out of bounds, we do fillValue
            if (val < 0.0) {
                output[r_out * 28 + c_out] = fillValue;
            }
            else {
                output[r_out * 28 + c_out] = val;
            }
        }
    }

    return output;
}
