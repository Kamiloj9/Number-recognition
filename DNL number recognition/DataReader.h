#pragma once
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdint>

namespace DataReader {
    uint32_t readBigEndianUInt32(std::ifstream& ifs) {
        unsigned char bytes[4];
        if (!ifs.read(reinterpret_cast<char*>(bytes), 4)) {
            throw std::runtime_error("Error: unable to read 4 bytes from file.");
        }
        // Convert from big-endian to host-endian
        return (uint32_t(bytes[0]) << 24) |
            (uint32_t(bytes[1]) << 16) |
            (uint32_t(bytes[2]) << 8) |
            uint32_t(bytes[3]);
    }

    // Read MNIST images & labels from the given file paths.
// Returns a pair: (images, labels)
//   - images: shape [num_samples][rows*cols], each pixel in [0..255]
//   - labels: shape [num_samples], each label in [0..9]
    static std::pair<std::vector<std::vector<double>>, std::vector<int>>
        readMNISTImagesAndLabels(const std::string& imagesPath, const std::string& labelsPath)
    {
        // 1) Read labels
        std::ifstream labelsFile(labelsPath, std::ios::binary);
        if (!labelsFile) {
            throw std::runtime_error("Cannot open labels file: " + labelsPath);
        }
        uint32_t magic = readBigEndianUInt32(labelsFile);
        if (magic != 2049) {
            throw std::runtime_error("Invalid magic number in labels file (expected 2049).");
        }
        uint32_t numLabels = readBigEndianUInt32(labelsFile);

        std::vector<int> labels(numLabels);
        for (uint32_t i = 0; i < numLabels; ++i) {
            unsigned char labelByte;
            if (!labelsFile.read(reinterpret_cast<char*>(&labelByte), 1)) {
                throw std::runtime_error("Error reading label data.");
            }
            labels[i] = static_cast<int>(labelByte);
        }
        labelsFile.close();

        // 2) Read images
        std::ifstream imagesFile(imagesPath, std::ios::binary);
        if (!imagesFile) {
            throw std::runtime_error("Cannot open images file: " + imagesPath);
        }
        magic = readBigEndianUInt32(imagesFile);
        if (magic != 2051) {
            throw std::runtime_error("Invalid magic number in images file (expected 2051).");
        }
        uint32_t numImages = readBigEndianUInt32(imagesFile);
        uint32_t numRows = readBigEndianUInt32(imagesFile);
        uint32_t numCols = readBigEndianUInt32(imagesFile);

        // Check label/image count
        if (numImages != numLabels) {
            // Typically they match (e.g., 60000/60000 for training, 10000/10000 for test)
            // but let's just warn
            throw std::runtime_error("Mismatch: number of images != number of labels.");
        }

        // Read image data
        const size_t imageSize = static_cast<size_t>(numRows) * numCols; // 28*28=784
        std::vector<std::vector<double>> images(numImages, std::vector<double>(imageSize));

        for (uint32_t i = 0; i < numImages; ++i) {
            for (uint32_t px = 0; px < imageSize; ++px) {
                unsigned char pixelByte;
                if (!imagesFile.read(reinterpret_cast<char*>(&pixelByte), 1)) {
                    throw std::runtime_error("Error reading image data.");
                }
                // Scale pixel from [0..255] to [0..1]
                images[i][px] = static_cast<double>(pixelByte) / 255.0;
            }
        }
        imagesFile.close();

        return { images, labels };
    }
}