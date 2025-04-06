#include <iostream>
#include <filesystem> 
#include <SFML/Graphics.hpp>
#include <random>

#include "math.h"
#include "DataReader.h"
#include "Model.h"

namespace fs = std::filesystem;

// A helper to get a grayscale brightness in [0..1] from an SFML pixel
inline double pixelBrightness(const sf::Color& c) {
    // simple average of (r,g,b)
    return (c.r + c.g + c.b) / 3.0 / 255.0;
}

// Helper: nearest-neighbor sampling from an sf::Image
inline double sampleNearest(const sf::Image& img, float x, float y) {
    // Round x,y
    int ix = static_cast<int>(std::round(x));
    int iy = static_cast<int>(std::round(y));
    // Clamp to image bounds
    ix = std::max(0, std::min(ix, (int)img.getSize().x - 1));
    iy = std::max(0, std::min(iy, (int)img.getSize().y - 1));
    return pixelBrightness(img.getPixel(ix, iy));
}

// Captures the user's 280×280 drawing in renderTex, extracts the bounding
// box of non-black pixels, scales it to ~20×20, and centers it in 28×28.
std::vector<double> captureAndScale(const sf::RenderTexture& renderTex) {
    // 1) Copy the entire 280×280 image
    sf::Image screenshot = renderTex.getTexture().copyToImage();
    unsigned int W = screenshot.getSize().x; // e.g. 280
    unsigned int H = screenshot.getSize().y; // e.g. 280

    // 2) Find bounding box of the digit
    //    i.e., minX, maxX, minY, maxY of pixels that are not "near black."
    int minX = W, maxX = -1;
    int minY = H, maxY = -1;

    // We'll define "black" as brightness < some threshold (like 0.1).
    // If your strokes are pure white on black, threshold can be small.
    const double threshold = 0.1;

    for (unsigned int y = 0; y < H; ++y) {
        for (unsigned int x = 0; x < W; ++x) {
            double b = pixelBrightness(screenshot.getPixel(x, y));
            if (b > threshold) {
                if ((int)x < minX) minX = x;
                if ((int)x > maxX) maxX = x;
                if ((int)y < minY) minY = y;
                if ((int)y > maxY) maxY = y;
            }
        }
    }

    // If nothing found (digit not drawn), return a blank 28×28
    if (maxX < 0 || maxY < 0) {
        return std::vector<double>(28 * 28, 0.0);
    }

    // 3) Determine bounding box width/height
    int bw = maxX - minX + 1;
    int bh = maxY - minY + 1;

    // 4) We want the largest dimension to be ~20,
    //    so let's compute a scale factor
    const int TARGET_SIZE = 20; // typical MNIST digit region is ~20x20
    float scale = 1.0f;
    int biggerDim = std::max(bw, bh);
    if (biggerDim > 0) {
        scale = (float)TARGET_SIZE / (float)biggerDim;
    }

    // 5) Now we create a 28x28 output, initially black
    std::vector<double> out(28 * 28, 0.0);

    // 6) We'll map each pixel in [0..27,0..27] to the bounding box
    //    using the inverse transform:
    //    X_in = minX + (x_out - cx_out)/scale + cx_box
    // We'll center the scaled bounding box in the 28×28. 
    // So the top-left corner of the scaled box in the 28×28 might be:
    int scaledW = (int)std::round(bw * scale);
    int scaledH = (int)std::round(bh * scale);

    // Center offset in 28×28
    int offsetX = (28 - scaledW) / 2;
    int offsetY = (28 - scaledH) / 2;

    for (int yOut = 0; yOut < 28; ++yOut) {
        for (int xOut = 0; xOut < 28; ++xOut) {
            // Check if we're within the scaled bounding box region
            // in the output
            if (xOut >= offsetX && xOut < offsetX + scaledW &&
                yOut >= offsetY && yOut < offsetY + scaledH)
            {
                // Map output coords -> input coords
                float inX = minX + ((xOut - offsetX) / scale);
                float inY = minY + ((yOut - offsetY) / scale);

                // Sample nearest from the screenshot
                double val = sampleNearest(screenshot, inX, inY);
                // Store in out[yOut * 28 + xOut]
                out[yOut * 28 + xOut] = val;
            }
        }
    }

    return out;
}

int main()
{

    std::string trainImagesFile = "dataset/train-images.idx3-ubyte";
    std::string trainLabelsFile = "dataset/train-labels.idx1-ubyte";
    std::string testImagesFile = "dataset/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte";
    std::string testLabelsFile = "dataset/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte";

    try {


        std::vector<std::string> modelFiles;

        fs::path modelDir("models");
        // Make sure the directory exists (optional).
        if (!fs::exists(modelDir)) {
            fs::create_directory(modelDir);
        }

        // List all ".model" files in "models/"
        for (auto& entry : fs::directory_iterator(modelDir)) {
            if (entry.is_regular_file()) {
                auto path = entry.path();
                if (path.extension() == ".model") {
                    modelFiles.push_back(path.string());
                }
            }
        }
        
        // 784 inputs -> 128 hidden -> 10 outputs
        Model net(784, 128, 10, 0.01);

        if (modelFiles.empty()) {

            // Read training data
            auto [trainImages, trainLabels] =
                DataReader::readMNISTImagesAndLabels(trainImagesFile, trainLabelsFile);

            // Read test data
            auto [testImages, testLabels] =
                DataReader::readMNISTImagesAndLabels(testImagesFile, testLabelsFile);

            std::cout << "Train set size: " << trainImages.size() << " images\n";
            std::cout << "Test set size:  " << testImages.size() << " images\n";

            std::vector<std::vector<double>> augmentedImages = trainImages;
            std::vector<int> augmentedLabels = trainLabels;

            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<double> angleDist(-15.0, 15.0);
            std::uniform_real_distribution<double> scaleDist(0.7, 1.3);
            std::uniform_int_distribution<int> shiftDist(-3, 3);

            for (std::size_t i = 0; i < trainImages.size(); ++i) {
                // Generate 2 random transformations
                for (int a = 0; a < 10; ++a) {
                    double angle = angleDist(gen);
                    double scale = scaleDist(gen);
                    int shiftX = shiftDist(gen);
                    int shiftY = shiftDist(gen);

                    auto newImg = utils::augmentImage(trainImages[i], angle, scale, shiftX, shiftY);

                    // Same label
                    augmentedImages.push_back(newImg);
                    augmentedLabels.push_back(trainLabels[i]);
                }
            }

            std::cout << "Augmented dataset size: " << augmentedImages.size() << " images\n";

            // 4) Train (for e.g. 5 epochs)
            net.train(augmentedImages, augmentedLabels, 8);

            // 5) Evaluate on test data (accuracy)
            int correct = 0;
            for (size_t i = 0; i < testImages.size(); ++i) {
                int pred = net.predict(testImages[i]);
                if (pred == testLabels[i]) {
                    correct++;
                }
            }
            double accuracy = 100.0 * correct / testImages.size();
            std::cout << "Test accuracy: " << accuracy << "%" << std::endl;

            // Save the newly trained model
            std::string defaultModel = (modelDir / "default.model").string();
            net.saveModel(defaultModel);
            std::cout << "Saved new model to: " << defaultModel << std::endl;
        }
        else {
            std::cout << "Found model files in 'models/' directory:\n";
            for (size_t i = 0; i < modelFiles.size(); ++i) {
                std::cout << "  [" << i << "] " << modelFiles[i] << "\n";
            }
            std::cout << "Choose a model index [0.." << (modelFiles.size() - 1) << "]: ";

            int choice = 0;
            std::cin >> choice;
            if (!std::cin || choice < 0 || static_cast<size_t>(choice) >= modelFiles.size()) {
                std::cout << "Invalid choice, defaulting to index 0.\n";
                choice = 0;
            }

            std::string chosenModel = modelFiles[choice];
            std::cout << "Loading model: " << chosenModel << std::endl;

            // Example: If you're expecting the same network shape as was saved
            net.loadModel(chosenModel);

            // Now we can use net for inference or further training
            // e.g., evaluate on test set

            /*
            std::string testImagesFile = "dataset/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte";
            std::string testLabelsFile = "dataset/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte";
            auto [testImages, testLabels] =
                DataReader::readMNISTImagesAndLabels(testImagesFile, testLabelsFile);

            int correct = 0;
            for (size_t i = 0; i < testImages.size(); ++i) {
                int pred = net.predict(testImages[i]);
                if (pred == testLabels[i]) {
                    correct++;
                }
            }
            double accuracy = 100.0 * correct / testImages.size();
            std::cout << "Test accuracy: " << accuracy << "%\n";
            */
        }

        // create GUI
        // bigger for user drawing
        const unsigned int CANVAS_WIDTH = 280;  
        const unsigned int CANVAS_HEIGHT = 280;

        const unsigned int WINDOW_WIDTH = 500;
        const unsigned int WINDOW_HEIGHT = 300;

        sf::RenderWindow window(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT),
            "Number Prediction Model");
        window.setFramerateLimit(60);

        // A render texture where we actually draw
        sf::RenderTexture renderTex;
        renderTex.create(CANVAS_WIDTH, CANVAS_HEIGHT);
        renderTex.clear(sf::Color::Black);
        renderTex.display();

        bool drawing = false;
        float brushRadius = 4.8f;//8.0f;

        // buttons
        sf::RectangleShape btnClear(sf::Vector2f(100, 40));
        btnClear.setPosition((float)(CANVAS_WIDTH + 20), 50.0f);
        btnClear.setFillColor(sf::Color(100, 100, 100));  // gray

        sf::RectangleShape btnPredict(sf::Vector2f(100, 40));
        btnPredict.setPosition((float)(CANVAS_WIDTH + 20), 120.0f);
        btnPredict.setFillColor(sf::Color(100, 100, 100)); // gray

        sf::Font font;
        if (!font.loadFromFile("Verdana.ttf")) {
            std::cout << "Warning: could not load font. Text won't display.\n";
        }

        sf::Text predictionText("Prediction: ?", font, 24);
        predictionText.setFillColor(sf::Color::White);
        predictionText.setPosition((float)(CANVAS_WIDTH + 20), 200.0f);

        // Label text for the buttons
        sf::Text clearLabel("Clear", font, 18);
        clearLabel.setFillColor(sf::Color::Black);
        clearLabel.setPosition(btnClear.getPosition().x + 10, btnClear.getPosition().y + 8);

        sf::Text predictLabel("Predict", font, 18);
        predictLabel.setFillColor(sf::Color::Black);
        predictLabel.setPosition(btnPredict.getPosition().x + 5, btnPredict.getPosition().y + 8);

        while (window.isOpen()) {
            sf::Event event;
            while (window.pollEvent(event)) {
                switch (event.type) {
                case sf::Event::Closed:
                    window.close();
                    break;

                case sf::Event::MouseButtonPressed:
                    if (event.mouseButton.button == sf::Mouse::Left) {
                        // Check if mouse is in canvas area
                        if (event.mouseButton.x >= 0 && event.mouseButton.x < (int)CANVAS_WIDTH &&
                            event.mouseButton.y >= 0 && event.mouseButton.y < (int)CANVAS_HEIGHT)
                        {
                            drawing = true;
                        }
                        else {
                            drawing = false;

                            // Check if clicked "Clear" button
                            sf::Vector2f mp((float)event.mouseButton.x, (float)event.mouseButton.y);
                            if (btnClear.getGlobalBounds().contains(mp)) {
                                // Clear the canvas
                                renderTex.clear(sf::Color::Black);
                                renderTex.display();
                                predictionText.setString("Prediction: ?");
                            }

                            // Check if clicked "Predict" button
                            if (btnPredict.getGlobalBounds().contains(mp)) {
                                // Predict
                                std::vector<double> scaled = captureAndScale(renderTex);
                                int pred = net.predict(scaled);
                                // Update the text
                                predictionText.setString("Prediction: " + std::to_string(pred));
                            }
                        }
                    }
                    break;

                case sf::Event::MouseButtonReleased:
                    if (event.mouseButton.button == sf::Mouse::Left) {
                        drawing = false;
                    }
                    break;

                default:
                    break;
                }
            }

            // If drawing == true, draw a circle on the renderTex
            if (drawing) {
                sf::Vector2i mousePos = sf::Mouse::getPosition(window);
                // Make sure it remains in canvas bounds
                if (mousePos.x >= 0 && mousePos.x < (int)CANVAS_WIDTH &&
                    mousePos.y >= 0 && mousePos.y < (int)CANVAS_HEIGHT)
                {
                    sf::CircleShape brush(brushRadius);
                    brush.setFillColor(sf::Color::White);
                    brush.setPosition(mousePos.x - brushRadius, mousePos.y - brushRadius);
                    renderTex.draw(brush);
                    renderTex.display();
                }
                else {
                    drawing = false;
                }
            }

            // ----------------------------------------------------------------
            // Draw everything
            // ----------------------------------------------------------------
            window.clear(sf::Color(50, 50, 50)); // some background color

            // 1) Draw the canvas (from renderTex)
            sf::Sprite canvasSprite(renderTex.getTexture());
            window.draw(canvasSprite);

            // 2) Draw buttons
            window.draw(btnClear);
            window.draw(btnPredict);

            // 3) Draw button labels
            if (font.getInfo().family != "") { // means we loaded a font
                window.draw(clearLabel);
                window.draw(predictLabel);
                window.draw(predictionText);
            }

            window.display();
        }
    }
    catch (std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    }

    return 0;
}