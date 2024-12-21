#include "jpeg_cpu.h"
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <filesystem> // Requires C++17
#include <algorithm>
#include "lanczos.h"

// Namespace alias for filesystem
namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
    // Ensure correct number of arguments
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <input_image> <output_image> <scale_factor>\n";
        return 1;
    }

    // Parse arguments
    fs::path input_image = argv[1];
    fs::path output_image = argv[2];
    float scale_factor;

    try {
        scale_factor = std::stof(argv[3]);
    }
    catch (const std::exception& e) {
        std::cerr << "Invalid scale factor: " << argv[3] << ". Please provide a positive number.\n";
        return 1;
    }

    if (scale_factor <= 0.0f) {
        std::cerr << "Scale factor must be a positive number.\n";
        return 1;
    }

    // Check if the input image exists
    if (!fs::exists(input_image)) {
        std::cerr << "Input image does not exist: " << input_image << "\n";
        return 1;
    }

    // Check for valid JPEG file extension
    std::string ext = input_image.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if (ext != ".jpg" && ext != ".jpeg") {
        std::cerr << "Input file must be a JPEG image (.jpg or .jpeg).\n";
        return 1;
    }

    // Initialize variables for image data
    std::vector<unsigned char> image_data;
    int width, height, channels;

    try {
        // Read the input JPEG file
        JPEGProcessor::read_jpeg_file(input_image.string().c_str(), image_data, width, height, channels);
        std::cout << "Image read: " << width << "x" << height << " with " << channels << " channels.\n";
    }
    catch (const std::exception& e) {
        std::cerr << "Error reading input image: " << e.what() << "\n";
        return 1;
    }

    // Calculate new dimensions based on scale factor
    int new_width = static_cast<int>(width * scale_factor);
    int new_height = static_cast<int>(height * scale_factor);

    std::vector<unsigned char> upscaled_image;
    try {
        // Perform the upscaling
        upscaled_image = Lanczos::upscale(image_data, width, height, channels, new_width, new_height);
    }
    catch (const std::exception& e) {
        std::cerr << "Error during upscaling: " << e.what() << "\n";
        return 1;
    }

    try {
        // Write the upscaled image to the output file
        JPEGProcessor::write_jpeg_file(output_image.string().c_str(), upscaled_image, new_width, new_height, channels, 90);
        std::cout << "Upscaled image written to " << output_image << "\n";
    }
    catch (const std::exception& e) {
        std::cerr << "Error writing output image: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
