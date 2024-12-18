#include "jpeg_cpu.h"
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <limits>
#include <filesystem> // Requires C++17
#include <algorithm>
#include "lanczos.h"

// Namespace alias for filesystem
namespace fs = std::filesystem;


// Function to list JPEG files in the current directory
std::vector<fs::path> list_jpeg_files(const fs::path& directory) {
    std::vector<fs::path> jpeg_files;
    if (!fs::exists(directory) || !fs::is_directory(directory)) {
        throw std::runtime_error("Invalid directory: " + directory.string());
    }

    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file()) {
            std::string ext = entry.path().extension().string();
            // Convert extension to lowercase for case-insensitive comparison
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".jpg" || ext == ".jpeg") {
                jpeg_files.push_back(entry.path());
            }
        }
    }

    return jpeg_files;
}

int main() {
    std::string output_image;
    float scale_factor;
    std::string method_str;

    // Get current working directory
    fs::path current_dir = fs::current_path();

    // List JPEG files in the current directory
    std::vector<fs::path> jpeg_files;
    try {
        jpeg_files = list_jpeg_files(current_dir);
    }
    catch (const std::exception& e) {
        std::cerr << "Error accessing directory: " << e.what() << "\n";
        return 1;
    }

    if (jpeg_files.empty()) {
        std::cerr << "No JPEG files found in the current directory (" << current_dir << ").\n";
        return 1;
    }

    // Display the list of JPEG files
    std::cout << "JPEG files found in " << current_dir << ":\n";
    for (size_t i = 0; i < jpeg_files.size(); ++i) {
        std::cout << "  [" << i + 1 << "] " << jpeg_files[i].filename().string() << "\n";
    }

    // Prompt user to select an image
    size_t selection = 0;
    while (true) {
        std::cout << "Select an image to upscale (1-" << jpeg_files.size() << "): ";
        std::cin >> selection;

        if (std::cin.fail() || selection < 1 || selection > jpeg_files.size()) {
            std::cin.clear(); // Clear error flags
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Discard invalid input
            std::cerr << "Invalid selection. Please enter a number between 1 and " << jpeg_files.size() << ".\n";
        }
        else {
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Discard any remaining input
            break;
        }
    }

    // Get the selected input image path
    fs::path input_image = jpeg_files[selection - 1];
    std::cout << "Selected image: " << input_image.filename().string() << "\n";

    // Prompt for output image path
    while (true) {
        std::cout << "Enter output image filename (e.g., upscaled.jpg): ";
        std::getline(std::cin, output_image);

        if (output_image.empty()) {
            std::cerr << "Output filename cannot be empty. Please try again.\n";
            continue;
        }

        // Check if the file already exists
        fs::path output_path = current_dir / output_image;
        if (fs::exists(output_path)) {
            std::cout << "File " << output_image << " already exists. Overwrite? (y/n): ";
            char overwrite;
            std::cin >> overwrite;
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Discard remaining input

            if (overwrite == 'y' || overwrite == 'Y') {
                break;
            }
            else {
                std::cout << "Please enter a different filename.\n";
            }
        }
        else {
            break;
        }
    }

    // Prompt for scale factor
    while (true) {
        std::cout << "Enter scale factor (e.g., 2.0 for 2x): ";
        std::cin >> scale_factor;

        if (std::cin.fail() || scale_factor <= 0.0f) {
            std::cin.clear(); // Clear error flags
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Discard invalid input
            std::cerr << "Invalid scale factor. Please enter a positive number.\n";
        }
        else {
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Discard any remaining input
            break;
        }
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
        fs::path output_path = current_dir / output_image;
        JPEGProcessor::write_jpeg_file(output_path.string().c_str(), upscaled_image, new_width, new_height, channels, 90);
        std::cout << "Upscaled image written to " << output_path << "\n";
    }
    catch (const std::exception& e) {
        std::cerr << "Error writing output image: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
