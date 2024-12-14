#include <iostream>
#include <opencv2/opencv.hpp>
#include "lanczos_resample.cuh"
#include <filesystem>  // Requires C++17
#include <limits>      // For std::numeric_limits

namespace fs = std::filesystem;

// Assuming AxisParam and LanczosWeight are defined in "lanczos_resample.cuh"
// If not, include their definitions or necessary headers.

void sRGB_to_linear(cv::Mat& img) {
    img.convertTo(img, CV_64F, 1.0 / 255.0);
    cv::MatIterator_<cv::Vec3d> it, end;
    for (it = img.begin<cv::Vec3d>(), end = img.end<cv::Vec3d>(); it != end; ++it) {
        for (int c = 0; c < 3; ++c) {
            double& channel = (*it)[c];
            if (channel <= 0.04045)
                channel /= 12.92;
            else
                channel = pow((channel + 0.055) / 1.055, 2.4);
        }
    }
}

void linear_to_sRGB(cv::Mat& img) {
    cv::MatIterator_<cv::Vec3d> it, end;
    for (it = img.begin<cv::Vec3d>(), end = img.end<cv::Vec3d>(); it != end; ++it) {
        for (int c = 0; c < 3; ++c) {
            double& channel = (*it)[c];
            if (channel <= 0.0031308)
                channel *= 12.92;
            else
                channel = 1.055 * pow(channel, 1.0 / 2.4) - 0.055;
        }
    }
    img.convertTo(img, CV_8U, 255.0);
}

int main() {
    std::string input_image_filename;
    std::string output_image_filename;
    double scale_x;
    double scale_y;

    // Get the current working directory
    fs::path current_dir = fs::current_path();
    std::cout << "Current Image Directory: " << current_dir << std::endl;

    // Prompt for input image filename
    std::cout << "Enter the input image filename (located in the current directory): ";
    std::getline(std::cin, input_image_filename);
    if (input_image_filename.empty()) {
        std::cerr << "Input image filename cannot be empty." << std::endl;
        return -1;
    }

    // Construct the full path for the input image
    fs::path input_image_path = current_dir / input_image_filename;

    // Check if the input image exists
    if (!fs::exists(input_image_path)) {
        std::cerr << "Input image does not exist in the current directory: "
            << input_image_path << std::endl;
        return -1;
    }

    // Prompt for output image filename
    std::cout << "Enter the output image filename (will be saved in the current directory): ";
    std::getline(std::cin, output_image_filename);
    if (output_image_filename.empty()) {
        std::cerr << "Output image filename cannot be empty." << std::endl;
        return -1;
    }

    // Construct the full path for the output image
    fs::path output_image_path = current_dir / output_image_filename;

    // Prompt for scale_x
    std::cout << "Enter the scaling factor for the x-axis (positive number): ";
    while (!(std::cin >> scale_x) || scale_x <= 0) {
        std::cerr << "Invalid input. Please enter a positive number for scale_x: ";
        std::cin.clear(); // Clear the error flag
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Discard invalid input
    }

    // Prompt for scale_y
    std::cout << "Enter the scaling factor for the y-axis (positive number): ";
    while (!(std::cin >> scale_y) || scale_y <= 0) {
        std::cerr << "Invalid input. Please enter a positive number for scale_y: ";
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }

    // Optional: Consume any remaining newline characters
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    const int a = 8;  // Lanczos parameter

    // Read input image
    cv::Mat input_image = cv::imread(input_image_path.string(), cv::IMREAD_COLOR);
    if (input_image.empty()) {
        std::cerr << "Could not read the image: " << input_image_path << std::endl;
        return -1;
    }

    int in_width = input_image.cols;
    int in_height = input_image.rows;
    int channels = input_image.channels();

    int out_width = static_cast<int>(in_width * scale_x);
    int out_height = static_cast<int>(in_height * scale_y);

    // Allocate output image

    cv::Mat output_image(out_height, out_width, input_image.type());

    
    // Perform resampling on the GPU
    lanczos_resample_cuda(
        input_image.data,
        output_image.data,
        in_width,
        in_height,
        out_width,
        out_height,
        channels,
        a);



    // Save the output image
    if (!cv::imwrite(output_image_path.string(), output_image)) {
        std::cerr << "Could not write the image: " << output_image_path << std::endl;
        return -1;
    }

    std::cout << "Image resizing completed successfully." << std::endl;
    std::cout << "Output image saved at: " << output_image_path << std::endl;

    return 0;
}
