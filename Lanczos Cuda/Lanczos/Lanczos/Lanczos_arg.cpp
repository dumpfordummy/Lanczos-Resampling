#include <iostream>
#include <opencv2/opencv.hpp>
#include "lanczos_resample.cuh"
#include <filesystem>  // Requires C++17

namespace fs = std::filesystem;

// sRGB to Linear Color Space Conversion
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

// Linear to sRGB Color Space Conversion
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

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <input_image> <output_image> <scale_factor>\n";
        return -1;
    }

    // Parse arguments
    fs::path input_image_path = argv[1];
    fs::path output_image_path = argv[2];
    double scale;

    try {
        scale = std::stod(argv[3]);
    }
    catch (...) {
        std::cerr << "Error: Scale factor must be a positive number.\n";
        return -1;
    }

    if (scale <= 0) {
        std::cerr << "Error: Scale factor must be greater than zero.\n";
        return -1;
    }

    // Check if the input image exists
    if (!fs::exists(input_image_path)) {
        std::cerr << "Error: Input image does not exist: " << input_image_path << "\n";
        return -1;
    }

    // Read input image
    cv::Mat input_image = cv::imread(input_image_path.string(), cv::IMREAD_COLOR);
    if (input_image.empty()) {
        std::cerr << "Error: Could not read the input image.\n";
        return -1;
    }

    int in_width = input_image.cols;
    int in_height = input_image.rows;
    int channels = input_image.channels();

    int out_width = static_cast<int>(in_width * scale);
    int out_height = static_cast<int>(in_height * scale);

    std::cout << "Resizing image from " << in_width << "x" << in_height
        << " to " << out_width << "x" << out_height << ".\n";

    // Allocate output image
    cv::Mat output_image(out_height, out_width, input_image.type());

    const int a = 8;  // Lanczos parameter

    // Perform resampling on the GPU
    try {
        lanczos_resample_cuda(
            input_image.data,
            output_image.data,
            in_width,
            in_height,
            out_width,
            out_height,
            channels,
            a);
    }
    catch (const std::exception& e) {
        std::cerr << "Error during resampling: " << e.what() << "\n";
        return -1;
    }

    // Save the output image
    if (!cv::imwrite(output_image_path.string(), output_image)) {
        std::cerr << "Error: Could not write the output image.\n";
        return -1;
    }

    std::cout << "Image resizing completed successfully.\n";
    std::cout << "Output image saved at: " << output_image_path << "\n";

    return 0;
}
