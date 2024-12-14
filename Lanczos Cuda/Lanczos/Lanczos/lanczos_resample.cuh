#ifndef LANCZOS_RESAMPLE_H
#define LANCZOS_RESAMPLE_H

#include <vector>
#include <cmath>
#include <algorithm>

// Constants
#define PI 3.14159265358979323846

// Structure to hold resampling parameters for an axis
//struct AxisParam {
//    std::vector<int> start;     // Starting index for each output pixel
//    std::vector<int> length;    // Number of source pixels contributing to each output pixel
//    std::vector<double> weight;  // Weights for resampling
//    std::vector<int> index;     // Starting index in weight array for each output pixel
//
//    template<typename TWeightFunc>
//    void calculateAxis(int srcstart, int srcend, int srclength, int dstlength, int a, TWeightFunc& func) {
//        start.clear();
//        length.clear();
//        index.clear();
//        weight.clear();
//
//        int total_weights = 0;
//
//        if (srclength <= dstlength) {
//            // Upscaling
//            double scale = static_cast<double>(dstlength) / srclength;
//            double inv_scale = 1.0f / scale;
//            double range = static_cast<double>(a);
//
//            for (int i = 0; i < dstlength; ++i) {
//                double center = (i + 0.5f) * inv_scale + srcstart;
//                int left = static_cast<int>(std::floor(center - range));
//                int right = static_cast<int>(std::ceil(center + range));
//
//                left = std::max(left, srcstart);
//                right = std::min(right, srcend - 1);
//
//                start.push_back(left);
//                int len = right - left + 1;
//                length.push_back(len);
//                index.push_back(total_weights);
//
//                for (int j = left; j <= right; ++j) {
//                    double dist = fabsf(center - (j + 0.5f));
//                    double w = func(dist);
//                    weight.push_back(w);
//                    total_weights++;
//                }
//            }
//        }
//        else {
//            // Downscaling
//            double scale = static_cast<double>(dstlength) / srclength;
//            double range = a / scale;
//
//            for (int i = 0; i < dstlength; ++i) {
//                double center = (i + 0.5f) / scale + srcstart;
//                int left = static_cast<int>(std::floor(center - range));
//                int right = static_cast<int>(std::ceil(center + range));
//
//                left = std::max(left, srcstart);
//                right = std::min(right, srcend - 1);
//
//                start.push_back(left);
//                int len = right - left + 1;
//                length.push_back(len);
//                index.push_back(total_weights);
//
//                for (int j = left; j <= right; ++j) {
//                    double dist = fabsf(center - (j + 0.5f));
//                    double w = func(dist * scale);
//                    weight.push_back(w);
//                    total_weights++;
//                }
//            }
//        }
//    }
//};

// Lanczos weight function
template<int TTap>
struct LanczosWeight {
    double operator()(double x) {
        if (fabs(x) < 1e-12) return 1.0;
        if (fabs(x) >= static_cast<double>(TTap)) return 0.0;
        double pix = PI * x;
        return sin(pix) * sin(pix / TTap) / (pix * pix / TTap);
    }
};

// Function to perform Lanczos resampling on the GPU
void lanczos_resample_cuda(
    const unsigned char* h_input,
    unsigned char* h_output,
    int in_width,
    int in_height,
    int out_width,
    int out_height,
    int channels,
    int a);

#endif  // LANCZOS_RESAMPLE_H
