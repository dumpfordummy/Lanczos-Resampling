#include "upscaler.h"
#include "bicubic.h"
#include "lanczos.h"
#include "edi.h"
#include <stdexcept>

std::vector<unsigned char> Upscaler::upscale(const std::vector<unsigned char>& input,
	int input_width, int input_height, int channels,
	int output_width, int output_height) {
	return Lanczos::upscale(input, input_width, input_height, channels, output_width, output_height);
}