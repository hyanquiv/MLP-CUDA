#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <vector>
#include <string>

struct MNISTData {
    std::vector<float> images;
    std::vector<int> labels;
    int num_samples;
    int image_size;
};

MNISTData load_mnist(const std::string& image_path, const std::string& label_path);
void normalize_data(std::vector<float>& images);
void free_mnist(MNISTData& dataset);
void shuffle_data(MNISTData &data);

#endif