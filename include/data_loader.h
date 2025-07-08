#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <vector>
#include <string>

struct MNISTData
{
    std::vector<float> images; // En este caso: ViT embeddings por fila
    std::vector<int> labels;
    int num_samples;
    int image_size; // será 768 (dimensión del embedding)
};

MNISTData load_from_csv(const std::string &features_csv, const std::string &labels_csv);
void normalize_data(std::vector<float> &images);
void free_mnist(MNISTData &dataset);
void shuffle_data(MNISTData &data);

#endif
