#include "data_loader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cstdlib>
#include <ctime>

MNISTData load_from_csv(const std::string &features_csv, const std::string &labels_csv)
{
    MNISTData dataset;

    // Leer archivo de características
    std::ifstream feature_file(features_csv);
    if (!feature_file.is_open())
    {
        throw std::runtime_error("No se pudo abrir el archivo de embeddings: " + features_csv);
    }

    std::string line;
    int embedding_dim = -1;

    while (std::getline(feature_file, line))
    {
        if (line.empty())
            continue; // ← Ignorar líneas vacías

        std::stringstream ss(line);
        std::string value;
        int count = 0;

        while (std::getline(ss, value, ','))
        {
            dataset.images.push_back(std::stof(value));
            count++;
        }

        if (embedding_dim == -1)
            embedding_dim = count;
    }

    dataset.image_size = embedding_dim;
    dataset.num_samples = dataset.images.size() / embedding_dim;

    // Leer archivo de etiquetas (CSV con encabezado)
    std::ifstream label_file(labels_csv);
    if (!label_file.is_open())
    {
        throw std::runtime_error("No se pudo abrir el archivo de etiquetas: " + labels_csv);
    }

    std::string label_line;

    // Saltar encabezado
    std::getline(label_file, label_line);

    // Leer etiquetas
    while (std::getline(label_file, label_line))
    {
        if (!label_line.empty())
        {
            dataset.labels.push_back(std::stoi(label_line));
        }
    }

    if (dataset.labels.size() != dataset.num_samples)
    {
        throw std::runtime_error("❌ Cantidad de etiquetas (" + std::to_string(dataset.labels.size()) +
                                 ") no coincide con cantidad de muestras (" + std::to_string(dataset.num_samples) + ")");
    }

    return dataset;
}

void normalize_data(std::vector<float> &images)
{
    // Embeddings de ViT ya están normalizados, no hacer nada.
}

void free_mnist(MNISTData &dataset)
{
    dataset.images.clear();
    dataset.images.shrink_to_fit();
    dataset.labels.clear();
    dataset.labels.shrink_to_fit();
    dataset.num_samples = 0;
    dataset.image_size = 0;
}

void shuffle_data(MNISTData &data)
{
    int n = data.num_samples;
    int image_size = data.image_size;

    static bool seeded = false;
    if (!seeded)
    {
        std::srand(static_cast<unsigned>(std::time(nullptr)));
        seeded = true;
    }

    for (int i = n - 1; i > 0; i--)
    {
        int j = std::rand() % (i + 1);

        for (int k = 0; k < image_size; k++)
        {
            std::swap(data.images[i * image_size + k],
                      data.images[j * image_size + k]);
        }

        std::swap(data.labels[i], data.labels[j]);
    }
}
