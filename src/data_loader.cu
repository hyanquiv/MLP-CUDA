#include "data_loader.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cstdlib>
#include <ctime>

int reverse_int(int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return (ch1 << 24) | (ch2 << 16) | (ch3 << 8) | ch4;
}

MNISTData load_mnist(const std::string &image_path, const std::string &label_path)
{
    MNISTData dataset;
    std::ifstream image_file(image_path, std::ios::binary);
    std::ifstream label_file(label_path, std::ios::binary);

    if (!image_file.is_open())
    {
        throw std::runtime_error("No se pudo abrir el archivo de imágenes: " + image_path);
    }
    if (!label_file.is_open())
    {
        throw std::runtime_error("No se pudo abrir el archivo de etiquetas: " + label_path);
    }

    // Leer encabezado de imágenes
    int magic_number = 0;
    image_file.read((char *)&magic_number, sizeof(magic_number));
    magic_number = reverse_int(magic_number);

    if (magic_number != 2051)
    {
        throw std::runtime_error("Número mágico inválido en archivo de imágenes");
    }

    image_file.read((char *)&dataset.num_samples, sizeof(dataset.num_samples));
    dataset.num_samples = reverse_int(dataset.num_samples);

    int num_rows = 0, num_cols = 0;
    image_file.read((char *)&num_rows, sizeof(num_rows));
    image_file.read((char *)&num_cols, sizeof(num_cols));
    num_rows = reverse_int(num_rows);
    num_cols = reverse_int(num_cols);

    dataset.image_size = num_rows * num_cols;

    // Leer encabezado de etiquetas
    int label_magic = 0;
    label_file.read((char *)&label_magic, sizeof(label_magic));
    label_magic = reverse_int(label_magic);

    if (label_magic != 2049)
    {
        throw std::runtime_error("Número mágico inválido en archivo de etiquetas");
    }

    int num_labels = 0;
    label_file.read((char *)&num_labels, sizeof(num_labels));
    num_labels = reverse_int(num_labels);

    if (num_labels != dataset.num_samples)
    {
        throw std::runtime_error("Número de imágenes y etiquetas no coincide");
    }

    // Leer datos de imágenes
    const size_t image_data_size = dataset.num_samples * dataset.image_size;
    dataset.images.resize(image_data_size);

    for (int i = 0; i < dataset.num_samples; i++)
    {
        std::vector<unsigned char> temp(dataset.image_size);
        image_file.read((char *)temp.data(), dataset.image_size);

        for (int j = 0; j < dataset.image_size; j++)
        {
            dataset.images[i * dataset.image_size + j] = static_cast<float>(temp[j]);
        }
    }

    // Leer etiquetas
    dataset.labels.resize(dataset.num_samples);
    for (int i = 0; i < dataset.num_samples; i++)
    {
        unsigned char label = 0;
        label_file.read((char *)&label, sizeof(label));
        dataset.labels[i] = static_cast<int>(label);
    }

    return dataset;
}

void normalize_data(std::vector<float> &images)
{
    for (auto &pixel : images)
    {
        pixel = pixel / 255.0f;
    }
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

    // Inicializar semilla aleatoria
    static bool seeded = false;
    if (!seeded)
    {
        std::srand(static_cast<unsigned>(std::time(nullptr)));
        seeded = true;
    }

    // Barajar usando el algoritmo Fisher-Yates
    for (int i = n - 1; i > 0; i--)
    {
        // Generar índice aleatorio entre 0 e i
        int j = std::rand() % (i + 1);

        // Intercambiar imágenes
        for (int k = 0; k < image_size; k++)
        {
            std::swap(data.images[i * image_size + k],
                      data.images[j * image_size + k]);
        }

        // Intercambiar etiquetas
        std::swap(data.labels[i], data.labels[j]);
    }
}