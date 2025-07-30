#include "data_loader.h"
#include "embedding.h"
#include "transformer.h"
#include "mlp_cuda.h"
#include "constants.h"
#include "cuda_utils.h"
#include "train.h"

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <iomanip>
#include <string>

int main()
{
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    std::cout << "Iniciando entrenamiento Transformer+MLP para MNIST" << std::endl;
    std::cout << "===================================================" << std::endl;

    const std::string base_path = "/content/MLP-CUDA/";

    // 1. Cargar datos
    MNISTData train_data = load_mnist(base_path + "data/train-images-idx3-ubyte",
                                      base_path + "data/train-labels-idx1-ubyte");

    MNISTData test_data = load_mnist(base_path + "data/t10k-images-idx3-ubyte",
                                     base_path + "data/t10k-labels-idx1-ubyte");

    std::cout << "\n - Entrenamiento: " << train_data.num_samples
              << "\n - Prueba: " << test_data.num_samples
              << "\n - Tamaño imagen: " << train_data.image_size << std::endl;

    // 2. Normalizar
    normalize_data(train_data.images);
    normalize_data(test_data.images);

    // 3. Crear modelo completo
    Embedding embed(IMG_HEIGHT, IMG_WIDTH, PATCH_SIZE, EMBED_DIM);
    Transformer transformer(EMBED_DIM);
    MLP mlp(NUM_PATCHES * EMBED_DIM, HIDDEN_SIZE, OUTPUT_SIZE);

    std::cout << "\nModelo inicializado." << std::endl;
    std::cout << "\nDimensiones:\n - MLP Input: " << NUM_PATCHES * EMBED_DIM
              << "\n - Num Patches: " << NUM_PATCHES
              << "\n - Embed Dim: " << EMBED_DIM << std::endl;

    // 4. Entrenar modelo
    try
    {
        train_model(embed, transformer, mlp, train_data, test_data);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error durante entrenamiento: " << e.what() << std::endl;
        free_mnist(train_data);
        free_mnist(test_data);
        return 1;
    }

    // 5. Guardar pesos del modelo
    embed.save_weights((base_path + "embedding_weights.bin").c_str());
    transformer.save_weights((base_path + "transformer_weights.bin").c_str());
    mlp.save_weights((base_path + "mlp_weights.bin").c_str());

    // 6. Evaluar
    const int num_classes = OUTPUT_SIZE;
    std::vector<int> true_per_class(num_classes, 0);
    std::vector<int> correct_per_class(num_classes, 0);
    std::vector<std::vector<int>> confusion_matrix(num_classes, std::vector<int>(num_classes, 0));

    float *d_embed_out, *d_trans_out;
    cudaMalloc(&d_embed_out, NUM_PATCHES * EMBED_DIM * sizeof(float));
    cudaMalloc(&d_trans_out, NUM_PATCHES * EMBED_DIM * sizeof(float));

    for (int i = 0; i < test_data.num_samples; ++i)
    {
        const float *img = test_data.images.data() + i * train_data.image_size;
        int true_label = test_data.labels[i];

        embed.forward(img, d_embed_out);
        transformer.forward(d_embed_out, d_trans_out);
        int pred = mlp.predict(d_trans_out);

        true_per_class[true_label]++;
        if (pred == true_label)
            correct_per_class[true_label]++;
        confusion_matrix[true_label][pred]++;
    }

    float balanced_acc = 0.0f;
    for (int i = 0; i < num_classes; ++i)
    {
        if (true_per_class[i] > 0)
            balanced_acc += static_cast<float>(correct_per_class[i]) / true_per_class[i];
    }
    balanced_acc /= num_classes;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\n✅ Accuracy balanceada: " << balanced_acc * 100.0f << "%" << std::endl;

    std::cout << "\n📊 Matriz de confusión:\n   ";
    for (int i = 0; i < num_classes; ++i)
        std::cout << std::setw(4) << i;
    std::cout << "\n";
    for (int i = 0; i < num_classes; ++i)
    {
        std::cout << std::setw(2) << i << ":";
        for (int j = 0; j < num_classes; ++j)
            std::cout << std::setw(4) << confusion_matrix[i][j];
        std::cout << "\n";
    }

    // 7. Predicción aleatoria
    int idx = std::rand() % test_data.num_samples;
    const float *rand_img = test_data.images.data() + idx * train_data.image_size;
    int true_label = test_data.labels[idx];

    embed.forward(rand_img, d_embed_out);
    transformer.forward(d_embed_out, d_trans_out);
    int pred_label = mlp.predict(d_trans_out);

    std::cout << "\nEjemplo de predicción aleatoria:\n";
    std::cout << " - Index: " << idx << "\n";
    std::cout << " - Verdadera: " << true_label << "\n";
    std::cout << " - Predicha: " << pred_label << "\n";
    std::cout << " - Resultado: " << (pred_label == true_label ? "✅ CORRECTO" : "❌ INCORRECTO") << "\n";

    // 8. Limpiar
    free_mnist(train_data);
    free_mnist(test_data);
    cudaFree(d_embed_out);
    cudaFree(d_trans_out);

    std::cout << "\n🏁 Entrenamiento y evaluación completados." << std::endl;
    return 0;
}
