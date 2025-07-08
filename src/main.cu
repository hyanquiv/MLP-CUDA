#include "data_loader.h"
#include "mlp_cuda.h"
#include "constants.h"
#include "cuda_utils.h"
#include "train.h"
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <iomanip>

int main()
{
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    std::cout << "Iniciando programa de entrenamiento de MLP para MNIST con CUDA" << std::endl;
    std::cout << "============================================================" << std::endl;

    // 1. Cargar datos
    std::cout << "\nCargando conjunto de entrenamiento..." << std::endl;
    MNISTData train_data = load_mnist("/content/MLP-CUDA/data/train-images-idx3-ubyte",
                                      "/content/MLP-CUDA/data/train-labels-idx1-ubyte");

    std::cout << "Cargando conjunto de prueba..." << std::endl;
    MNISTData test_data = load_mnist("/content/MLP-CUDA/data/t10k-images-idx3-ubyte",
                                     "/content/MLP-CUDA/data/t10k-labels-idx1-ubyte");

    std::cout << "\nDatos cargados exitosamente:" << std::endl;
    std::cout << " - Muestras de entrenamiento: " << train_data.num_samples << std::endl;
    std::cout << " - Muestras de prueba: " << test_data.num_samples << std::endl;
    std::cout << " - Dimensiones de imagen: " << train_data.image_size << " pixeles" << std::endl;

    // 2. Normalizar
    std::cout << "\nNormalizando datos..." << std::endl;
    normalize_data(train_data.images);
    normalize_data(test_data.images);

    // 3. Crear modelo
    std::cout << "\nCreando modelo MLP..." << std::endl;
    MLP model(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);

    // 4. Entrenar modelo
    std::cout << "\nIniciando entrenamiento..." << std::endl;
    try
    {
        train_model(model, train_data, test_data);
    }
    catch (const std::exception &e)
    {
        std::cerr << "\nError durante el entrenamiento: " << e.what() << std::endl;
        free_mnist(train_data);
        free_mnist(test_data);
        return 1;
    }

    // 5. Evaluación intensiva en todo el test set
    std::cout << "\nEvaluando en todo el conjunto de prueba..." << std::endl;

    const int num_classes = OUTPUT_SIZE;
    std::vector<int> true_per_class(num_classes, 0);
    std::vector<int> correct_per_class(num_classes, 0);
    std::vector<std::vector<int>> confusion_matrix(num_classes, std::vector<int>(num_classes, 0));

    for (int i = 0; i < test_data.num_samples; ++i)
    {
        const float *image = test_data.images.data() + i * test_data.image_size;
        int true_label = test_data.labels[i];
        int predicted_label = model.predict(image);

        // Conteo por clase
        true_per_class[true_label]++;
        if (predicted_label == true_label)
        {
            correct_per_class[true_label]++;
        }

        // Matriz de confusión
        confusion_matrix[true_label][predicted_label]++;
    }

    // Calcular accuracy balanceada
    float balanced_accuracy = 0.0f;
    for (int i = 0; i < num_classes; ++i)
    {
        if (true_per_class[i] > 0)
        {
            balanced_accuracy += static_cast<float>(correct_per_class[i]) / true_per_class[i];
        }
    }
    balanced_accuracy /= num_classes;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\n✅ Accuracy balanceada: " << balanced_accuracy * 100 << "%" << std::endl;

    // Mostrar matriz de confusión
    std::cout << "\n📊 Matriz de confusión:\n   ";
    for (int i = 0; i < num_classes; ++i)
        std::cout << std::setw(4) << i;
    std::cout << "\n";

    for (int i = 0; i < num_classes; ++i)
    {
        std::cout << std::setw(2) << i << ":";
        for (int j = 0; j < num_classes; ++j)
        {
            std::cout << std::setw(4) << confusion_matrix[i][j];
        }
        std::cout << "\n";
    }

    // 6. Ejemplo de predicción
    int sample_idx = std::rand() % test_data.num_samples;
    const float *sample_image = test_data.images.data() + sample_idx * test_data.image_size;
    int true_label = test_data.labels[sample_idx];
    int predicted_label = model.predict(sample_image);

    std::cout << "\nEjemplo de predicción aleatoria:" << std::endl;
    std::cout << " - Muestra #" << sample_idx << std::endl;
    std::cout << " - Etiqueta verdadera: " << true_label << std::endl;
    std::cout << " - Predicción del modelo: " << predicted_label << std::endl;
    std::cout << " - Resultado: " << (true_label == predicted_label ? "CORRECTO" : "INCORRECTO") << std::endl;

    // 7. Liberar recursos
    free_mnist(train_data);
    free_mnist(test_data);

    std::cout << "\nPrograma completado exitosamente!" << std::endl;
    return 0;
}
