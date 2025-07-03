#include "data_loader.h"
#include "mlp_cuda.h"
#include "constants.h"
#include "cuda_utils.h"
#include "train.h"
#include <iostream>
#include <cstdlib>
#include <ctime>

int main()
{
    // Inicializar semilla aleatoria
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    std::cout << "Iniciando programa de entrenamiento de MLP para MNIST con CUDA" << std::endl;
    std::cout << "============================================================" << std::endl;

    // 1. Cargar datos
    std::cout << "\nCargando conjunto de entrenamiento..." << std::endl;
    MNISTData train_data = load_mnist("data/train-images-idx3-ubyte",
                                      "data/train-labels-idx1-ubyte");

    std::cout << "Cargando conjunto de prueba..." << std::endl;
    MNISTData test_data = load_mnist("data/t10k-images-idx3-ubyte",
                                     "data/t10k-labels-idx1-ubyte");

    // Mostrar estadísticas de los datos
    std::cout << "\nDatos cargados exitosamente:" << std::endl;
    std::cout << " - Muestras de entrenamiento: " << train_data.num_samples << std::endl;
    std::cout << " - Muestras de prueba: " << test_data.num_samples << std::endl;
    std::cout << " - Dimensiones de imagen: " << train_data.image_size << " pixeles" << std::endl;

    // 2. Preprocesamiento de datos
    std::cout << "\nNormalizando datos..." << std::endl;
    normalize_data(train_data.images);
    normalize_data(test_data.images);

    // 3. Crear modelo MLP
    std::cout << "\nCreando modelo MLP..." << std::endl;
    std::cout << "Arquitectura: " << INPUT_SIZE << "-" << HIDDEN_SIZE << "-" << OUTPUT_SIZE << std::endl;

    MLP model(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);

    // 4. Entrenar el modelo
    std::cout << "\nIniciando entrenamiento..." << std::endl;
    std::cout << "Configuración:" << std::endl;
    std::cout << " - Épocas: " << EPOCHS << std::endl;
    std::cout << " - Tamaño de lote: " << BATCH_SIZE << std::endl;
    std::cout << " - Tasa de aprendizaje: " << LEARNING_RATE << std::endl;

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

    // 5. Evaluación final
    std::cout << "\nEvaluando modelo final en conjunto de prueba completo..." << std::endl;
    float final_accuracy = evaluate(model, test_data);
    std::cout << "\nPrecisión final: " << final_accuracy * 100 << "%" << std::endl;

    // 6. Ejemplo de predicción
    std::cout << "\nRealizando predicción de ejemplo..." << std::endl;
    int sample_idx = std::rand() % test_data.num_samples;
    const float *sample_image = test_data.images.data() + sample_idx * test_data.image_size;
    int true_label = test_data.labels[sample_idx];
    int predicted_label = model.predict(sample_image);

    std::cout << " - Muestra #" << sample_idx << std::endl;
    std::cout << " - Etiqueta verdadera: " << true_label << std::endl;
    std::cout << " - Predicción del modelo: " << predicted_label << std::endl;
    std::cout << " - Resultado: " << (true_label == predicted_label ? "CORRECTO" : "INCORRECTO") << std::endl;

    // 7. Liberar recursos
    std::cout << "\nLiberando recursos..." << std::endl;
    free_mnist(train_data);
    free_mnist(test_data);

    std::cout << "\nPrograma completado exitosamente!" << std::endl;
    return 0;
}