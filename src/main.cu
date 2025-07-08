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

    std::cout << "Iniciando programa de entrenamiento de MLP con embeddings ViT desde CSV\n";
    std::cout << "=======================================================================\n";

    // 1. Cargar datos
    std::cout << "\n📂 Cargando conjunto de entrenamiento desde CSV..." << std::endl;
    MNISTData train_data = load_from_csv(
        "/content/drive/MyDrive/MLP-CUDA/data/vit_mnist_train_embeddings.csv",
        "/content/drive/MyDrive/MLP-CUDA/data/vit_mnist_train_labels.csv");

    std::cout << "📂 Cargando conjunto de prueba desde CSV..." << std::endl;
    MNISTData test_data = load_from_csv(
        "/content/drive/MyDrive/MLP-CUDA/data/vit_mnist_test_embeddings.csv",
        "/content/drive/MyDrive/MLP-CUDA/data/vit_mnist_test_labels.csv");

    std::cout << "\n✅ Datos cargados exitosamente:" << std::endl;
    std::cout << " - Muestras de entrenamiento: " << train_data.num_samples << std::endl;
    std::cout << " - Muestras de prueba: " << test_data.num_samples << std::endl;
    std::cout << " - Tamaño de embedding (input size): " << train_data.image_size << std::endl;

    // 2. Normalizar (si quieres escalar los valores del embedding)
    std::cout << "\n⚙️ Normalizando datos (opcional)..." << std::endl;
    normalize_data(train_data.images);
    normalize_data(test_data.images);

    // 3. Crear modelo
    std::cout << "\n🚧 Creando modelo MLP..." << std::endl;
    MLP model(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);

    // 4. Entrenar modelo
    std::cout << "\n🚀 Iniciando entrenamiento..." << std::endl;
    try
    {
        train_model(model, train_data, test_data);
    }
    catch (const std::exception &e)
    {
        std::cerr << "\n❌ Error durante el entrenamiento: " << e.what() << std::endl;
        free_mnist(train_data);
        free_mnist(test_data);
        return 1;
    }

    // 5. Evaluación en todo el test set
    std::cout << "\n🧪 Evaluando en conjunto de prueba..." << std::endl;

    const int num_classes = OUTPUT_SIZE;
    std::vector<int> true_per_class(num_classes, 0);
    std::vector<int> correct_per_class(num_classes, 0);
    std::vector<std::vector<int>> confusion_matrix(num_classes, std::vector<int>(num_classes, 0));

    for (int i = 0; i < test_data.num_samples; ++i)
    {
        const float *image = test_data.images.data() + i * test_data.image_size;
        int true_label = test_data.labels[i];
        int predicted_label = model.predict(image);

        true_per_class[true_label]++;
        if (predicted_label == true_label)
            correct_per_class[true_label]++;
        confusion_matrix[true_label][predicted_label]++;
    }

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

    // 6. Ejemplo de predicción aleatoria
    int sample_idx = std::rand() % test_data.num_samples;
    const float *sample_image = test_data.images.data() + sample_idx * test_data.image_size;
    int true_label = test_data.labels[sample_idx];
    int predicted_label = model.predict(sample_image);

    std::cout << "\n🔍 Ejemplo de predicción aleatoria:" << std::endl;
    std::cout << " - Muestra #" << sample_idx << std::endl;
    std::cout << " - Etiqueta verdadera: " << true_label << std::endl;
    std::cout << " - Predicción del modelo: " << predicted_label << std::endl;
    std::cout << " - Resultado: " << (true_label == predicted_label ? "CORRECTO" : "INCORRECTO") << std::endl;

    // 7. Liberar recursos
    free_mnist(train_data);
    free_mnist(test_data);

    std::cout << "\n✅ Programa completado exitosamente!" << std::endl;
    return 0;
}
