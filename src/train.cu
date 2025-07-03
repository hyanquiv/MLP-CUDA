#include "mlp_cuda.h"
#include "data_loader.h"
#include "constants.h"
#include "cuda_utils.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <random>
#include <cmath>

// Función para barajar los datos
void shuffle_data(MNISTData &data)
{
    int n = data.num_samples;
    int image_size = data.image_size;
    std::vector<int> indices(n);

    // Crear índices secuenciales
    for (int i = 0; i < n; i++)
    {
        indices[i] = i;
    }

    // Barajar los índices
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    // Crear copias temporales
    std::vector<float> shuffled_images(n * image_size);
    std::vector<int> shuffled_labels(n);

    // Reordenar según índices barajados
    for (int i = 0; i < n; i++)
    {
        int orig_idx = indices[i];
        std::copy(data.images.begin() + orig_idx * image_size,
                  data.images.begin() + (orig_idx + 1) * image_size,
                  shuffled_images.begin() + i * image_size);
        shuffled_labels[i] = data.labels[orig_idx];
    }

    // Reemplazar con datos barajados
    data.images = std::move(shuffled_images);
    data.labels = std::move(shuffled_labels);
}

// Función para evaluar el modelo
float evaluate(MLP &model, const MNISTData &data, int max_samples = -1)
{
    int n = data.num_samples;
    if (max_samples > 0 && max_samples < n)
    {
        n = max_samples;
    }

    int correct = 0;
    int image_size = data.image_size;

    for (int i = 0; i < n; i++)
    {
        const float *image = data.images.data() + i * image_size;
        int prediction = model.predict(image);
        if (prediction == data.labels[i])
        {
            correct++;
        }

        // Mostrar progreso cada 1000 muestras
        if (i % 1000 == 0)
        {
            std::cout << "Evaluando: " << i << "/" << n << "\r" << std::flush;
        }
    }
    std::cout << "Evaluación completada. " << std::endl;

    return static_cast<float>(correct) / n;
}

// Función de pérdida de entropía cruzada
float cross_entropy_loss(const float *output, int target, int num_classes)
{
    float loss = 0.0f;
    for (int i = 0; i < num_classes; i++)
    {
        if (i == target)
        {
            loss += -logf(output[i] + 1e-8); // Evitar log(0)
        }
    }
    return loss;
}

// Entrenamiento principal
void train_model(MLP &model, MNISTData &train_data, const MNISTData &test_data)
{
    const int num_train = train_data.num_samples;
    const int num_test = test_data.num_samples;
    const int image_size = train_data.image_size;
    const int output_size = OUTPUT_SIZE;
    const int num_batches = (num_train + BATCH_SIZE - 1) / BATCH_SIZE;

    float current_learning_rate = LEARNING_RATE;

    // Variables para seguimiento de progreso
    float best_test_accuracy = 0.0f;
    std::vector<float> epoch_losses(EPOCHS, 0.0f);

    std::cout << "Iniciando entrenamiento..." << std::endl;
    std::cout << "Configuración:" << std::endl;
    std::cout << " - Épocas: " << EPOCHS << std::endl;
    std::cout << " - Tamaño de lote: " << BATCH_SIZE << std::endl;
    std::cout << " - Tasa de aprendizaje: " << current_learning_rate << std::endl;
    std::cout << " - Muestras de entrenamiento: " << num_train << std::endl;
    std::cout << " - Muestras de prueba: " << num_test << std::endl;

    for (int epoch = 0; epoch < EPOCHS; epoch++)
    {
        shuffle_data(train_data);
        float epoch_loss = 0.0f;

        std::cout << "\nÉpoca " << epoch + 1 << "/" << EPOCHS << std::endl;

        // Entrenamiento por lotes
        for (int batch = 0; batch < num_batches; batch++)
        {
            const int start = batch * BATCH_SIZE;
            const int end = std::min(start + BATCH_SIZE, num_train);
            const int current_batch_size = end - start;

            float batch_loss = 0.0f;
            int correct_in_batch = 0;

            // Procesar cada muestra en el lote
            for (int i = 0; i < current_batch_size; i++)
            {
                const int idx = start + i;
                const float *image = train_data.images.data() + idx * image_size;
                const int label = train_data.labels[idx];

                // Pase hacia adelante
                model.forward(image);

                // Obtener salida y calcular pérdida
                const float *output = model.get_output();
                batch_loss += cross_entropy_loss(output, label, output_size);

                // Calcular precisión en el lote
                int prediction = 0;
                float max_val = 0.0f;
                for (int j = 0; j < output_size; j++)
                {
                    if (output[j] > max_val)
                    {
                        max_val = output[j];
                        prediction = j;
                    }
                }
                if (prediction == label)
                {
                    correct_in_batch++;
                }

                // Pase hacia atrás
                model.backward(image, &label, current_learning_rate);

                // Liberar memoria de salida
                delete[] output;
            }

            // Actualizar pesos (si no se actualizan automáticamente en backward)
            model.update_weights(current_learning_rate);

            // Calcular métricas del lote
            batch_loss /= current_batch_size;
            epoch_loss += batch_loss;
            float batch_accuracy = static_cast<float>(correct_in_batch) / current_batch_size;

            // Mostrar progreso
            if (batch % 10 == 0 || batch == num_batches - 1)
            {
                std::cout << "Lote " << std::setw(4) << batch + 1 << "/" << num_batches
                          << " - Pérdida: " << std::fixed << std::setprecision(4) << batch_loss
                          << " - Prec.: " << std::setprecision(2) << batch_accuracy * 100 << "%"
                          << "\r" << std::flush;
            }
        }

        // Calcular pérdida promedio de la época
        epoch_loss /= num_batches;
        epoch_losses[epoch] = epoch_loss;

        // Evaluar en conjunto de entrenamiento (subconjunto)
        float train_accuracy = evaluate(model, train_data, 5000);

        // Evaluar en conjunto de prueba completo
        float test_accuracy = evaluate(model, test_data);

        // Actualizar mejor precisión
        if (test_accuracy > best_test_accuracy)
        {
            best_test_accuracy = test_accuracy;
        }

        // Mostrar resultados de la época
        std::cout << "\nResumen Época " << epoch + 1 << ":"
                  << "\n  Pérdida: " << std::fixed << std::setprecision(4) << epoch_loss
                  << "\n  Precisión Entrenamiento: " << std::setprecision(2) << train_accuracy * 100 << "%"
                  << "\n  Precisión Prueba: " << std::setprecision(2) << test_accuracy * 100 << "%"
                  << std::endl;

        // Reducción de tasa de aprendizaje
        if (epoch > 0 && epoch_losses[epoch] > epoch_losses[epoch - 1] * 0.95)
        {
            current_learning_rate *= 0.9;
            std::cout << "  Reduciendo tasa de aprendizaje a: " << current_learning_rate << std::endl;
        }
    }

    // Resultados finales
    std::cout << "\nEntrenamiento completado!"
              << "\nMejor precisión en prueba: " << std::setprecision(2) << best_test_accuracy * 100 << "%"
              << std::endl;
}