#include "mlp_cuda.h"
#include "data_loader.h"
#include "constants.h"
#include "cuda_utils.h"
#include "embedding.h"
#include "transformer_cuda.h"

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>

// Función para evaluar el modelo
float evaluate(MLP &model, Embedding &embedding, Transformer &transformer, const MNISTData &data, int max_samples = -1)
{
    int n = data.num_samples;
    if (max_samples > 0 && max_samples < n)
        n = max_samples;

    int correct = 0;
    int image_size = data.image_size;

    for (int i = 0; i < n; i++)
    {
        const float *image = data.images.data() + i * image_size;

        // Embedding
        embedding.forward(image);
        float *emb_out = embedding.get_output();

        // Transformer
        transformer.forward(emb_out);
        float *trans_out = transformer.get_output();

        // MLP
        model.forward(trans_out);
        int prediction = model.predict_from_output();

        if (prediction == data.labels[i])
            correct++;

        if (i % 1000 == 0)
            std::cout << "Evaluando: " << i << "/" << n << "\r" << std::flush;
    }
    std::cout << "Evaluación completada.\n";
    return static_cast<float>(correct) / n;
}

float cross_entropy_loss(const float *output, int target, int num_classes)
{
    float loss = 0.0f;
    for (int i = 0; i < num_classes; i++)
        if (i == target)
            loss += -logf(output[i] + 1e-8f); // evitar log(0)
    return loss;
}

void train_model(MLP &model, MNISTData &train_data, const MNISTData &test_data)
{
    const int num_train = train_data.num_samples;
    const int image_size = train_data.image_size;
    const int output_size = OUTPUT_SIZE;
    const int num_batches = (num_train + BATCH_SIZE - 1) / BATCH_SIZE;

    float current_lr = LEARNING_RATE;

    // Parámetros de Embedding y Transformer
    const int patch_size = 4;
    const int embed_dim = 64;
    const int image_w = 28;
    const int image_h = 28;
    const int num_heads = 1;
    const int num_patches = (image_w / patch_size) * (image_h / patch_size); // 49

    Embedding embedding(image_w, image_h, patch_size, embed_dim);
    Transformer transformer(embed_dim, num_heads);

    // Intentar cargar pesos si existen
    try
    {
        load_mlp_weights(model, "/content/MLP-CUDA/pesos/mlp_weights.bin");
        load_transformer_weights(transformer, "/content/MLP-CUDA/pesos/transformer_weights.bin");
        std::cout << "✅ Pesos cargados correctamente.\n";
    }
    catch (...)
    {
        std::cout << "⚠️ No se pudieron cargar pesos. Entrenando desde cero.\n";
    }

    float best_test_accuracy = 0.0f;
    std::vector<float> epoch_losses(EPOCHS, 0.0f);

    std::cout << "Iniciando entrenamiento...\n";
    std::cout << " - Épocas: " << EPOCHS << "\n";
    std::cout << " - Lote: " << BATCH_SIZE << "\n";
    std::cout << " - LR: " << current_lr << "\n";

    for (int epoch = 0; epoch < EPOCHS; epoch++)
    {
        shuffle_data(train_data);
        float epoch_loss = 0.0f;

        std::cout << "\nÉpoca " << epoch + 1 << "/" << EPOCHS << "\n";

        for (int batch = 0; batch < num_batches; batch++)
        {
            int start = batch * BATCH_SIZE;
            int end = std::min(start + BATCH_SIZE, num_train);
            int batch_size = end - start;

            float batch_loss = 0.0f;
            int correct = 0;

            for (int i = 0; i < batch_size; i++)
            {
                int idx = start + i;
                const float *image = train_data.images.data() + idx * image_size;
                int label = train_data.labels[idx];

                // Embedding
                embedding.forward(image);
                float *emb_out = embedding.get_output();

                // Transformer
                transformer.forward(emb_out);
                float *trans_out = transformer.get_output();

                // MLP Forward
                model.forward(trans_out);
                const float *output = model.get_output();

                batch_loss += cross_entropy_loss(output, label, output_size);

                // Accuracy
                int pred = 0;
                float max_val = -1e9f;
                for (int j = 0; j < output_size; j++)
                {
                    if (output[j] > max_val)
                    {
                        max_val = output[j];
                        pred = j;
                    }
                }
                if (pred == label)
                    correct++;

                // Backward
                model.backward(trans_out, &label, current_lr);

                delete[] output;
            }

            model.update_weights(current_lr);

            batch_loss /= batch_size;
            epoch_loss += batch_loss;

            float batch_acc = static_cast<float>(correct) / batch_size;

            if (batch % 10 == 0 || batch == num_batches - 1)
            {
                std::cout << "Lote " << std::setw(4) << batch + 1 << "/" << num_batches
                          << " - Pérdida: " << std::fixed << std::setprecision(4) << batch_loss
                          << " - Precisión: " << std::setprecision(2) << batch_acc * 100 << "%\r" << std::flush;
            }
        }

        epoch_loss /= num_batches;
        epoch_losses[epoch] = epoch_loss;

        float train_acc = evaluate(model, embedding, transformer, train_data, 5000);
        float test_acc = evaluate(model, embedding, transformer, test_data);

        if (test_acc > best_test_accuracy)
            best_test_accuracy = test_acc;

        std::cout << "\nResumen Época " << epoch + 1 << ":"
                  << "\n  Pérdida: " << std::fixed << std::setprecision(4) << epoch_loss
                  << "\n  Precisión Entrenamiento: " << std::setprecision(2) << train_acc * 100 << "%"
                  << "\n  Precisión Test: " << test_acc * 100 << "%\n";

        if (epoch > 0 && epoch_losses[epoch] > epoch_losses[epoch - 1] * 0.95)
        {
            current_lr *= 0.9f;
            std::cout << "  ↓ Reduciendo LR a " << current_lr << "\n";
        }
    }

    std::cout << "\n✅ Entrenamiento finalizado."
              << "\n🏆 Mejor precisión en test: " << std::setprecision(2) << best_test_accuracy * 100 << "%\n";

    // Guardar pesos al final
    std::cout << "\n💾 Guardando pesos del modelo...\n";
    save_mlp_weights(model, "/content/MLP-CUDA/pesos/mlp_weights.bin");
    save_transformer_weights(transformer, "/content/MLP-CUDA/pesos/transformer_weights.bin");
}
