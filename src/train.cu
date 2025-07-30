#include "train.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iomanip>

void train_model(Embedding &embedding, Transformer &transformer, MLP &model,
                 const std::vector<std::vector<float>> &train_images,
                 const std::vector<int> &train_labels,
                 const std::vector<std::vector<float>> &val_images,
                 const std::vector<int> &val_labels,
                 int epochs, float learning_rate, const std::string &csv_path)
{
    std::ofstream log_file(csv_path);
    log_file << "epoch,train_loss,train_acc,val_loss,val_acc\n";

    int num_classes = 10;
    int num_train = train_images.size();
    int num_val = val_images.size();

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        float train_loss = 0.0f;
        int train_correct = 0;

        for (int i = 0; i < num_train; ++i)
        {
            const float *image = train_images[i].data();
            int label = train_labels[i];

            // --- FORWARD PASS ---
            float *emb_out = embedding.forward(image);
            float *trans_out = transformer.forward(emb_out);
            model.forward(trans_out);

            const float *output = model.get_output();

            // --- LOSS ---
            float pred_loss = -std::log(std::max(output[label], 1e-7f));
            train_loss += pred_loss;

            // --- ACCURACY ---
            int pred = model.predict(trans_out);
            if (pred == label)
                train_correct++;

            // --- BACKWARD PASS ---
            model.backward(trans_out, &label, learning_rate);
            float *d_mlp_input = model.get_input_gradient();

            transformer.backward(emb_out, d_mlp_input, learning_rate);
            float *d_trans_input = transformer.get_input_gradient();

            embedding.backward(image, d_trans_input, learning_rate);
        }

        float train_acc = static_cast<float>(train_correct) / num_train;
        train_loss /= num_train;

        // --- VALIDATION ---
        float val_loss = 0.0f;
        int val_correct = 0;

        for (int i = 0; i < num_val; ++i)
        {
            const float *image = val_images[i].data();
            int label = val_labels[i];

            float *emb_out = embedding.forward(image);
            float *trans_out = transformer.forward(emb_out);
            model.forward(trans_out);

            const float *output = model.get_output();
            float pred_loss = -std::log(std::max(output[label], 1e-7f));
            val_loss += pred_loss;

            int pred = model.predict(trans_out);
            if (pred == label)
                val_correct++;
        }

        float val_acc = static_cast<float>(val_correct) / num_val;
        val_loss /= num_val;

        // --- LOGGING ---
        std::cout << "Epoch [" << epoch + 1 << "/" << epochs << "] "
                  << "Train Loss: " << train_loss << ", Train Acc: " << train_acc
                  << ", Val Loss: " << val_loss << ", Val Acc: " << val_acc << std::endl;

        log_file << epoch + 1 << "," << train_loss << "," << train_acc << ","
                 << val_loss << "," << val_acc << "\n";
    }

    log_file.close();
}
