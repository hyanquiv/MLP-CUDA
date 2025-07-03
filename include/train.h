#ifndef TRAIN_H
#define TRAIN_H

#include "mlp_cuda.h"
#include "data_loader.h"

// Cambiar el segundo parámetro a referencia no constante
void train_model(MLP &model, MNISTData &train_data, const MNISTData &test_data);
float evaluate(MLP &model, const MNISTData &data, int max_samples = -1);

#endif