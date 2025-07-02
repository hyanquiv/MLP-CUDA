// include/train.h
#ifndef TRAIN_H
#define TRAIN_H

#include "mlp_cuda.h"
#include "data_loader.h"

void train_model(MLP &model, const MNISTData &train_data, const MNISTData &test_data);
float evaluate(MLP &model, const MNISTData &data, int max_samples = -1);

#endif