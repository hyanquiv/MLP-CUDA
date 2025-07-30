#ifndef CONSTANTS_H
#define CONSTANTS_H

// MLP clásico
const int INPUT_SIZE = 784;
const int HIDDEN_SIZE = 256;
const int OUTPUT_SIZE = 10;
const int BATCH_SIZE = 64;
const float LEARNING_RATE = 0.001f;
const int EPOCHS = 10;

// Embedding + Transformer
const int IMG_HEIGHT = 28;
const int IMG_WIDTH = 28;
const int PATCH_SIZE = 4;
const int EMBED_DIM = 64;
const int NUM_PATCHES = (IMG_HEIGHT / PATCH_SIZE) * (IMG_WIDTH / PATCH_SIZE);

#endif
