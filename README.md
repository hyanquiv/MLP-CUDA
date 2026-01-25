# MLP-CUDA

Una implementación de **Red Neuronal Multicapa (MLP)** en **CUDA** para clasificación de dígitos MNIST usando embeddings de Vision Transformer (ViT).

## Descripción

Este proyecto implementa un perceptrón multicapa completamente optimizado con CUDA para aprovechar la potencia de las GPU NVIDIA. El modelo se entrena con embeddings de ViT (Vision Transformer) del dataset MNIST, permitiendo clasificación rápida y eficiente de dígitos manuscritos.

**Características principales:**
- Implementación completamente en CUDA para máximo rendimiento
- Arquitectura MLP configurable con múltiples capas
- Funciones de activación: ReLU y Softmax
- Backpropagation GPU-acelerado
- Soporte para embeddings ViT (768 dimensiones)
- Métricas de evaluación: Precision, Recall, F1-Score
- Matriz de confusión y balanced accuracy
- Gestión optimizada de memoria GPU

## Estructura del Proyecto

```
├── include/              # Headers del proyecto
│   ├── mlp_cuda.h       # Clase principal del modelo MLP
│   ├── train.h          # Funciones de entrenamiento y evaluación
│   ├── activation.h     # Kernels de funciones de activación
│   ├── data_loader.h    # Carga de datos desde CSV
│   ├── cuda_utils.h     # Utilidades CUDA
│   └── constants.h      # Constantes del modelo
├── src/                 # Archivos de implementación (.cu)
│   ├── main.cu          # Punto de entrada principal
│   ├── mlp_cuda.cu      # Implementación de la clase MLP
│   ├── train.cu         # Lógica de entrenamiento
│   ├── activation.cu    # Kernels de activación (GPU)
│   ├── data_loader.cu   # Carga y normalización de datos
│   └── cuda_utils.cu    # Funciones CUDA auxiliares
├── data/                # Dataset MNIST
│   ├── train-images-idx3-ubyte
│   ├── train-labels-idx1-ubyte
│   ├── t10k-images-idx3-ubyte
│   └── t10k-labels-idx1-ubyte
├── Makefile            # Sistema de compilación
└── README.md           # Este archivo
```

## Configuración del Modelo

Las constantes principales se definen en [constants.h](include/constants.h):

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| INPUT_SIZE | 768 | Dimensión de embeddings ViT |
| HIDDEN_SIZE | 256 | Neuronas en capa oculta |
| OUTPUT_SIZE | 10 | Número de clases (dígitos 0-9) |
| BATCH_SIZE | 64 | Tamaño de batch |
| LEARNING_RATE | 0.001 | Tasa de aprendizaje |
| EPOCHS | 10 | Número de épocas |

## Compilación e Instalación

### Requisitos
- **CUDA Toolkit** (versión 11.0 o superior)
- **NVIDIA GPU** compatible (arquitectura sm_75 o superior recomendado)
- **GNU Make**
- **Compilador C++11 o superior**

### Pasos de compilación

1. **Clonar el repositorio:**
```bash
git clone https://github.com/tu-usuario/MLP-CUDA.git
cd MLP-CUDA
```

2. **Verificar/Actualizar arquitectura GPU en Makefile:**
```makefile
CFLAGS = -std=c++11 -O3 -arch=sm_75  # Cambiar sm_75 por tu arquitectura
```

Para encontrar tu arquitectura GPU:
```bash
nvidia-smi --query-gpu=name --format=csv
# Luego usar: sm_35, sm_50, sm_60, sm_70, sm_75, sm_86, sm_89, sm_90, etc.
```

3. **Compilar el proyecto:**
```bash
make clean
make
```

4. **Ejecutar el programa:**
```bash
make run
# O directamente:
./build/mnist_mlp
```

## Uso

### Estructura de datos esperada

El programa carga datos desde archivos CSV:
- **Features CSV**: Matriz de (N_samples × 768) con embeddings ViT
- **Labels CSV**: Vector de (N_samples) con etiquetas 0-9

### Flujo de ejecución

1. **Carga de datos**: Lee embeddings ViT desde CSV para conjuntos train y test
2. **Normalización**: Normaliza los datos (opcional)
3. **Inicialización**: Crea el modelo MLP
4. **Entrenamiento**: Ejecuta 10 épocas con backpropagation
5. **Evaluación**: Calcula métricas en el conjunto de prueba
6. **Métricas**: Muestra Precision, Recall, F1-Score por clase y matriz de confusión

### Ejemplo de salida

```
Iniciando programa de entrenamiento de MLP con embeddings ViT desde CSV
=======================================================================

Cargando conjunto de entrenamiento desde CSV...
Cargando conjunto de prueba desde CSV...

Datos cargados exitosamente:
 - Muestras de entrenamiento: 60000
 - Muestras de prueba: 10000
 - Tamaño de embedding (input size): 768

Normalizando datos (opcional)...

Creando modelo MLP...

Iniciando entrenamiento...

Evaluando en conjunto de prueba...

Métricas por clase:
Clase | Precision | Recall  | F1-Score
[Resultados por clase...]
```

## Componentes Principales

### Clase MLP (`mlp_cuda.h`)
- Constructor: Configura arquitectura de red
- `forward()`: Propagación hacia adelante (GPU)
- `backward()`: Backpropagation (GPU)
- `update_weights()`: Actualización de pesos
- `predict()`: Predicción de clase

### Funciones de activación (`activation.h`)
- **ReLU**: Rectified Linear Unit (capa oculta)
- **Softmax**: Para capa de salida
- Versiones forward y backward para training

### Carga de datos (`data_loader.h`)
- `load_from_csv()`: Carga embeddings y etiquetas
- `normalize_data()`: Normaliza características
- `shuffle_data()`: Mezcla datos de entrenamiento
- `free_mnist()`: Libera memoria

### Utilities CUDA (`cuda_utils.h`)
- `cuda_alloc()`: Asignación de memoria GPU
- `cuda_free()`: Liberación de memoria GPU
- `copy_to_device()`: Host → Device
- `copy_to_host()`: Device → Host
- `CHECK_CUDA()`: Macro para manejo de errores

## Resultados Esperados

Entrenando con embeddings ViT, el modelo típicamente alcanza:
- **Accuracy**: 95%+ en datos de test
- **Balanced Accuracy**: Distribución equitativa por clase
- **Macro F1-Score**: >0.95 con convergencia rápida

## Detalles Técnicos

### Kernels CUDA principales
- `init_weights_kernel`: Inicialización Xavier
- `linear_forward_kernel`: Multiplicación matriz-vector + bias
- `weight_gradient_kernel`: Cálculo de gradientes
- `relu_forward/backward`: Activación ReLU
- `softmax`: Normalización de salida

### Optimizaciones
- Operaciones batching GPU
- Memoria coalescida para máximo ancho de banda
- Kernels optimizados para bloques de 256 threads
- Uso de CUBLAS para operaciones lineales (cuando aplicable)

## Troubleshooting

### Error: CUDA error - device synchronization
```bash
# Asegúrate que el programa tenga acceso a la GPU
nvidia-smi
# Verifica tu GPU es compatible con CUDA
```

### Error: Incompatible GPU architecture
```bash
# Actualiza CFLAGS en Makefile con tu arquitectura correcta
nvcc --version
```

### Memory allocation errors
- Reduce BATCH_SIZE en constants.h
- Reduce HIDDEN_SIZE
- Asegúrate que tu GPU tiene suficiente memoria

## Licencia

Este proyecto está bajo la licencia [LICENSE](LICENSE).

## Autor

Proyecto educativo de implementación CUDA de redes neuronales.

---

**Nota**: Los archivos de datos MNIST incluyen tanto el formato binario original como soporte para embeddings ViT en CSV.
