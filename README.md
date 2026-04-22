# Sistema de Reconocimiento de Palabras Habladas

Un sistema de reconocimiento automático de palabras utilizando **LPC (Linear Predictive Coding)** y **Cuantización Vectorial (VQ)** con clustering **LBG** y distancia **Itakura-Saito**.

## Descripción

Este proyecto implementa un clasificador de palabras habladas en Python. El flujo es:

1. **Preprocesamiento**: Carga WAV → pre-énfasis → detección de actividad de voz (VAD)
2. **Extracción de características**: LPC orden 12 → conversión a LSF
3. **Entrenamiento**: Clustering LBG en espacio LSF (10 archivos por palabra)
4. **Reconocimiento**: Clasificación usando distancia Itakura-Saito en espacio LPC (5 archivos por palabra)
5. **Evaluación**: Matrices de confusión para codebooks de tamaño 16, 32, 64

## Requisitos

- Python 3.x
- Librerías: `numpy`, `scipy`, `librosa`, `matplotlib`, `scikit-learn`, `soundfile`

## Instalación

```bash
# Clonar o descargar el repositorio
cd reconocimiento-voz

# Instalar dependencias
pip install -r requirements.txt
```

## Estructura de archivos

```
reconocimiento-voz/
├── preprocessing.py          # Pre-énfasis, Hamming windowing, VAD
├── features.py               # LPC (Levinson-Durbin), LSF
├── vq.py                     # Algoritmo LBG, Itakura-Saito, codebooks
├── recognition.py            # Clasificación, matrices de confusión, plots
├── main.py                   # Script principal con CLI
├── generate_dummy_data.py    # Generador de datos sintéticos (para testing)
├── inicio_fin.m              # Referencia: código original MATLAB para VAD
├── requirements.txt
└── README.md

data/                          # (Estructura esperada para archivos WAV)
├── start/    → start_01.wav, start_02.wav, ..., start_15.wav
├── stop/     → stop_01.wav, stop_02.wav, ..., stop_15.wav
├── lift/
├── drop/
├── left/
├── right/
├── up/
├── down/
├── go/
└── back/

output/                        # (Generado)
├── confusion_16.png
├── confusion_32.png
├── confusion_64.png
├── vad_start.png
├── vad_stop.png
└── codebooks.pkl
```

## Uso

### 1. Preparar datos

Coloca tus grabaciones en carpetas siguiendo la estructura anterior:

```
data/
  start/    start_01.wav ... start_15.wav
  stop/     stop_01.wav  ... stop_15.wav
  ...
```

**Alternativa**: Generar datos sintéticos para testing (15 archivos por palabra, formato WAV a 16 kHz):

```bash
python generate_dummy_data.py
```

### 2. Ejecutar el sistema completo

```bash
python main.py --data_dir data --codebook_sizes 16 32 64
```

**Salida esperada**:
- 3 matrices de confusión (PNG): `output/confusion_16.png`, `confusion_32.png`, `confusion_64.png`
- Codebooks entrenados: `output/codebooks.pkl`
- Accuracy global y por palabra (consola)

### 3. Opciones de línea de comando

```bash
# Visualizar detección VAD para las primeras 2 palabras
python main.py --data_dir data --vad_demo

# Cambiar split train/test (default: 1-10 train, 11-15 test)
python main.py --data_dir data --train_files 1 2 3 4 5 6 7 8 9 10 --test_files 11 12 13 14 15

# Reusar codebooks ya entrenados
python main.py --data_dir data --skip_training

# Cambiar tamaños de codebooks
python main.py --data_dir data --codebook_sizes 32 64 128

# Cambiar orden LPC
python main.py --data_dir data --lpc_order 16
```

## Detalles técnicos

### Preprocesamiento (preprocessing.py)

**VAD (Voice Activity Detection)** — traducción fiel de `inicio_fin.m`:
- Energía lineal normalizada: E[i] = Σ(frame²) / frame_len
- Zero Crossing Rate: ZCR[i] = Σ(|diff(sign(frame))|) / 2 / frame_len
- Umbrales:
  - ZCR_thresh = 0.08 × max(ZCR)
  - Energy_thresh = 0.03 × max(Energy)
- Frame de voz si: **ZCR > umbral AND Energy > umbral** (ambas condiciones)
- Parámetros VAD: 320 muestras (20 ms), hop 160 (10 ms) a 16 kHz

**Framing para LPC**:
- Ventana Hamming: 320 puntos
- Hop: 128 muestras (8 ms a 16 kHz)

**Pre-énfasis**:
- Filtro H_p(z) = 1 - 0.95z⁻¹
- y[n] = x[n] - 0.95 × x[n-1]

### Características (features.py)

**LPC** (Linear Predictive Coding):
- Orden: 12 coeficientes
- Algoritmo: Levinson-Durbin (autocorrelación vía FFT)

**LSF** (Line Spectral Frequencies):
- Conversión desde LPC mediante polinomios simétricos P(z) y antisimétricos Q(z)
- Raíces sobre el círculo unitario

### Cuantización Vectorial (vq.py)

**LBG** (Linde-Buzo-Gray):
- Clustering iterativo en espacio LSF (distancia Euclidea)
- Splitting binario: 1 → 2 → 4 → 8 → 16/32/64 codevectores
- Lloyd-Max refinement en cada paso

**Itakura-Saito**:
- Distancia espectral entre frames LPC
- Calculada vía razón H1/H2 de espectros de potencia en 512 puntos
- Fórmula: d_IS = mean(H1/H2 - log(H1/H2) - 1)

**Estrategia de entrenamiento**:
1. Entrenar codebooks LSF con LBG
2. Asignar training vectors LSF a clusters
3. Promediar los correspondientes LPC vectors por cluster → codebook LPC
4. Guardar ambas representaciones

### Reconocimiento (recognition.py)

**Clasificación**:
- Input: frames de test (LPC + gains)
- Para cada palabra: distancia IS promedio al codebook LPC
- Predecir: palabra con distancia mínima

**Matriz de confusión**:
- Filas: palabras verdaderas
- Columnas: palabras predichas
- Celda [i,j]: # de archivos de palabra i clasificados como j

**Métricas**:
- Accuracy global: correcciones / total
- Accuracy por palabra: correcciones de esa palabra / total de esa palabra

## Parámetros por defecto

| Parámetro | Valor | Descripción |
|---|---|---|
| `sr` | 16000 | Frecuencia de muestreo (Hz) |
| `lpc_order` | 12 | Orden de coeficientes LPC |
| `preemph_coeff` | 0.95 | Coeficiente de pre-énfasis |
| `codebook_sizes` | [16, 32, 64] | Tamaños a probar |
| `train_files` | 1-10 | Índices para entrenamiento |
| `test_files` | 11-15 | Índices para testing |
| `epsilon` | 0.01 | Tolerancia de convergencia LBG |
| `max_iter` | 100 | Iteraciones máx. Lloyd por step |

## Ejemplo de ejecución

```bash
$ python main.py --data_dir data --vad_demo --codebook_sizes 16 32
Palabras encontradas: ['start', 'stop', 'lift', 'drop', 'left', 'right', 'up', 'down', 'go', 'back']
Tamaños de codebook: [16, 32]
Índices train: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  |  Índices test: [11, 12, 13, 14, 15]

Generating VAD demo plots...
  Saved: output/vad_start.png
  Saved: output/vad_stop.png

--- Training phase ---

  Palabra: start
  Training vectors: (400, 12)
  Training codebook size=16 ... done
  Training codebook size=32 ... done
  
  [continuación con otras palabras...]
  
Codebooks saved to output/codebooks.pkl

--- Recognition phase (using Itakura-Saito distance) ---
  [16cv] start_11.wav: true=start     pred=start
  [32cv] start_11.wav: true=start     pred=start
  [16cv] start_12.wav: true=start     pred=start
  ...
  
Saved: output/confusion_16.png
=== Codebook size 16 ===
  Global accuracy: 72.0%
  start     : 80.0%
  stop      : 60.0%
  ...

Saved: output/confusion_32.png
=== Codebook size 32 ===
  Global accuracy: 78.0%
  ...
```

## Resultados esperados

Con datos reales (grabaciones claras):
- **Codebook 16**: ~70-75% accuracy
- **Codebook 32**: ~80-85% accuracy
- **Codebook 64**: ~85-90% accuracy

Con datos sintéticos: resultados más altos (señales limpias sin ruido real).

## Referencias

- Rabiner, L. R., & Sambur, M. R. (1975). "An algorithm for determining the endpoints of isolated utterances." *Bell System Technical Journal*, 54(2), 297-315.
- Itakura, F. (1975). "Minimum prediction residual principle applied to speech recognition." *IEEE Trans. Acoustics, Speech, Signal Processing*, 23(1), 67-72.
- Linde, Y., Buzo, A., & Gray, R. M. (1980). "An algorithm for vector quantizer design." *IEEE Trans. Communications*, 28(1), 84-95.

## Notas

- El archivo `generate_dummy_data.py` es solo para testing. Eliminarlo antes de usar datos reales.
- El archivo `inicio_fin.m` es la referencia original MATLAB para validación.
- La distancia Itakura-Saito se calcula espectralmente (FFT). Para aplicaciones de tiempo real, considerar aproximaciones más rápidas.

## Licencia

Proyecto académico. Libre para uso educativo.

---

**Autor**: Implementación Python de sistema clásico de reconocimiento de voz (LPC + VQ)  
**Versión**: 1.0  
**Última actualización**: 2026-04-22
