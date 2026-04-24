# Sistema de Reconocimiento de Palabras Habladas

Un sistema de reconocimiento automático de palabras utilizando **LPC (Linear Predictive Coding)** y **Cuantización Vectorial (VQ)** con clustering **LBG** y distancia **Itakura-Saito**.

## Descripción

Este proyecto implementa un clasificador de palabras habladas en Python. El flujo completo es:

1. **Preprocesamiento**: Carga WAV → pre-énfasis → detección de actividad de voz (VAD) → recorte de señal
2. **Extracción de características**: LPC orden 12 → conversión a LSF
3. **Entrenamiento**: Clustering LBG en espacio LSF (10 archivos por palabra)
4. **Reconocimiento**: Clasificación usando distancia Itakura-Saito en espacio LPC (5 archivos por palabra)
5. **Evaluación**: Matrices de confusión, comparativa de accuracy y análisis de confusiones por codebook (16, 32, 64)

## Palabras reconocidas

El sistema fue entrenado y evaluado con las siguientes 10 palabras de navegación para puzzlebot:

`abajo` · `alto` · `arriba` · `avanzar` · `bajar` · `derecha` · `inicio` · `izquierda` · `retroceder` · `subir`

## Grabaciones

Se cuenta con datos de dos hablantes, 15 grabaciones por palabra cada uno:

- `data_jorge/` — 150 archivos WAV (10 palabras × 15 grabaciones)
- `data_valeria/` — 150 archivos WAV (10 palabras × 15 grabaciones)

Formato: mono, 16 kHz, nombrados `<palabra>_01.wav` … `<palabra>_15.wav`.

## Requisitos

- Python 3.x
- Librerías: `numpy`, `scipy`, `librosa`, `matplotlib`, `scikit-learn`, `soundfile`

## Instalación

```bash
git clone https://github.com/Gees14/reconocimiento-voz.git
cd reconocimiento-voz
pip install -r requirements.txt
```

## Estructura del repositorio

```
reconocimiento-voz/
├── preprocessing.py          # Pre-énfasis, Hamming windowing, VAD
├── features.py               # LPC (Levinson-Durbin), LSF
├── vq.py                     # Algoritmo LBG, distancia Itakura-Saito, codebooks
├── recognition.py            # Clasificación, métricas, todas las gráficas
├── main.py                   # Script principal con CLI
├── inicio_fin.m              # Referencia: código original MATLAB para VAD
├── generate_dummy_data.py    # Generador de datos sintéticos (solo para testing)
├── requirements.txt
├── README.md
│
├── data_jorge/               # Grabaciones del hablante Jorge
│   ├── abajo/   → abajo_01.wav … abajo_15.wav
│   ├── alto/
│   ├── arriba/
│   ├── avanzar/
│   ├── bajar/
│   ├── derecha/
│   ├── inicio/
│   ├── izquierda/
│   ├── retroceder/
│   └── subir/
│
├── data_valeria/             # Grabaciones del hablante Valeria (misma estructura)
│
├── output_jorge/             # Resultados generados con data_jorge
│   ├── vad_abajo.png         # Gráfica VAD (señal + energía + ZCR) por palabra
│   ├── vad_alto.png
│   ├── … (10 gráficas VAD)
│   ├── confusion_16.png      # Matriz de confusión codebook=16
│   ├── confusion_32.png      # Matriz de confusión codebook=32
│   ├── confusion_64.png      # Matriz de confusión codebook=64
│   ├── accuracy_by_size.png  # Comparativa de accuracy entre los 3 codebooks
│   ├── top_confusions.png    # Top palabras que más se confunden entre sí
│   └── codebooks.pkl         # Codebooks entrenados (para --skip_training)
│
└── output_valeria/           # Resultados generados con data_valeria (misma estructura)
```

## Uso

### Ejecutar el pipeline completo

```bash
# Con grabaciones de Jorge
python main.py --data_dir data_jorge --output_dir output_jorge

# Con grabaciones de Valeria
python main.py --data_dir data_valeria --output_dir output_valeria
```

El pipeline genera automáticamente:
- Gráficas VAD para **todas** las palabras
- 3 matrices de confusión (codebooks 16, 32, 64)
- Gráfica comparativa de accuracy por tamaño de codebook
- Gráfica de top palabras que más se confunden
- Recomendación del codebook óptimo en consola

### Reusar codebooks ya entrenados (más rápido)

```bash
python main.py --data_dir data_jorge --output_dir output_jorge --skip_training
```

### Otras opciones

```bash
# Cambiar split train/test (default: 1-10 train, 11-15 test)
python main.py --data_dir data_jorge --train_files 1 2 3 4 5 6 7 8 9 10 --test_files 11 12 13 14 15

# Cambiar tamaños de codebook
python main.py --data_dir data_jorge --codebook_sizes 32 64 128

# Cambiar orden LPC (default: 12)
python main.py --data_dir data_jorge --lpc_order 16
```

## Outputs generados

### Gráfica VAD (`vad_<palabra>.png`)

Tres paneles para cada palabra:
- **Panel 1**: Señal de audio con marcadores de inicio/fin detectados por VAD
- **Panel 2**: Energía normalizada por frame con umbral (3% del máximo)
- **Panel 3**: Zero Crossing Rate por frame con umbral (8% del máximo)

### Matrices de confusión (`confusion_16/32/64.png`)

Heatmap donde:
- Filas = palabra real (true label)
- Columnas = palabra predicha
- Diagonal = clasificaciones correctas
- Fuera de diagonal = errores y con qué palabra se confundió

### Comparativa de accuracy (`accuracy_by_size.png`)

Gráfica de barras con la precisión global obtenida con cada tamaño de codebook (16, 32, 64), con el porcentaje anotado sobre cada barra.

### Top confusiones (`top_confusions.png`)

Gráfica de barras horizontales con los pares de palabras que más se confunden entre sí, agregando los errores de los tres codebooks.

## Salida en consola

```
Words found: ['abajo', 'alto', 'arriba', 'avanzar', 'bajar', 'derecha', 'inicio', 'izquierda', 'retroceder', 'subir']
Codebook sizes: [16, 32, 64]
Train indices: [1..10]  |  Test indices: [11..15]

Generating VAD plots for all words...
  Saved: output_jorge/vad_abajo.png
  ...

--- Training phase ---
  Word: abajo
  Training vectors: (N, 12)
  Training codebook size=16 ... done
  Training codebook size=32 ... done
  Training codebook size=64 ... done
  ...

--- Recognition phase (using Itakura-Saito distance) ---
  [16cv] abajo_11.wav: true=abajo      pred=abajo
  [32cv] abajo_11.wav: true=abajo      pred=abajo
  ...

=== Codebook tamaño 16 ===
  Precisión global: 74.0%
  abajo       : 80.0%
  alto        : 60.0%
  ...
  Top confusiones:
    arriba     → avanzar  : 2 veces
    ...

=== Codebook tamaño 32 ===
  Precisión global: 82.0%
  ...

=== Codebook tamaño 64 ===
  Precisión global: 86.0%
  ...

============================================================
RECOMENDACIÓN — Tamaño de codebook óptimo
============================================================
  Codebook  16: 74.0%
  Codebook  32: 82.0%
  Codebook  64: 86.0%  ←  ÓPTIMO

  El codebook de tamaño 64 obtuvo la mayor precisión (86.0%).
============================================================
```

## Detalles técnicos

### Preprocesamiento (`preprocessing.py`)

**Pre-énfasis**:
- Filtro: H_p(z) = 1 − 0.95z⁻¹
- y[n] = x[n] − 0.95 × x[n−1]

**VAD (Voice Activity Detection)** — traducción fiel de `inicio_fin.m`:
- Energía lineal normalizada: E[i] = Σ(frame²) / frame_len
- Zero Crossing Rate: ZCR[i] = Σ(|diff(sign(frame))|) / 2 / frame_len
- Umbrales: ZCR_thr = 0.08 × max(ZCR), Energy_thr = 0.03 × max(Energy)
- Frame de voz si: **ZCR > umbral AND Energy > umbral**
- Parámetros VAD: frame 320 muestras (20 ms), hop 160 muestras (10 ms)

**Framing para LPC**:
- Ventana Hamming: 320 puntos
- Hop: 128 muestras (8 ms a 16 kHz)

### Características (`features.py`)

**LPC** (Linear Predictive Coding):
- Orden: 12 coeficientes
- Algoritmo: Levinson-Durbin (autocorrelación vía FFT)

**LSF** (Line Spectral Frequencies):
- Conversión desde LPC mediante polinomios simétricos P(z) y antisimétricos Q(z)
- Raíces sobre el círculo unitario

### Cuantización Vectorial (`vq.py`)

**LBG** (Linde-Buzo-Gray):
- Clustering iterativo en espacio LSF (distancia Euclidea)
- Splitting binario: 1 → 2 → 4 → … → 16/32/64 codevectores
- Lloyd-Max refinement en cada paso (ε = 0.01, máx. 100 iteraciones)

**Itakura-Saito**:
- Distancia espectral entre frames LPC
- Calculada vía razón de espectros de potencia en 512 puntos FFT
- d_IS = mean(H1/H2 − log(H1/H2) − 1)

### Reconocimiento (`recognition.py`)

**Clasificación**: Para cada archivo de test, se computa la distancia IS promedio al codebook LPC de cada palabra. Se predice la palabra con distancia mínima.

**Métricas**:
- Accuracy global: aciertos / total de archivos de test
- Accuracy por palabra: aciertos de esa palabra / total de esa palabra
- Top confusiones: pares (verdadero → predicho) con mayor frecuencia de error

## Parámetros por defecto

| Parámetro | Valor | Descripción |
|---|---|---|
| `sr` | 16000 | Frecuencia de muestreo (Hz) |
| `lpc_order` | 12 | Orden de coeficientes LPC |
| `preemph_coeff` | 0.95 | Coeficiente de pre-énfasis |
| `codebook_sizes` | [16, 32, 64] | Tamaños de codebook a evaluar |
| `train_files` | 1–10 | Índices para entrenamiento |
| `test_files` | 11–15 | Índices para evaluación |
| `epsilon` | 0.01 | Tolerancia de convergencia LBG |
| `max_iter` | 100 | Iteraciones máx. Lloyd por step |

## Referencias

- Rabiner, L. R., & Sambur, M. R. (1975). "An algorithm for determining the endpoints of isolated utterances." *Bell System Technical Journal*, 54(2), 297–315.
- Itakura, F. (1975). "Minimum prediction residual principle applied to speech recognition." *IEEE Trans. Acoustics, Speech, Signal Processing*, 23(1), 67–72.
- Linde, Y., Buzo, A., & Gray, R. M. (1980). "An algorithm for vector quantizer design." *IEEE Trans. Communications*, 28(1), 84–95.

---

**Proyecto académico** — Reconocimiento de palabras habladas con LPC + VQ  
**Última actualización**: 2026-04-23
