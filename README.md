# KKBox Churn Prediction

![CI](https://github.com/gloriadrm/kkbox-churn-prediction/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.13-blue)
![MLflow](https://img.shields.io/badge/MLflow-DagsHub-0194E2)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688)
![Docker](https://img.shields.io/badge/Docker-940%20MB-2496ED)

Predicción de abandono de clientes para un servicio de streaming musical por suscripción, desde los datos en bruto hasta un modelo versionado en MLflow y listo para servirse.

**Experimentos y modelos:** [MLflow en DagsHub](https://dagshub.com/gloriadrm/kkbox-churn-prediction.mlflow)

## El problema

KKBox, un servicio de streaming musical asiático, quiere anticipar qué suscriptores no renovarán su suscripción tras su vencimiento. Detectarlos a tiempo permite lanzar acciones de retención (descuentos, recordatorios, ofertas) antes de perderlos.

El reto principal es el **desbalanceo**: solo el **9 %** de los clientes hace churn, así que un modelo que dijera "nadie se va" acertaría el 91 % de las veces sin ser útil.

## Resultados

Evaluados sobre un conjunto de validación que el modelo no ve ni para entrenar ni para elegir el umbral (291.288 clientes):

| Métrica | Valor | Lectura |
|---|---|---|
| **Average precision (PR-AUC)** | **0,73** | 8 veces mejor que el azar (0,09) |
| ROC-AUC | 0,92 | Muy buena capacidad de ordenar clientes por riesgo |
| Recall de churn | 81 % | Detecta 8 de cada 10 clientes que se van |
| Precisión de churn | 43 % | De cada 10 clientes marcados, 4 se van de verdad |
| Clientes marcados | 17 % | Volumen que tendría que gestionar el equipo de retención |

**En términos de negocio:** contactando al 17 % de los clientes se captura el 81 % del churn.

## Enfoque

**Datos.** Tres tablas de Kaggle: etiquetas de churn, perfil de los miembros y un histórico de transacciones de casi 1 GB, agregado a una fila por cliente (970.960 clientes).

**Prevención de *data leakage*.** Todas las variables se calculan solo con información anterior a una **fecha de corte** (1 de marzo de 2017), para que el modelo no use datos del futuro que no tendría en producción.

**Feature engineering.** 38 variables por cliente: recencia, antigüedad, días hasta la expiración, frecuencia y regularidad de pagos, tasas de cancelación y renovación automática, comportamiento del último día de transacciones, etc.

**Calidad de datos.** Edades fuera de rango convertidas en nulos con un indicador `age_unknown`, imputación con medianas aprendidas solo en train y validación automática de tipos y columnas.

**Selección de modelo.** Comparación de regresión logística (L1, L2, Elastic Net), Random Forest, XGBoost y HistGradientBoosting con búsqueda de hiperparámetros y validación cruzada. Ganó **HistGradientBoosting**, sin overfitting (AUC de train y validación prácticamente iguales).

**Umbral de decisión.** En lugar del 0,5 por defecto (inadecuado con un 9 % de churn), el umbral se ajusta en cada entrenamiento con validación cruzada siguiendo el **criterio de Youden** y se guarda junto al modelo. Prioriza no dejar escapar clientes, asumiendo que una acción de retención es barata frente al coste de perder un suscriptor.

**Champion / challenger.** Cada reentrenamiento registra una nueva versión del modelo en MLflow, pero solo sustituye al modelo en producción (`champion`) si mejora su **average precision** en validación. Se elige esta métrica porque, con clases desbalanceadas, se centra en la clase que importa y no depende del umbral.

## Arquitectura

```mermaid
flowchart LR
    A[Datos en bruto<br/>Kaggle] -->|kkbox-churn prepare| B[Features por cliente<br/>train / validación]
    B -->|kkbox-churn train| C[Entrenamiento<br/>pipeline + umbral<br/>evaluación en validación]
    C -->|nueva versión<br/>champion solo si mejora el AP| D[(MLflow Model Registry<br/>DagsHub)]
    D -->|alias champion| E[kkbox-churn predict<br/>predicciones por lotes en CSV]

    subgraph Docker
        F[API FastAPI<br/>/predict · /predict/batch · /health]
    end

    D -->|alias champion<br/>al arrancar| F
```

## Estructura del proyecto

```
├── src/kkbox_churn/
│   ├── config.py            # rutas, columnas, hiperparámetros y MLflow
│   ├── cli.py               # comandos prepare / train / predict
│   ├── logging_config.py    # logs en consola y archivo
│   ├── data/                # carga de datos y construcción del dataset
│   ├── features/            # feature engineering desde las tablas en bruto
│   ├── preprocessing/       # calidad de datos y transformaciones (sklearn)
│   ├── models/              # entrenamiento, predicción, evaluación y tracking
│   └── api/                 # API FastAPI y esquemas Pydantic
├── tests/                   # tests con datos sintéticos y modelo simulado
├── notebooks/
│   ├── dev/                 # exploración, EDA y experimentación
│   └── pro/                 # flujo de producción paso a paso
├── .github/workflows/ci.yml # tests y build de Docker en cada push
├── Dockerfile
└── pyproject.toml
```

## Cómo ejecutarlo

**1. Instalar** (requiere [uv](https://docs.astral.sh/uv/))

```bash
uv sync --extra dev
```

**2. Datos.** Descarga `train_v2.csv`, `members_v3.csv`, `transactions.csv` y `transactions_v2.csv` del [reto de Kaggle](https://www.kaggle.com/c/kkbox-churn-prediction-challenge/data) en `data/raw/`.

**3. MLflow (opcional).** Sin configuración, los experimentos se guardan en local (`mlflow.db`). Para usar un servidor remoto, crea un archivo `.env`:

```bash
MLFLOW_TRACKING_URI=https://dagshub.com/<usuario>/<repo>.mlflow
MLFLOW_TRACKING_USERNAME=<usuario>
MLFLOW_TRACKING_PASSWORD=<token>
```

**4. Pipeline**

```bash
uv run kkbox-churn prepare    # construye las features y separa train / validación
uv run kkbox-churn train      # entrena, evalúa y registra el modelo en MLflow
uv run kkbox-churn predict    # predicción por lotes con el modelo champion
```

`train --force-promote` promueve el nuevo modelo aunque no mejore al actual (por ejemplo, si cambian las features).

## Tests

```bash
uv run pytest
```

17 tests que no necesitan los datos reales ni conexión a MLflow: se ejecutan sobre clientes sintéticos generados a partir de la configuración del proyecto, y la API se prueba con un modelo entrenado sobre esos datos. Cubren la limpieza e imputación (incluida la ausencia de *data leakage* en las medianas), el entrenamiento y la aplicación del umbral, la validación de entradas de la API y la gestión de errores del CLI. Se ejecutan automáticamente en cada push con GitHub Actions.

## API

```bash
uv run uvicorn kkbox_churn.api.main:app
```

Documentación interactiva en `http://127.0.0.1:8000/docs`.

| Endpoint | Descripción |
|---|---|
| `GET /health` | Estado de la API, versión y umbral del modelo cargado |
| `POST /predict` | Probabilidad y clase de churn de un cliente |
| `POST /predict/batch` | Lo mismo para hasta 1.000 clientes |

Las entradas se validan con Pydantic (tipos, rangos y campos desconocidos). Los valores ausentes se imputan igual que en el entrenamiento, y se registra un aviso si a un cliente le falta más de la mitad de las features.

## Docker

**Construir la imagen**

```bash
docker build -t kkbox-churn-api .
```

**Ejecutar el contenedor**

```bash
docker run --rm -p 8000:8000 --env-file .env kkbox-churn-api
```

- `-p 8000:8000`: conecta el puerto 8000 de la máquina con el del contenedor.
- `--env-file .env`: pasa las credenciales de MLflow como variables de entorno. Nunca se incluyen en la imagen.
- `--rm`: elimina el contenedor al pararlo.

La API queda disponible en `http://127.0.0.1:8000/docs`. Al arrancar, descarga de MLflow el modelo con alias `champion`, así que para servir un modelo recién promovido basta con reiniciar el contenedor, sin reconstruir la imagen.

**Comprobar el estado**

```bash
docker ps    # la columna STATUS muestra (healthy) cuando /health responde
```
  
La imagen ocupa unos 940 MB (frente a 3,5 GB en la primera versión): usa `mlflow-skinny` en lugar de MLflow completo, monta la caché de `uv` solo durante el build y ejecuta la API con un usuario sin privilegios.

## Stack

Python · pandas · scikit-learn · MLflow · DagsHub · FastAPI · Pydantic · Docker · pytest · GitHub Actions · uv

## Roadmap

- [x] Feature engineering con prevención de *data leakage*
- [x] Comparación de modelos y ajuste de hiperparámetros
- [x] Ajuste del umbral de decisión
- [x] Paquete Python reproducible con CLI, logging y control de errores
- [x] Tracking de experimentos y Model Registry con MLflow (champion / challenger)
- [x] API de predicción con FastAPI y Pydantic
- [x] Docker
- [x] Tests con pytest e integración continua con GitHub Actions

## Limitaciones y próximos pasos

- **Validación en el mismo periodo.** Train y validación proceden de la misma fecha de corte. Una validación temporal (entrenar con un mes y evaluar con el siguiente) mediría mejor el rendimiento real en producción.
- **Umbral basado en un supuesto de costes.** El criterio de Youden asume que contactar a un cliente es barato frente a perderlo. Con costes reales de retención, el umbral debería optimizar el beneficio esperado.
- **Datos de actividad no utilizados.** El dataset incluye registros diarios de escucha (`user_logs`), que no se han incorporado por su tamaño y que probablemente aportarían señal sobre el compromiso del usuario.
- **Edad desconocida en el 60 % de los clientes.** Se trata con imputación y un indicador, pero limita el valor de esta variable.
- **Monitorización.** El siguiente paso sería detectar *data drift* en las features de entrada y lanzar el reentrenamiento automáticamente.
- **Explicabilidad.** Añadir SHAP para justificar cada predicción ante el equipo de negocio.