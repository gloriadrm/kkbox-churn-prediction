# KKBox Churn Prediction

Predicción de abandono de clientes de un servicio de streaming musical (dataset KKBox, Kaggle).

**Resultado:** ROC-AUC 0,919 en validación · recall 82 % del churn marcando al 18 % de los clientes.

## Estructura
- `src/kkbox_churn/`: paquete con la lógica (datos, features, preprocesado, modelo)
- `notebooks/dev/`: exploración y experimentación
- `notebooks/pro/`: flujo de producción (prepare → train → predict)

## Reproducir
1. Descargar los datos de [Kaggle](https://www.kaggle.com/c/kkbox-churn-prediction-challenge/data) en `02_data/raw/`
2. `pip install -e .`
3. Ejecutar los notebooks de `notebooks/pro/` en orden