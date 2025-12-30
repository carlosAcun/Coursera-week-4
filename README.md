#  NLP Disaster Tweets Classification

Este proyecto aborda el problema de **clasificación de tweets** para identificar si un mensaje corresponde a un **desastre real** o no, utilizando técnicas de **Procesamiento de Lenguaje Natural (NLP)** y **Machine Learning**.  
El trabajo está basado en la competencia **“Natural Language Processing with Disaster Tweets”** de Kaggle.

---

## Dataset

El dataset proviene de Kaggle:

- **train.csv**: Tweets etiquetados (`target = 1` desastre, `target = 0` no desastre)
- **test.csv**: Tweets sin etiqueta para predicción
- **sample_submission.csv**: Ejemplo de archivo de envío

Ruta utilizada en Kaggle:
```
/kaggle/input/nlp-getting-started/
```

---

## Librerías utilizadas

- numpy  
- pandas  
- matplotlib  
- seaborn  
- scikit-learn  
- re, string, os  

No se utiliza NLTK para evitar descargas externas; se define manualmente una lista de *stopwords*.

---

## Preprocesamiento de texto

Cada tweet pasa por las siguientes etapas:

- Conversión a minúsculas  
- Eliminación de URLs  
- Eliminación de menciones (@user) y hashtags  
- Eliminación de signos de puntuación  
- Normalización de espacios  
- Eliminación de *stopwords*  

Función principal:
```python
clean_text(text)
```

---

## Análisis Exploratorio de Datos (EDA)

Se generan visualizaciones para analizar el dataset:

- Distribución de clases (disaster / non-disaster)
- Distribución del número de palabras por tweet
- Palabras clave más frecuentes según la clase

Archivos generados:
- word_count_distribution.png  
- top_keywords.png  

---

##  Modelos implementados

### Logistic Regression + TF-IDF

Pipeline principal:

- TF-IDF Vectorizer  
  - max_features = 5000  
  - ngram_range = (1, 2)  
- Logistic Regression  
  - max_iter = 1000  

Evaluación:
- Validación cruzada (5 folds)
- Métrica: F1-score

Se analiza la importancia de características (palabras más relevantes).

Archivos generados:
- feature_importance.png  
- predictive_words.png  

Archivo de envío:
```
submission.csv
```

---

### Random Forest + TF-IDF

Modelo alternativo para comparación:

- TF-IDF Vectorizer  
  - max_features = 3000  
- Random Forest  
  - n_estimators = 100  

Evaluación:
- Validación cruzada con F1-score

Archivo de envío:
```
rf_submission.csv
```

---

##  Ejecución

1. Abrir el notebook en Kaggle  
2. Verificar la ruta del dataset  
3. Ejecutar todas las celdas  
4. Subir el archivo de envío a Kaggle  

---

##  Conclusiones

- TF-IDF + Logistic Regression ofrece un desempeño sólido como línea base.
- El análisis de coeficientes permite interpretar el modelo.
- El pipeline es reproducible y eficiente para clasificación de texto.

---

##  Autor

Proyecto desarrollado como ejercicio de **Procesamiento de Lenguaje Natural y Machine Learning**.
