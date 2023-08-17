# Objetivos

Desarrollar una aplicación web que permita detectar códigos QR en imágenes para decodificar y tener acceso al contenido del código QR.

# Metodología

1. Definir base de datos con imágenes de códigos QR con sus labels.
2. Desarrollar un modelo de Deep Learning para detectar y extraer los códigos QR que se encuentren en una imagen.
3. Construir el pipeline completo de predicción que permita hacer inferencia de tal manera que tenga de entrada una imagen y de como salida los codigos QR extraidos.
4. Alinear los códigos QR si se encuentran rotados
5. Analizar los códigos QR mediante técnicas de decodificación para extraer el texto que contiene.
6. Tener disponible el modelo para hacer inferencia mediante una aplicación web local y en la nube.

# Desarrollo

## Definir base de datos para entrenar el modelo

Se realiza una búsqueda exhaustiva de bases de datos con códigos QR que se encuentre debidamente etiquetada para tareas de detección de objetos, para esto se buscan bases de datos en Kaggle y Roboflow, encontrando una excelente base de datos en Roboflow llamada *****xcerno30 QR codes Image Dataset*** publicada el 27 de marzo del 2023 que se encuentra en este [link](https://universe.roboflow.com/matej-cernohous/xcerno30-qr-codes/dataset/21).

La base de datos consta de 2.1K imágenes de entrenamiento, 607 imágenes de validación y 305 imágenes para test.

## Desarrollar un modelo de Deep Learning para detectar y extraer los códigos QR que se encuentren en una imagen.

La transferencia de aprendizaje nos permite entrenar un modelo personalizado en poco tiempo y con altos estándares de validación ya que este inicialmente tiene una capacidad de reconocer patrones para la tarea a desarrollar, en este caso, se trabaja con imágenes, estos modelos pre-entrenados son capaz de detectar patrones como líneas, figuras, texturas entere otros.

Para esta tarea es indicado trabajar con un modelo de detección de objetos, en este caso usaremos transferencia de aprendizaje del modelo pre-entrenado [YOLO](https://docs.ultralytics.com/models/yolov8/#usage) en su versión 8 capaz de detectar múltiples objetos en una imagen y encerrarlos en cuadros delimitadores (bounding boxes). 

Al hacer transferencia de aprendizaje en YOLOv8 se eliminan las capas de salida de la red neuronal con el fin de aprovechar las capas previas que tienen los pesos pre-entrenados, y las capas de salida se modifican para que aprendan las características mas complejas del dataset deseado y asi personalizar los objetos a detectar, en este caso se utiliza para detectar códigos QR en imágenes.

**En el notebook llamado ([1. YOLO_QR_Train.ipynb](https://github.com/MrMercado/QR_code_detector/blob/main/1.%20YOLO_QR_Train.ipynb)) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MrMercado/QR_code_detector/blob/main/1.%20YOLO_QR_Train.ipynb) se utiliza el entorno de google colab y asi aprovechar el poder de computo para realizar el entrenamiento y validación del modelo, en este notebook se explica paso a paso como se lleva a cabo el entrenamiento y se evalúan las métricas de desempeño.**
Una vez entrenado el modelo de YOLOv8 personalizado se almacena el modelo guardado con el nombre de *xcerno30_QR_codes.pt* en la carpeta */Modelos*

## Construir el pipeline completo de predicción que permita ingresar una imagen y de como salida los códigos QR extraídos.

Este proceso se realizo de manera sencilla ya que el modelo YOLOv8 que se entreno previamente es capaz de extraer el recorte del bounding box que da como resultado la detección del código QR, a continuación se muestra un ejemplo del resultado:

![image](https://github.com/MrMercado/QR_code_detector/assets/126843626/cd91a963-de71-4b22-88ff-ee57bb239d6f)

En la imagen de la izquierda se encuentra la imagen subida por el usuario, en la imagen del medio la detección realizada por el modelo de Deep Learning y en la imagen de la derecha el recorte realizado para obtener exclusivamente el contenido del código QR
