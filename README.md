# Objetivos

Desarrollar una aplicación web que permita detectar códigos QR en imágenes para decodificar y tener acceso al contenido del código QR.

# Metodología

1. Definir base de datos con imágenes de códigos QR con sus labels.
2. Desarrollar un modelo de Deep Learning para detectar y extraer los códigos QR que se encuentren en una imagen.
3. Construir el pipeline completo de predicción que permita hacer inferencia de tal manera que tenga de entrada una imagen y de como salida los códigos QR extraídos.
4. Alinear los códigos QR si se encuentran rotados
5. Analizar los códigos QR mediante técnicas de decodificación para extraer el texto que contiene.
6. Tener disponible el modelo para hacer inferencia mediante una aplicación web local y en la nube.

# Desarrollo

## 1. Definir base de datos para entrenar el modelo

Se realiza una búsqueda exhaustiva de bases de datos con códigos QR que se encuentre debidamente etiquetada para tareas de detección de objetos, para esto se buscan bases de datos en Kaggle y Roboflow, encontrando una excelente base de datos en Roboflow llamada **xcerno30 QR codes Image Dataset** publicada el 27 de marzo del 2023 que se encuentra en este [link](https://universe.roboflow.com/matej-cernohous/xcerno30-qr-codes/dataset/21).

La base de datos consta de 2.1K imágenes de entrenamiento, 607 imágenes de validación y 305 imágenes para test.

## 2. Desarrollar un modelo de Deep Learning para detectar y extraer los códigos QR que se encuentren en una imagen.

La transferencia de aprendizaje nos permite entrenar un modelo personalizado en poco tiempo y con altos estándares de validación, ya que este inicialmente tiene una capacidad de reconocer patrones para la tarea a desarrollar; en este caso, se trabaja con imágenes, estos modelos pre-entrenados son capaz de detectar patrones como líneas, figuras, texturas entere otros.

Para esta tarea es indicado trabajar con un modelo de detección de objetos, en este caso usaremos transferencia de aprendizaje del modelo pre-entrenado [YOLO](https://docs.ultralytics.com/models/yolov8/#usage) en su versión 8 capaz de detectar múltiples objetos en una imagen y encerrarlos en cuadros delimitadores (bounding boxes). 

Al hacer transferencia de aprendizaje en YOLOv8 no se tiene en cuenta los pesos pre-entrenados de las capas de salida de la red neuronal, pero se mantiene el resto de la red con el fin de aprovechar las capas previas que tienen los pesos pre-entrenados, y las capas de salida se modifican para que aprendan las características más complejas del dataset deseado y así personalizar los objetos a detectar, en este caso se utiliza para detectar códigos QR en imágenes.

En el notebook llamado ([1. YOLO_QR_Train.ipynb](https://github.com/MrMercado/QR_code_detector/blob/main/1.%20YOLO_QR_Train.ipynb)) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MrMercado/QR_code_detector/blob/main/1.%20YOLO_QR_Train.ipynb) se utiliza el entorno de google colab y así aprovechar el poder de cómputo para realizar el entrenamiento y validación del modelo, en este notebook se explica paso a paso como se lleva a cabo el entrenamiento y se evalúan las métricas de desempeño.
Una vez entrenado el modelo de YOLOv8 personalizado se almacena con el nombre de `xcerno30_QR_codes.pt` en la carpeta `/Modelos`

## 3. Construir el pipeline completo de predicción que permita ingresar una imagen y de como salida los códigos QR extraídos.

Este proceso se realizó de manera sencilla, ya que el modelo YOLOv8 que se entrenó previamente es capaz de extraer el recorte del bounding box que da como resultado la detección del código QR, a continuación se muestra un ejemplo del resultado:

![image](https://github.com/MrMercado/QR_code_detector/assets/126843626/cd91a963-de71-4b22-88ff-ee57bb239d6f)

En la imagen de la izquierda se encuentra la imagen subida por el usuario, en la imagen del medio la detección realizada por el modelo de Deep Learning y en la imagen de la derecha el recorte realizado para obtener exclusivamente el contenido del código QR.

Este proceso es gracias al parámetro `save_crop=True` que tiene el módulo predict, los recortes se guardan en una carpeta que crea el modelo por defecto y almacena las imágenes en la ruta `./runs/detect/predict/crops/qr_code/`, las imágenes se almacenan en una lista y se elimina esta ruta para no utilizar información redundante. Todo este proceso se lleva a cabo con la función `detect_qr(model, image_path):` del notebook ([2. Deteccion QR.ipynb](https://github.com/MrMercado/QR_code_detector/blob/main/3.%20Deteccion%20QR.ipynb)) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MrMercado/QR_code_detector/blob/main/3.%20Deteccion%20QR.ipynb)

## 4. Alinear los códigos QR si se encuentran rotados

Para alinear los códigos QR que se encuentran rotados se diseña una función de procesamiento de imágenes que se define a continuación:

La función `align_qr_image(img)` que se encuentra en el notebook ([2. Deteccion QR.ipynb](https://github.com/MrMercado/QR_code_detector/blob/main/3.%20Deteccion%20QR.ipynb)) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MrMercado/QR_code_detector/blob/main/3.%20Deteccion%20QR.ipynb) toma una imagen que contiene un código QR que podría estar rotado y se encarga de alinearla correctamente. Primero, determina la dimensión más grande entre el ancho y el alto de la imagen y crea una nueva imagen cuadrada de fondo negro con esa dimensión máxima. Luego, centra la imagen original dentro de esta imagen cuadrada. Esta imagen se convierte a escala de grises para facilitar el procesamiento. Después, la función itera sobre posibles ángulos de rotación, desde -90 a 90 grados, para determinar cuál es el ángulo que mejor alinea el código QR. Utiliza la detección de bordes verticales y horizontales para calcular un puntaje de alineación, y el ángulo que minimiza este puntaje se considera el mejor ángulo de alineación. Finalmente, rota la imagen cuadrada original usando el mejor ángulo encontrado y devuelve esta imagen alineada.

A continuación se muestra un ejemplo de la rotación realizada:

![image](https://github.com/MrMercado/QR_code_detector/assets/126843626/d7cf42b6-f8d6-43ba-ad92-7a8726f660cd)



## 5. Analizar los códigos QR mediante técnicas de decodificación para extraer el texto que contiene.

En este paso se implementa una función para decodificar el código QR que se explica a continuación:

La función **`decode_qr(img)`** que se encuentra en el notebook ([2. Deteccion QR.ipynb](https://github.com/MrMercado/QR_code_detector/blob/main/3.%20Deteccion%20QR.ipynb)) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MrMercado/QR_code_detector/blob/main/3.%20Deteccion%20QR.ipynb) tiene como objetivo decodificar el contenido de un código QR presente en una imagen. La función acepta una imagen como entrada y utiliza dos métodos distintos para intentar decodificar el código QR: primero, utiliza la biblioteca **`QReader`** y, si este método falla, utiliza el **`QRCodeDetector`** de OpenCV. Si alguno de estos métodos tiene éxito en la decodificación, la función devuelve el texto decodificado. En caso de que ambos métodos fallen, la función devuelve **`None`**, indicando que no se pudo decodificar el contenido del código QR en la imagen proporcionada.

Para continuar con el ejemplo, el código QR mostrado anteriormente contiene el texto `http://commons.wikimedia.org/`

## 6. Tener disponible el modelo para hacer inferencia mediante una aplicación web local y en la nube.

Una vez implementado todo el desarrollo mencionado anteriormente, se desarrolla una aplicación web con la librería de `streamlit`, la cual permite implementaciones de concepto de manera rápida con una interfaz intuitiva, a continuación se encuentran las instrucciones para hacer inferencia en la aplicación web de streamlit de manera local.

1. Crea un entorno virtual en tu espacio local
2. Clonar el repositorio e instalar las dependencias del documento `requirements.txt` 
3. Ejecutar la aplicación Web en local con el comando `streamlit run web_app_local.py`

Con esto se abrirá en el navegador la siguiente interfaz:

![image](https://github.com/MrMercado/QR_code_detector/assets/126843626/aff585ce-6846-4d3a-81a1-7e46fd561b5c)

La aplicación web local tiene tres maneras de hacer inferencia que se seleccionan en el sidebar de la izquierda

- La primera opción es seleccionando un documento desde tu computador, el resultado del procesamiento se muestra en varias columnas definidas a continuación:
    - La primera columna corresponde a la imagen seleccionada por el usuario
    - La segunda columna muestra la detección realizada por el modelo de Deep Learning
    - La tercer columna muestra los códigos QR extraídos de la imagen
    - La cuarta columna muestra el código QR rotado
    - Por último, el contenido del código QR se encuentra en el caption de cada código QR detectado.

![image](https://github.com/MrMercado/QR_code_detector/assets/126843626/8dd25e16-bdba-4784-9bcc-c3235d20dfc6)

- La segunda opción da la posibilidad de seleccionar entre 6 imágenes previamente definidas como se muestra a continuación:
![image](https://github.com/MrMercado/QR_code_detector/assets/126843626/8c57e7c7-60d1-47dc-b7fa-f277e71914b1)
- La tercer opción permite tomar una fotografía con la cámara web y hacer inferencia ¡EN VIVO!
![image](https://github.com/MrMercado/QR_code_detector/assets/126843626/0e4969b1-6a9d-4dd8-92ec-92f9926fd749)
![image](https://github.com/MrMercado/QR_code_detector/assets/126843626/200a9799-50e3-49a2-b9e4-67dc916927f9)

# Finalmente, esta aplicación web se despliega en la nube de streamlit para hacer inferencia en línea sin instalar ni descargar paquetes de python, esta app se encuentra en el siguiente link:

# [QR_code_detector](https://qrcodedetector.streamlit.app/)

### **¡NOTA! La app web desplegada en la nube corresponde al script `web_app.py`, streamlit online tiene problemas con las dependencias de la librería `QReader` al momento de hacer el despligue, por esta razón solo se intenta extraer el texto del código QR con una estrategia de decodificación en lugar de dos, lo cual reduce la robustez de la extracción de información del código QR en comparación a la app `web_app_local.py` instalada en el entorno local.**
