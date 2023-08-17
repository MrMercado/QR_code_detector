# Importamos Librerias
import streamlit as st
import numpy as np
import os                                       # para trabajar sobre directorios
import shutil                                   # para trabajar sobre directorios
import cv2                                      # Para leer y mostrar imagenes
from ultralytics import YOLO                    # Nuestro detector de codigos QR
from cv2 import QRCodeDetector                  # Para extraer codigo QR
#from qreader import QReader


def process_image(image):
    # Aplicar detección y decodificación de códigos QR
    results = detect_and_decode_qr(image)

    if results:
        qr_image, crop_images, aligned_images, qr_text = results

        # Crear las columnas
        col1, col2, col3, col4 = st.columns(4)

        # Agregar títulos a las columnas
        col1.write("**Imagen Procesada**")
        col2.write("**Detección de los QR con Deep Learning**")
        col3.write("**Imágenes Recortadas**")
        col4.write("**Imágenes Alineadas**")

        # Columna 1: Imagen procesada
        col1.image(image, use_column_width=True)

        # Columna 2: Imagen resultante del detector YOLO
        col2.image(qr_image, use_column_width=True)

        # Columna 3: Imágenes recortadas
        for i, img in enumerate(crop_images):
            caption = qr_text[i] if i < len(qr_text) else None
            col3.image(resize_image(img, 800, 800), caption=caption, use_column_width=True)

        # Columna 4: Imágenes alineadas
        for i, img in enumerate(aligned_images):
            caption = qr_text[i] if i < len(qr_text) else None
            col4.image(resize_image(img, 800, 800), caption=caption, use_column_width=True)

    else:
        st.warning("No se encontraron códigos QR en la imagen.")


def resize_image(image, width, height):
    """
    Redimensionar la imagen a las dimensiones especificadas
    """
    return cv2.resize(image, (width, height))

def detect_and_decode_qr(image):
    model = YOLO("./Modelos/xcerno30_QR_codes.pt")
    resultados = model.predict(image, save_crop=True)
    qr_detectado = resultados[0].plot(labels=False)

    # Intentar obtener los recortes de códigos QR
    try:
        path_crop = './runs/detect/predict/crops/qr_code/'
        imagenes = [cv2.imread(os.path.join(path_crop, imagen)) for imagen in os.listdir(path_crop) if imagen.endswith(('.jpg'))]
        shutil.rmtree('./runs', ignore_errors=True)
    except Exception as e:
        imagenes = []

    #qreader_reader, cv2_reader = QReader(), QRCodeDetector()
    cv2_reader = QRCodeDetector()

    qr_images = []
    aligned_images = []
    qr_text = []

    for imagen in imagenes:
        try:
            aligned_img, aligned_gray = align_image(imagen)
            qr_images.append(imagen)
            aligned_images.append(aligned_img)

            #qreader_out = qreader_reader.detect_and_decode(image=imagen)
            cv2_out = cv2_reader.detectAndDecode(img=imagen)[0]

            # Validar la decodificación
            if cv2_out:  # Verificamos que la cadena no esté vacía
                qr_text.append(cv2_out)
            else:
                qr_text.append("No se logró la decodificación del código QR")
            
        except cv2.error as e:
            qr_text.append(f"No se logró la decodificación del código QR")

    return qr_detectado, qr_images, aligned_images, qr_text


def align_image(imagen):
    ###################### Alineamos imagenes con QR rotados #################################

    # Calcular la dimensión cuadrada deseada
    max_dimension = max(imagen.shape[0], imagen.shape[1])
    square_imagen = np.zeros((max_dimension, max_dimension, 3), dtype=np.uint8)
    square_imagen.fill(0)  # Rellenar con fondo negro

    # Calcular los desplazamientos para centrar la imagen original en la imagen cuadrada
    x_offset = (max_dimension - imagen.shape[1]) // 2
    y_offset = (max_dimension - imagen.shape[0]) // 2

    # Copiar la imagen original en el centro de la imagen cuadrada
    square_imagen[y_offset:y_offset + imagen.shape[0], x_offset:x_offset + imagen.shape[1]] = imagen

    # Convertir la imagen cuadrada a escala de grises
    gray_imagen = cv2.cvtColor(square_imagen, cv2.COLOR_BGR2GRAY)

    # Realizar búsqueda iterativa de ángulo de rotación
    best_angle = 0.0
    best_alignment_score = float('inf')

    for angle in range(-90, 91):
        rotation_matrix = cv2.getRotationMatrix2D((gray_imagen.shape[1] / 2, gray_imagen.shape[0] / 2), angle, 1)
        rotated_imagen = cv2.warpAffine(gray_imagen, rotation_matrix, (gray_imagen.shape[1], gray_imagen.shape[0]))
        
        vertical_edges = cv2.Sobel(rotated_imagen, cv2.CV_64F, 1, 0, ksize=5)
        horizontal_edges = cv2.Sobel(rotated_imagen, cv2.CV_64F, 0, 1, ksize=5)
        alignment_score = np.sum(np.abs(vertical_edges)) + np.sum(np.abs(horizontal_edges))
        
        if alignment_score < best_alignment_score:
            best_alignment_score = alignment_score
            best_angle = angle

    # Aplicar la rotación óptima a la imagen cuadrada
    rotation_matrix = cv2.getRotationMatrix2D((square_imagen.shape[1] / 2, square_imagen.shape[0] / 2), best_angle, 1)
    aligned_imagen = cv2.warpAffine(square_imagen, rotation_matrix, (square_imagen.shape[1], square_imagen.shape[0]))

    # Convertir la imagen alineada a escala de grises
    aligned_gray = cv2.cvtColor(aligned_imagen, cv2.COLOR_BGR2GRAY)
    
    return aligned_imagen, aligned_gray

def main():
    st.title("Detector y Decodificador de Códigos QR con Deep Learning y procesamiento de imagenes")

    option = st.sidebar.selectbox(
        'Elige una opción para hacer inferencia:',
        ('Selecciona una imagen de tu ordenador', 'Usa una imagen de muestra', 'Toma una foto con tu cámara web')
    )

    if option == 'Selecciona una imagen de tu ordenador':
        st.header('¡Selecciona una imagen de tu ordenador!')
        uploaded_image = st.file_uploader("", type=["jpg", "png", "jpeg"])

        if uploaded_image is not None:
            image = np.array(bytearray(uploaded_image.read()), dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            process_image(image)
    
    elif option == 'Usa una imagen de muestra':
        st.header('¡Selecciona una imagen de muestra!')
        sample_images = os.listdir("./img_inferencia")

        # Crear matriz 3x2 para mostrar las imágenes
        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]

        # Mostrar las imágenes en miniatura con su respectivo caption
        for idx, img_name in enumerate(sample_images):
            image_path = os.path.join("./img_inferencia", img_name)
            image = cv2.imread(image_path)
            if image is not None:
                current_col = cols[idx % 3]  # Determinar la columna actual basándose en el índice
                current_col.image(resize_image(image, 100, 100), caption=img_name)  # Ajusta el tamaño según tus preferencias

        # Luego, permitir que el usuario seleccione la imagen usando el selectbox
        selected_image = st.selectbox('Elige una imagen de muestra basándote en las miniaturas de arriba:', sample_images)

        if selected_image:
            image_path = os.path.join("./img_inferencia", selected_image)
            image = cv2.imread(image_path)

            if image is None:  # Verifica si la imagen se cargó correctamente
                st.error(f"Error al cargar la imagen {selected_image}. Asegúrate de que el archivo exista y sea una imagen válida.")
            else:
                process_image(image)

    elif option == 'Toma una foto con tu cámara web':
        st.header('¡Toma una foto con tu cámara web!')
        st.write("Haz clic en el botón para capturar una imagen desde tu cámara y analizarla en busca de códigos QR.")
        st.write("Te recomendamos estar quieto 3 segundos.")

        if st.button('Capturar Imagen'):
            cap = cv2.VideoCapture(0)
            cv2.waitKey(2000)
            ret, frame = cap.read()
            cap.release()

            if ret:
                process_image(frame)
            else:
                st.error("Error al capturar la imagen desde la cámara.")


if __name__ == "__main__":
    main()
