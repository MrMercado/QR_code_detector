import cv2
import numpy as np

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

def find_qr_corners(image):
    edges = cv2.Canny(image, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    qr_corners = None
    max_area = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # Filtro para ignorar contornos pequeños
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:  # Cuatro esquinas
                if area > max_area:
                    max_area = area
                    qr_corners = approx.reshape(-1, 2).astype(np.float32)
    
    return qr_corners

def straighten_qr_code(image, qr_corners):
    qr_width = 200  # Tamaño deseado para el código QR recto
    qr_height = 200
    target_corners = np.array([[0, 0], [qr_width, 0], [qr_width, qr_height], [0, qr_height]], dtype=np.float32)

    transformation_matrix = cv2.getPerspectiveTransform(qr_corners, target_corners)
    straightened_qr = cv2.warpPerspective(image, transformation_matrix, (qr_width, qr_height))

    return straightened_qr

def main(image_path):
    image = preprocess_image(image_path)
    
    qr_corners = find_qr_corners(image)
    
    if qr_corners is not None:
        straightened_qr = straighten_qr_code(image, qr_corners)
        
        cv2.imshow('Código QR Rectificado', straightened_qr)
        cv2.imwrite('straightened_qr.jpg', straightened_qr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No se pudo encontrar un código QR en la imagen.")


if __name__ == "__main__":
    image_path = "./Prueba_crop_opencv/camera1.jpg"
    main(image_path)
