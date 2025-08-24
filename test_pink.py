import cv2
import numpy as np

# Fungsi untuk mendeteksi objek dengan warna merah ke pink dan menggambar bounding box
def detect_red_to_pink_objects(image):
    # Mengonversi gambar dari BGR ke HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Menentukan rentang warna untuk merah hingga pink dalam ruang warna HSV
    lower_red = np.array([160, 100, 100])  # Rentang bawah (merah muda)
    upper_red = np.array([180, 255, 255])  # Rentang atas (pink)

    # Membuat mask berdasarkan rentang warna
    mask = cv2.inRange(hsv_image, lower_red, upper_red)

    # Menggunakan kontur untuk menemukan objek yang terdeteksi
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Menggambar bounding box untuk setiap objek yang terdeteksi
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Hanya deteksi objek yang cukup besar
            # Mendapatkan koordinat bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Menggambar bounding box di atas gambar
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Gambar kotak hijau
    
    return image  # Mengembalikan gambar dengan bounding box

# Menginisialisasi webcam dengan index 2
cap = cv2.VideoCapture(2)  # Menggunakan webcam dengan index 2 (biasanya webcam eksternal)

# Memeriksa apakah kamera terbuka dengan benar
if not cap.isOpened():
    print("Gagal membuka webcam dengan index 2!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal menangkap frame dari webcam!")
        break

    # Deteksi objek dengan warna merah hingga pink dan gambar bounding box
    result_image = detect_red_to_pink_objects(frame)

    # Menampilkan hasil
    cv2.imshow("Detected Objects", result_image)  # Menampilkan gambar dengan bounding box

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Melepaskan kamera dan menutup jendela
cap.release()
cv2.destroyAllWindows()
