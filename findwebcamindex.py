import cv2

def find_available_cameras(max_index=10):
    available_cameras = []
    for index in range(max_index):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            available_cameras.append(index)
            cap.release()
    return available_cameras

def show_all_webcams():
    cameras = find_available_cameras()
    if not cameras:
        print("Tidak ada kamera yang ditemukan.")
        return

    caps = [cv2.VideoCapture(idx, cv2.CAP_DSHOW) for idx in cameras]
    window_names = [f'Webcam {idx}' for idx in cameras]

    while True:
        for cap, win_name in zip(caps, window_names):
            ret, frame = cap.read()
            if ret:
                cv2.imshow(win_name, frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Tekan ESC untuk keluar
            break

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    show_all_webcams()