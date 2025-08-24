import argparse
import os
import platform
import sys
import time  # Tambahkan impor untuk penghitungan waktu
from pathlib import Path
import torch
import cv2
from models.common import DetectMultiBackend
from utils.general import (
    LOGGER,
    check_img_size,
    check_imshow,
    non_max_suppression,
    scale_boxes,
)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, smart_inference_mode
from DroneController import Drone  # Impor kelas Drone dari file DroneController.py

# Membuat instance drone
drone = Drone()
drone.connect()
drone.calibrate()

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# Fungsi untuk menerjemahkan label hasil deteksi model ke perintah penggerak drone
def interpret_hand_gesture(label):
    gesture_map = {
        'takeoff': lambda: drone.take_off(2),
        'land': lambda: drone.land(2),
        'forward': lambda: drone.move_forward(30, 0.2),
        'backward': lambda: drone.move_backward(30, 0.2),
        'right': lambda: drone.move_right(30, 0.2),
        'left': lambda: drone.move_left(30, 0.2),
        'cw': lambda: drone.rotate_cw(30, 0.2),
        'ccw': lambda: drone.rotate_ccw(30, 0.2),
        'up': lambda: drone.move_up(30, 0.2),
        'down': lambda: drone.move_down(30, 0.2),
        'stop': lambda: drone.stop(),
         # atau sesuai arah flip yang diinginkan
    }

    # Jika gesture terdeteksi dan ada di dalam peta
    if label in gesture_map:
        print(f"Executing {label} command...")
        gesture_map[label]()  # Panggil fungsi yang sesuai untuk perintah tersebut
    else:
        print("Perintah tidak dikenali, coba lagi!")

@smart_inference_mode()
def run(
    weights=ROOT / "hand_sign.pt",  # model path
    source="0",  # webcam source
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.75,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
):
    source = str(source)
    save_img = False
    webcam = source.isnumeric()

    # Pilih device
    device = select_device(device)
    
    # Patch: Ganti PosixPath ke WindowsPath jika dijalankan di Windows
    if platform.system() == "Windows":
        import pathlib
        pathlib.PosixPath = pathlib.WindowsPath

    # Load model
    model = DetectMultiBackend(weights, device=device, dnn=False, data=None, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # periksa ukuran gambar

    # Inisialisasi webcam
    cap = cv2.VideoCapture(int(source))
    assert cap.isOpened(), f"Failed to open {source}"

    # Lakukan pemanasan model
    model.warmup(imgsz=(1 if pt or model.triton else 1, 3, *imgsz))
    seen = 0

    while True:
        ret, im0 = cap.read()
        if not ret:
            LOGGER.warning(f"Failed to grab frame from {source}")
            break
        
        start_time = time.time()  # Mulai penghitungan waktu di sini

        # Konversi gambar dari BGR ke RGB dan ubah bentuk tensor
        im = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 ke fp16/32
        im /= 255.0  # Normalisasi ke rentang 0.0 - 1.0
        im = im.permute(2, 0, 1).unsqueeze(0)  # Ubah bentuk menjadi (1, 3, tinggi, lebar)

        # Inference
        pred = model(im)

        # Non-Maximum Suppression (NMS)
        pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=max_det)

        # Proses hasil deteksi
        for i, det in enumerate(pred):  # per image
            seen += 1
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Ubah skala bounding box dari ukuran model ke ukuran asli gambar
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Tambahkan kotak dan label
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                    annotator.box_label(xyxy, label, color=colors(c, True))

                    # Eksekusi perintah drone berdasarkan label yang terdeteksi
                    interpret_hand_gesture(names[c])  # Menjalankan perintah berdasarkan label yang terdeteksi

            # Tampilkan hasil deteksi
            im0 = annotator.result()
            elapsed_time = time.time() - start_time  # Hitung waktu yang dibutuhkan
            fps = 1 / elapsed_time if elapsed_time > 0 else 0  # Hitung FPS
            
            # Tampilkan FPS di pojok kanan bawah
            cv2.putText(im0, f'FPS: {fps:.0f}', (im0.shape[1] - 120, im0.shape[0] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            cv2.imshow("YOLOv5 Detection", im0)
            if cv2.waitKey(1) == ord("q"):  # tekan 'q' untuk keluar
                break

    cap.release()
    cv2.destroyAllWindows()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "Hand_sign.pt", help="model path")
    parser.add_argument("--source", type=str, default="0", help="webcam source (default is 0)")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.50, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--line-thickness", default=3, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand ukuran gambar jika hanya satu nilai diberikan
    return opt

def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
