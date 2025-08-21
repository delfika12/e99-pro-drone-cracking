import argparse
import os
import platform
import sys
from pathlib import Path
import torch
import cv2
import time
from models.common import DetectMultiBackend
from utils.general import (
    LOGGER,
    check_img_size,
    non_max_suppression,
    scale_boxes,
)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, smart_inference_mode

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

@smart_inference_mode()
def run(
    weights=ROOT / "best.pt",  # model path
    source1="0",  # first webcam source
    source2="1",  # second webcam source
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.35,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=5,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=True,  # use FP16 half-precision inference
    frameskip=1,  # number of frames to skip (set to 3)
):
    source1 = str(source1)
    source2 = str(source2)
    save_img = False

    # Define messages for each grid position
    grid_messages1 = {
        1: "Roll--",
        2: "Go Forward",
        3: "Roll++",
        4: "Roll--",
        5: "Go Forward",
        6: "Roll++",
        7: "Roll--",
        8: "Go Forward",
        9: "Roll++"
    }

    grid_messages2 = {
        1: "Roll--",
        2: "Go Forward",
        3: "Roll++",
        4: "Roll--",
        5: "Drop the payload",
        6: "Roll++",
        7: "Roll--",
        8: "Go Backward",
        9: "Roll++"
    }

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=False, data=None, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Initialize webcams
    cap1 = cv2.VideoCapture(int(source1))
    cap2 = cv2.VideoCapture(int(source2))
    assert cap1.isOpened(), f"Failed to open {source1}"
    assert cap2.isOpened(), f"Failed to open {source2}"

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else 1, 3, *imgsz))  # warmup
    seen = 0
    frame_count = 0  # Counter for frame skipping
    start_time = time.time()

    def get_grid_position(x, y, width, height):
        """Determine the grid position (1-9) based on the coordinates."""
        step_x = width // 3
        step_y = height // 3
        col = x // step_x
        row = y // step_y
        return int(row * 3 + col + 1)  # Convert to 1-based grid index and ensure it's an integer

    def draw_grid(frame):
        """Draw grid lines on the frame."""
        height, width, _ = frame.shape
        step_x = width // 3
        step_y = height // 3

        # Draw vertical lines
        for i in range(1, 3):
            cv2.line(frame, (i * step_x, 0), (i * step_x, height), (255, 255, 255), 1)
        
        # Draw horizontal lines
        for i in range(1, 3):
            cv2.line(frame, (0, i * step_y), (width, i * step_y), (255, 255, 255), 1)
        
        return frame

    def add_message(frame, message, object_label, fps):
        """Add message text, object label, and FPS to the frame."""
        height, width, _ = frame.shape
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (0, 255, 0)
        thickness = 1
        
        # Add object label text
        cv2.putText(frame, object_label, (10, height - 30), font, font_scale, color, thickness)
        
        # Add message text
        cv2.putText(frame, message, (10, height - 10), font, font_scale, color, thickness)
        
        # Add FPS text
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(frame, fps_text, (width - 100, height - 10), font, font_scale, color, thickness)

    while True:
        ret1, im0_1 = cap1.read()
        ret2, im0_2 = cap2.read()
        if not ret1 or not ret2:
            LOGGER.warning(f"Failed to grab frame from {source1} or {source2}")
            break

        frame_count += 1

        # Record time at the start of processing
        frame_start_time = time.time()

        if frame_count % (frameskip + 1) == 0:  # Process only every (frameskip + 1) frame
            # Process first webcam
            im1 = cv2.cvtColor(im0_1, cv2.COLOR_BGR2RGB)
            im1 = torch.from_numpy(im1).to(device)
            im1 = im1.half() if half else im1.float()  # uint8 to fp16/32
            im1 /= 255.0  # Normalize to 0.0 - 1.0
            im1 = im1.permute(2, 0, 1).unsqueeze(0)  # Change shape to (1, 3, height, width)
            pred1 = model(im1)
            pred1 = non_max_suppression(pred1, conf_thres, iou_thres, max_det=max_det)

            # Process second webcam
            im2 = cv2.cvtColor(im0_2, cv2.COLOR_BGR2RGB)
            im2 = torch.from_numpy(im2).to(device)
            im2 = im2.half() if half else im2.float()  # uint8 to fp16/32
            im2 /= 255.0  # Normalize to 0.0 - 1.0
            im2 = im2.permute(2, 0, 1).unsqueeze(0)  # Change shape to (1, 3, height, width)
            pred2 = model(im2)
            pred2 = non_max_suppression(pred2, conf_thres, iou_thres, max_det=max_det)

            # Process predictions for first webcam
            annotator1 = Annotator(im0_1, line_width=line_thickness, example=str(names))
            message1 = ""
            object_label1 = "No object detected"
            if len(pred1[0]):
                det1 = pred1[0]
                det1[:, :4] = scale_boxes(im1.shape[2:], det1[:, :4], im0_1.shape).round()
                for *xyxy, conf, cls in reversed(det1):
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                    annotator1.box_label(xyxy, label, color=colors(c, True))
                    # Calculate center point and determine grid position
                    x_center = (xyxy[0] + xyxy[2]) / 2
                    y_center = (xyxy[1] + xyxy[3]) / 2
                    grid_pos = get_grid_position(x_center, y_center, im0_1.shape[1], im0_1.shape[0])
                    message1 = grid_messages1.get(grid_pos, "Detected")
                    object_label1 = names[c] + " detected"  # Update object label with the detected class name
                    print(f"Webcam 1: Detected {names[c]} with confidence {conf:.2f} at grid position {grid_pos}")

            im0_1 = annotator1.result()
            im0_1 = draw_grid(im0_1)
            
            # Record end time and calculate FPS for the first webcam
            frame_end_time = time.time()
            fps1 = 1 / (frame_end_time - frame_start_time)
            add_message(im0_1, message1, object_label1, fps1)

            # Process predictions for second webcam
            annotator2 = Annotator(im0_2, line_width=line_thickness, example=str(names))
            message2 = ""
            object_label2 = "No object detected"
            if len(pred2[0]):
                det2 = pred2[0]
                det2[:, :4] = scale_boxes(im2.shape[2:], det2[:, :4], im0_2.shape).round()
                for *xyxy, conf, cls in reversed(det2):
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                    annotator2.box_label(xyxy, label, color=colors(c, True))
                    # Calculate center point and determine grid position
                    x_center = (xyxy[0] + xyxy[2]) / 2
                    y_center = (xyxy[1] + xyxy[3]) / 2
                    grid_pos = get_grid_position(x_center, y_center, im0_2.shape[1], im0_2.shape[0])
                    
                    # Check for basket detection in grid 5
                    if names[c] == "basket" and grid_pos == 5:
                        message2 = "Drop the payload"
                    else:
                        message2 = grid_messages2.get(grid_pos, "Detected")
                    
                    object_label2 = names[c] + " detected"  # Update object label with the detected class name
                    print(f"Webcam 2: Detected {names[c]} with confidence {conf:.2f} at grid position {grid_pos}")

            im0_2 = annotator2.result()
            im0_2 = draw_grid(im0_2)
            
            # Record end time and calculate FPS for the second webcam
            frame_end_time = time.time()
            fps2 = 1 / (frame_end_time - frame_start_time)
            add_message(im0_2, message2, object_label2, fps2)

            # Display results in separate windows
            cv2.imshow("Webcam 1 Depan", im0_1)
            cv2.imshow("Webcam 2 Bawah", im0_2)

            if cv2.waitKey(1) == ord("q"):  # press 'q' to quit
                break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "best.pt", help="model path")
    parser.add_argument("--source1", type=str, default="0", help="first webcam source (default is 0)")
    parser.add_argument("--source2", type=str, default="1", help="second webcam source (default is 1)")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640], help="inference size h,w")
    parser.add_argument("--conf-thres", type=float, default=0.35, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=2, help="maximum detections per image")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--line-thickness", default=2, type=int, help="bounding box thickness (pixels)")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    parser.add_argument("--hide-conf", default=False, action="store_true", help="hide confidences")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--frameskip", type=int, default=3, help="number of frames to skip")  # Set frameskip to 3
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    return opt

def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
