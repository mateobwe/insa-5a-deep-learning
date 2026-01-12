import sys
import os
import cv2
import yt_dlp
from ultralytics import YOLO

REPO_ROOT="/home/boukari/5A-TP-DeepLearning/insa-5a-deep-learning"

our_model = YOLO(f"{REPO_ROOT}/ultralytics/runs/detect/train4/weights/best.pt")
model_100e = YOLO(f"{REPO_ROOT}/model/100e_yolo11n_img360/weights/best.pt")
model_1024 = YOLO(f"{REPO_ROOT}/model/yolo11n_img1024_ep20/weights/best.pt")
model_640 = YOLO(f"{REPO_ROOT}/model/yolo11m_img640_ep20/weights/best.pt")

model = model_640

def process_capture(capture, writer=None):
    cv2.namedWindow("Détection", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = capture.read()
        if not ret:  
            break 
        
        frame_with_prediction = model(frame)[0].plot()

        if writer is not None:
            writer.write(frame_with_prediction)

        cv2.imshow("Détection", frame_with_prediction)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()

    if writer is not None:
        writer.release()

if sys.argv[1] == "file":
    for vid_nb in [1, 2, 3, 4, 5]:
        capture = cv2.VideoCapture(f"{REPO_ROOT}/data/videos/video{vid_nb}.mp4")
        process_capture(capture)
elif sys.argv[1] == "yt":
    for url in [
        "https://www.youtube.com/watch?v=4D7vSYky2pE", # jo_url
        "https://www.youtube.com/watch?v=OTc4TUImlag", # sony_whch720n_url
    ]: 
        video_info = yt_dlp.YoutubeDL({
            "quiet":True,
            "format":"best[ext=mp4]/best"
        }).extract_info(url, download = False)
        video_url = video_info.get("url")

        capture = cv2.VideoCapture(video_url)
        process_capture(capture)
elif sys.argv[1] == "cam":
    capture = cv2.VideoCapture(0)
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    outdir = f"{REPO_ROOT}/data/webcam"
    os.makedirs(outdir, exist_ok=True)
    out = cv2.VideoWriter(
        f"{outdir}/output2.mp4", 
        fourcc, 
        20.0, 
        (frame_width, frame_height)
    )
    process_capture(capture, out)

cv2.destroyAllWindows()
