import os, cv2
FILE = os.path.join("input_video", "parking_space.mp4")
def get_stream():
    path = FILE if os.path.exists(FILE) else "parking_sample.mp4"
    return cv2.VideoCapture(path)
