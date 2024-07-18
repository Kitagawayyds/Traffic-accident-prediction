import cv2
from ultralytics import YOLO, solutions

model = YOLO("yolov10n.pt")
names = model.model.names

cap = cv2.VideoCapture("test.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (
cv2.CAP_PROP_FRAME_WIDTH,
cv2.CAP_PROP_FRAME_HEIGHT,
cv2.CAP_PROP_FPS))

print(w,h,fps)

line_pts = [(0, 3/4*h), (w, 3/4*h)]

speed_obj = solutions.SpeedEstimator(reg_pts=line_pts,
                                     names=names,
                                     view_img=True,
                                     line_thickness=3
                                     )

while cap.isOpened():
  success, im0 = cap.read()
  if not success:
    break
  tracks = model.track(im0, persist=True, tracker="bytetrack.yaml")
  im0 = speed_obj.estimate_speed(im0, tracks)


cap.release()
cv2.destroyAllWindows()
