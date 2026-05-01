import cv2
from ultralytics import YOLO

VIDEO_PATH = 1
MODEL_PATH = "yolov8n.pt"

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

LINE_START = (250, 470)
LINE_END   = (900, 400)

in_count = 0
out_count = 0

track_history = {}
counted_ids = set()

# ===== ปุ่ม Exit =====
exit_btn = (20, 140, 160, 190)  # (x1, y1, x2, y2)
exit_clicked = False

def side_of_line(point, a, b):
    x, y = point
    x1, y1 = a
    x2, y2 = b
    return (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)

# ===== Mouse Click =====
def mouse_event(event, x, y, flags, param):
    global exit_clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        x1, y1, x2, y2 = exit_btn
        if x1 <= x <= x2 and y1 <= y <= y2:
            exit_clicked = True

cv2.namedWindow("People Counter")
cv2.setMouseCallback("People Counter", mouse_event)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        classes=[0],
        conf=0.4,
        iou=0.5,
        verbose=False
    )

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)

        for box, track_id in zip(boxes, ids):
            x1, y1, x2, y2 = map(int, box)
            cx = int((x1 + x2) / 2)
            cy = int(y2)

            current_side = side_of_line((cx, cy), LINE_START, LINE_END)

            if track_id not in track_history:
                track_history[track_id] = current_side
            else:
                previous_side = track_history[track_id]

                if previous_side * current_side < 0 and track_id not in counted_ids:
                    if previous_side > 0 and current_side < 0:
                        in_count += 1
                    else:
                        out_count += 1

                    counted_ids.add(track_id)

                track_history[track_id] = current_side

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"ID {track_id}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ===== วาดเส้น =====
    cv2.line(frame, LINE_START, LINE_END, (0, 255, 255), 4)

    # ===== แสดงจำนวน =====
    cv2.rectangle(frame, (20, 20), (280, 120), (0, 0, 0), -1)
    cv2.putText(frame, f"IN  : {in_count}", (35, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    cv2.putText(frame, f"OUT : {out_count}", (35, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # ===== วาดปุ่ม Exit =====
    x1, y1, x2, y2 = exit_btn
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), -1)
    cv2.putText(frame, "EXIT", (x1 + 20, y1 + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow("People Counter", frame)

    # ===== กดปุ่ม หรือ ESC =====
    if exit_clicked:
        break
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()