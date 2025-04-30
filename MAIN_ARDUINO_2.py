import cv2
from ultralytics import YOLO
import serial
import time

# === Arduino Serial Setup ===
arduino = serial.Serial('COM3', 9600)  # Update COM port if needed
time.sleep(2)  # Wait for Arduino to initialize

# === Load YOLOv8 Model ===
model = YOLO(best.pt)  # Make sure path is correct     'models/best.pt'

# === Camera Setup ===
cap1 = cv2.VideoCapture(0)  # Default camera
cap2 = cv2.VideoCapture("http://100.86.54.241:8080/video")  # IP camera

# === Send Signal to Arduino ===
def send_to_arduino():
    arduino.write(b'*1')

# === Process Frame for Detection ===
def process_frame(frame):
    results = model(frame)

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            confidence = float(box.conf)
            label = result.names[class_id]

            # Only if confidence is >= 95% and it's the target class
            if confidence >= 0.95 and label in ["Gopi", "Gautham_krithick"]:
                print(f"Detected {label} with {confidence:.2%} confidence")
                send_to_arduino()

                # Draw bounding box and label
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                (x1, y1, x2, y2) = xyxy
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {confidence:.2%}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# === Main Loop ===
def main():
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if ret1:
            process_frame(frame1)
            cv2.imshow('Camera 1', frame1)

        if ret2:
            process_frame(frame2)
            cv2.imshow('Camera 2', frame2)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()
    arduino.close()

if __name__ == "__main__":
    main()
