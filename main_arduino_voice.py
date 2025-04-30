import cv2
from ultralytics import YOLO
import serial
import time

# === Arduino Serial Setup ===
arduino = serial.Serial('COM3', 9600)  # Change COM port as needed
time.sleep(2)  # Allow Arduino to initialize

# === Load YOLOv8 Model ===
model = YOLO('models/best.pt')  # Path to your YOLOv8 trained model

# === Camera Setup ===
cap = cv2.VideoCapture(0)  # Use 0 for default camera OR replace with IP camera URL

# === Send *1 to Arduino ===
def send_to_arduino():
    arduino.write(b'*1')
    print("Sent *1 to Arduino")

# === Frame Processing ===
def process_frame(frame):
    results = model(frame)

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            confidence = float(box.conf)
            label = result.names[class_id]

            # Check for class match and high confidence
            if confidence >= 0.95 and label in ["Gopi", "Gautham_krithick"]:
                print(f"Detected: {label} ({confidence:.2%})")
                send_to_arduino()

                # Draw bounding box and label (optional visualization)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {confidence:.2%}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# === Main Loop ===
def main():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        process_frame(frame)
        cv2.imshow("Detection Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    arduino.close()

if __name__ == "__main__":
    main()
