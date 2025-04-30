from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import os
from PIL import Image, ImageEnhance
import requests
import random
from geopy.geocoders import Nominatim


app = Flask(__name__)
UPLOAD_FOLDER = 'dataset'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
training_data_path = "trainer/trainer.yml"

global stop_recognition
stop_recognition = False

import serial
import time
import os

COM_PORT = "COM4"  # Change this if needed
BAUD_RATE = 9600

# 🔍 Step 1: Find process using COM5
def kill_com_port():
    print("🔍 Checking for processes using", COM_PORT)
    try:
        output = os.popen(f'tasklist | findstr /i {COM_PORT}').read()
        if output:
            print("⚠️ Process found using", COM_PORT)
            os.system(f'taskkill /F /IM {output.split()[0]}')  # Kill process
            print("✅ Process killed. Now opening COM port...")
        else:
            print("✅ No process found. Proceeding...")
    except Exception as e:
        print("❌ Error checking COM port:", e)

# 🔥 Step 2: Kill any locked process
kill_com_port()

# 🔌 Step 3: Open Serial Connection
try:
    arduino = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # Allow time to establish connection
    print("✅ Serial Connection Established on", COM_PORT)
except serial.SerialException as e:
    print(f"❌ Serial Connection Failed: {e}")

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/get_location', methods=['GET'])
def get_location():
    lat = request.args.get("lat")
    lng = request.args.get("lng")

    if lat and lng:
        try:
            geolocator = Nominatim(user_agent="geoapi")
            location = geolocator.reverse((lat, lng), exactly_one=True)
            address = location.address if location else "Unknown Location"
        except:
            address = "Error fetching location"
    else:
        address = "Location Unavailable"

    return jsonify({"address": address})

url ="http://100.78.68.237:8080/video"
@app.route('/collect', methods=['POST'])
def collect():
    user_id = request.form['user_id']
    user_name = request.form['user_name']
    capture_mode = request.form['capture_mode']

    # Ensure user ID is valid
    if not user_id.isdigit():
        return jsonify({'status': 'Invalid user ID'}), 400

    # Delete previous images for this user
    existing_images = [f for f in os.listdir(UPLOAD_FOLDER) if f.startswith(f"User.{user_id}.")]
    for img_file in existing_images:
        os.remove(os.path.join(UPLOAD_FOLDER, img_file))

    # Update names dynamically
    names = load_names()
    names[user_id] = user_name
    save_names(names)

    if capture_mode == 'webcam':
        cam = cv2.VideoCapture(0)
        count = 0
        while count < 100:  # Capture 100 images
            ret, img = cam.read()
            if not ret:
                break
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)  # Apply histogram equalization
            faces = face_detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                count += 1
                cv2.imwrite(f"{UPLOAD_FOLDER}/User.{user_id}.{count}.jpg", gray[y:y + h, x:x + w])
        cam.release()
        return jsonify({'status': 'Dataset collection complete'})

    elif capture_mode == 'upload':
        if 'file' not in request.files:
            return jsonify({'status': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'status': 'No selected file'}), 400

        # Ensure file is saved before augmentation
        img_path = os.path.join(UPLOAD_FOLDER, f'User.{user_id}.original.jpg')
        file.save(img_path)  # Save original image

        if not os.path.exists(img_path):  # Check if file is actually saved
            return jsonify({'status': 'File upload failed'}), 500

        augment_images(img_path, user_id)  # Call function to generate augmented images
        return jsonify({'status': 'Image uploaded and augmented'})

    return jsonify({'status': 'Invalid request'}), 400



def augment_images(img_path, user_id):
    # Ensure original image exists
    if not os.path.exists(img_path):
        print(f"Error: File {img_path} not found!")
        return

    img = Image.open(img_path).convert('L')  # Convert to grayscale
    img.save(f"{UPLOAD_FOLDER}/User.{user_id}.1.jpg")

    for i in range(2, 91):  # Generate 90 variations
        img_aug = img.copy()

        if i % 3 == 0:
            img_aug = img_aug.rotate(random.randint(-20, 20))  # Rotate
        elif i % 3 == 1:
            enhancer = ImageEnhance.Brightness(img_aug)
            img_aug = enhancer.enhance(random.uniform(0.8, 1.2))  # Adjust brightness
        elif i % 3 == 2:
            img_aug = img_aug.transpose(Image.FLIP_LEFT_RIGHT)  # Flip horizontally

        img_aug.save(f"{UPLOAD_FOLDER}/User.{user_id}.{i}.jpg")

    # Delete the original uploaded image **after** augmentation
    os.remove(img_path)



@app.route('/train', methods=['GET'])
def train():
    image_paths = [os.path.join(UPLOAD_FOLDER, f) for f in os.listdir(UPLOAD_FOLDER)]
    if not image_paths:
        return jsonify({'status': 'No dataset found. Collect images first!'})

    face_samples, ids = [], []
    for img_path in image_paths:
        img = Image.open(img_path).convert('L')
        img_numpy = np.array(img, 'uint8')
        id = int(os.path.split(img_path)[-1].split('.')[1])
        faces = face_detector.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            face_samples.append(img_numpy[y:y + h, x:x + w])
            ids.append(id)

    recognizer.train(face_samples, np.array(ids))
    recognizer.write(training_data_path)
    return jsonify({'status': 'Training completed'})


@app.route('/recognize', methods=['GET'])
def recognize():
    global stop_recognition
    stop_recognition = False

    cam = cv2.VideoCapture(url)
    recognizer.read(training_data_path)
    minW, minH = 0.1 * cam.get(3), 0.1 * cam.get(4)
    names = load_names()

    last_sent_data = None  # Stores the last sent value to avoid repeated sending

    while not stop_recognition:
        ret, img = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # Apply histogram equalization
        faces = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(int(minW), int(minH)))

        for (x, y, w, h) in faces:
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

            if confidence < 98:  # Recognized person
                name = names.get(str(id), "Unknown")
                if last_sent_data != "*1":  # Prevent repeated sending
                    arduino.write(b"*1")  # Send *1 to Arduino
                    last_sent_data = "*1"
            else:  # Unknown person
                name = "Unknown"
                if last_sent_data != "*2":  # Prevent repeated sending
                    arduino.write(b"*2")  # Send *2 to Arduino
                    last_sent_data = "*2"

            # Display result on frame
            cv2.putText(img, name, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Recognition', img)
        if cv2.waitKey(10) & 0xFF == 27:
            break

    cam.release()
    cv2.destroyAllWindows()
    return "", 204  # No response displayed on frontend


@app.route('/stop_recognition', methods=['GET'])
def stop_face_recognition():
    global stop_recognition
    stop_recognition = True
    return "", 204  # No content response


def load_names():
    try:
        with open("names.txt", "r") as f:
            return eval(f.read())  # Read stored names
    except:
        return {}


def save_names(names):
    with open("names.txt", "w") as f:
        f.write(str(names))  # Save names to file


if __name__ == '__main__':
    app.run(debug=False)  # Disable Flask's automatic restart

