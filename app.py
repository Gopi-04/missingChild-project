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

    # Delete previous images for this user
    existing_images = [f for f in os.listdir(UPLOAD_FOLDER) if f.startswith(f"User.{user_id}.")]
    for img_file in existing_images:
        os.remove(os.path.join(UPLOAD_FOLDER, img_file))

    # Update names dynamically
    names = load_names()
    names[user_id] = user_name
    save_names(names)

    if capture_mode == 'webcam':
        cam = cv2.VideoCapture(url)
        count = 0
        while count < 100:  # Ensure we capture at least 100 images for better accuracy
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
        file = request.files['file']
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], f'User.{user_id}.original.jpg')
        file.save(img_path)
        augment_images(img_path, user_id)
        return jsonify({'status': 'Image uploaded and augmented'})


def augment_images(img_path, user_id):
    # Delete old images
    existing_images = [f for f in os.listdir(UPLOAD_FOLDER) if f.startswith(f"User.{user_id}.")]
    for img_file in existing_images:
        os.remove(os.path.join(UPLOAD_FOLDER, img_file))

    img = Image.open(img_path).convert('L')
    img.save(f"{UPLOAD_FOLDER}/User.{user_id}.1.jpg")

    for i in range(2, 31):  # Generate 30 variations
        img_aug = img.copy()

        if i % 3 == 0:
            img_aug = img_aug.rotate(random.randint(-20, 20))  # Rotation
        elif i % 3 == 1:
            enhancer = ImageEnhance.Brightness(img_aug)
            img_aug = enhancer.enhance(random.uniform(0.8, 1.2))  # Brightness
        elif i % 3 == 2:
            img_aug = img_aug.transpose(Image.FLIP_LEFT_RIGHT)  # Flip

        img_aug.save(f"{UPLOAD_FOLDER}/User.{user_id}.{i}.jpg")


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

    while not stop_recognition:
        ret, img = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # Apply histogram equalization
        faces = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(int(minW), int(minH)))

        for (x, y, w, h) in faces:
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

            if confidence < 98:  # Ensure only 98%+ confidence is accepted
                name = names.get(str(id), "Unknown")
            else:
                name = "Unknown"  # Strictly label unknown persons

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
    app.run(debug=True)
