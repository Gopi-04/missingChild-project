# MISSING CHILD TRACKING AND RECOGNITION SYSTEM USING MACHINE LEARNING.

A real-time **Child Tracking and Recognition System** using Machine Learning and IP Webcam.

## 📌 Overview
This project aims to assist in the identification and tracking of missing children using live camera feeds and machine learning. Leveraging YOLOv8, OpenCV, and an Android-based IP Webcam, the system can detect and recognize children in real-time from video streams.

## 🚀 Features
- Real-time object detection using YOLOv8
- Live IP camera integration via mobile phone
- Bounding box and label rendering
- Distance estimation from camera to subject
- Custom model training via Roboflow

## 🛠 Technologies Used
- Python  
- YOLOv8 (Ultralytics)  
- OpenCV  
- NumPy  
- Roboflow  
- IP Webcam App (Android)

## 🔧 Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/missingChild-project.git
   cd missingChild-project
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the YOLOv8 Model**
   - Place the trained `.pt` file (e.g., `child_model.pt`) in the project directory.
   - You can train or export the model using [Roboflow](https://roboflow.com).

4. **Run the Script**
   ```bash
   python detect.py --source http://<YOUR_IP>:8080/video --weights child_model.pt --conf 0.5
   ```
   Replace `<YOUR_IP>` with the IP address shown in your IP Webcam app.

## 📷 Camera Setup
- Install the **IP Webcam** app from the Play Store.
- Launch the server and note the streaming URL (e.g., `http://192.168.1.100:8080/video`).
- Ensure both the phone and the PC are connected to the same Wi-Fi network.

## 🧠 Model Training
- Images labeled and dataset prepared using Roboflow.
- Trained a YOLOv8 object detection model optimized for recognizing children from various angles and lighting conditions.

## 🔮 Future Enhancements
- Integrate face recognition for improved accuracy.
- Add a web-based dashboard for live monitoring and alerting.
- Implement logging and auto-notification to guardians or authorities.

## 🤝 Contribution
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
```
