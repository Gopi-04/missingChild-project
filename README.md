```markdown
# MISSING CHILD TRACKING AND RECOGNITION SYSTEM USING MACHINE LEARNING.

A real-time **Child Tracking and Recognition System** using Machine Learning and IP Webcam.

## 📌 Overview
This project aims to assist in the identification and tracking of missing children using live camera feeds and machine learning. Leveraging YOLOv8, OpenCV, and an Android-based IP Webcam, the system can detect and recognize children in real-time from video streams.

##🚀 Features
- Real-time object detection using YOLOv8
- Live IP camera integration via mobile phone
- Bounding box and label rendering
- Distance estimation from camera to subject
- Custom model training via Roboflow

##🛠 Technologies Used
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
   - Place the trained `.pt` file in the project directory.
   - You can train or export from [Roboflow](https://roboflow.com).

4. **Run the Script**
   ```bash
   python detect.py --source http://<IP>:8080/video --weights child_model.pt --conf 0.5
   ```

   Replace `<IP>` with the actual IP shown in your IP Webcam app.

## 📷 Camera Setup
- Install the **IP Webcam** app from the Play Store.
- Start the server and note the IP address (e.g., `http://192.168.1.100:8080`).
- Ensure the PC and phone are connected to the **same Wi-Fi network**.

## 🧠 Model Training
- Data prepared and labeled using Roboflow.
- YOLOv8 model trained for object detection on child images.

## 📎 Future Enhancements
- Add facial recognition for better identification.
- Build a web-based dashboard to view results and logs.
- Integrate alert system for authorities or guardians.

## 🙌 Contribution
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).

---

Let me know if you'd like me to generate the `requirements.txt` content too!
