# 🅿️ Real-Time Parking Slot Detection

A real-time smart parking system that uses **Computer Vision** and a **custom-trained SVC classifier** to detect and count available vs. occupied parking slots from video feeds.

<img width="1920" height="1080" alt="thumb" src="https://github.com/user-attachments/assets/a76854b9-c656-483f-90a3-cb9fd5233d31" />


---

## 🚀 Features

- 🟣 Real-time video processing using OpenCV  
- 🟣 Slot segmentation using a binary mask  
- 🟣 Occupancy detection with a trained SVC classifier  
- 🟣 Visual display of available vs. total slots  
- 🟣 Scalable design for smart cities, malls, or surveillance drones  

---

## 🧠 Tech Stack

- Python 🐍  
- OpenCV  
- NumPy  
- Scikit-learn (SVC Classifier)  
- Skimage  
- Matplotlib  

---

## 🗂️ Project Structure
```bash
├── Data/ # Contains parking lot video and mask images
├── Train classifier/ # Classifier training scripts & saved models
├── main.py # Main pipeline for real-time detection
├── utils.py # Helper functions (slot detection, classifier interface)
├── requirements.txt # Python dependencies
```

---
## ⚙️ How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ayyash1/parking-slot-detector.git
   cd parking-slot-detector
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your input video and mask:**
   - Add your video to Data/
   - Use a binary mask image (PNG) to mark parking slot regions

   
2. **Run detection:**
   ```bash
   python main.py
   ```
---

## 📈 Future Scope
 - Mobile dashboard integration
 - YOLOv8 or hybrid detection
 - Edge device deployment
 - IoT display for slot guidance
 - 
---

## 🙍‍♂️Author
Developed by Ayyash Fous

---

## 📄 License
This project is open-source and available under the MIT License
