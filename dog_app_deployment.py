import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, Label, Button, Frame
from PIL import Image, ImageTk
import cv2
import os
import urllib.request
from datetime import datetime

# ==========================================
# 🛑 CONFIGURATION
# ==========================================
# We use 'r' to fix the path error
MODEL_FOLDER = r"C:\Users\abhin_yr5b44s\OneDrive\Desktop\dogclassifier" 
HISTORY_FOLDER = r"C:\Users\abhin_yr5b44s\OneDrive\Desktop\dogclassifier\images"

# Create folders if missing
if not os.path.exists(HISTORY_FOLDER):
    os.makedirs(HISTORY_FOLDER)

# --- 1. DOWNLOAD OPENCV TOOLS ---
print("🚀 Initializing...")
files_needed = {
    "MobileNetSSD_deploy.prototxt.txt": "https://raw.githubusercontent.com/djmv/MobilNet_SSD_opencv/master/MobileNetSSD_deploy.prototxt",
    "MobileNetSSD_deploy.caffemodel": "https://raw.githubusercontent.com/djmv/MobilNet_SSD_opencv/master/MobileNetSSD_deploy.caffemodel"
}
for fname, url in files_needed.items():
    if not os.path.exists(fname): 
        try: urllib.request.urlretrieve(url, fname)
        except: pass

try:
    net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")
    print("✅ Smart Crop Ready")
except:
    net = None

# --- 2. LOAD MODELS ---
print("⏳ Loading AI Brains...")
try:
    models = {
        'resnet50': keras.models.load_model(os.path.join(MODEL_FOLDER, "resnet50_dogs.keras")),
        'efficientnetb0': keras.models.load_model(os.path.join(MODEL_FOLDER, "efficientnetb0_dogs.keras")),
        'mobilenetv2': keras.models.load_model(os.path.join(MODEL_FOLDER, "mobilenetv2_dogs.keras"))
    }
    mlp = keras.models.load_model(os.path.join(MODEL_FOLDER, "mlp_ensemble.keras"))
    print("✅ Models Loaded!")
except Exception as e:
    print(f"❌ CRITICAL ERROR: Could not find models in {MODEL_FOLDER}")
    print(f"Details: {e}")
    exit()

# --- 3. BREED DATA ---
breed_info = {
    'american_bulldog': {'life': '10-12 yrs', 'traits': 'Confident, Social, Active'},
    'american_pit_bull_terrier': {'life': '8-15 yrs', 'traits': 'Loyal, Courageous, Friendly'},
    'basset_hound': {'life': '12-13 yrs', 'traits': 'Patient, Low-energy, Charming'},
    'beagle': {'life': '10-15 yrs', 'traits': 'Curious, Merry, Friendly'},
    'boxer': {'life': '10-12 yrs', 'traits': 'Bright, Fun-loving, Active'},
    'chihuahua': {'life': '14-16 yrs', 'traits': 'Charming, Graceful, Sassy'},
    'english_cocker_spaniel': {'life': '12-14 yrs', 'traits': 'Merry, Responsive, Gentle'},
    'english_setter': {'life': '12 yrs', 'traits': 'Friendly, Mellow, Merry'},
    'german_shorthaired': {'life': '10-12 yrs', 'traits': 'Friendly, Smart, Willing'},
    'great_pyrenees': {'life': '10-12 yrs', 'traits': 'Smart, Patient, Calm'},
    'havanese': {'life': '14-16 yrs', 'traits': 'Funny, Intelligent, Outgoing'},
    'japanese_chin': {'life': '10-12 yrs', 'traits': 'Charming, Noble, Loving'},
    'keeshond': {'life': '12-15 yrs', 'traits': 'Friendly, Lively, Outgoing'},
    'leonberger': {'life': '7 yrs', 'traits': 'Gentle, Friendly, Playful'},
    'miniature_pinscher': {'life': '12-16 yrs', 'traits': 'Fearless, Spirited, Proud'},
    'newfoundland': {'life': '9-10 yrs', 'traits': 'Sweet, Patient, Devoted'},
    'pomeranian': {'life': '12-16 yrs', 'traits': 'Lively, Bold, Inquisitive'},
    'pug': {'life': '13-15 yrs', 'traits': 'Loving, Charming, Mischievous'},
    'saint_bernard': {'life': '8-10 yrs', 'traits': 'Playful, Charming, Inquisitive'},
    'samoyed': {'life': '12-14 yrs', 'traits': 'Adaptable, Friendly, Gentle'},
    'scottish_terrier': {'life': '12 yrs', 'traits': 'Independent, Confident, Spirited'},
    'shiba_inu': {'life': '13-16 yrs', 'traits': 'Alert, Active, Attentive'},
    'staffordshire_bull_terrier': {'life': '12-14 yrs', 'traits': 'Clever, Brave, Tenacious'},
    'wheaten_terrier': {'life': '12-14 yrs', 'traits': 'Happy, Steady, Confident'},
    'yorkshire_terrier': {'life': '11-15 yrs', 'traits': 'Sprightly, Tomboyish, Loving'}
}

# Fetch Names
cat_breeds = {"Abyssinian","Bengal","Birman","Bombay","British_Shorthair","Egyptian_Mau",
              "Maine_Coon","Persian","Ragdoll","Russian_Blue","Siamese","Sphynx"}
info = tfds.builder("oxford_iiit_pet").info
dog_names = [n for n in info.features["label"].names if n not in cat_breeds]

# --- 4. LOGIC ---
def get_dog_roi(img):
    (h, w) = img.shape[:2]
    if net:
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()
        best_conf = 0.0
        dog_box = None
        for i in range(detections.shape[2]):
            conf = detections[0, 0, i, 2]
            if int(detections[0, 0, i, 1]) == 12 and conf > 0.2:
                if conf > best_conf:
                    best_conf = conf
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    dog_box = box.astype("int")
        if dog_box is not None:
            (sx, sy, ex, ey) = dog_box
            sx, sy = max(0, sx-20), max(0, sy-20)
            ex, ey = min(w, ex+20), min(h, ey+20)
            return img[sy:ey, sx:ex], True
    return img, False

def predict_frame(cv2_img):
    roi, cropped = get_dog_roi(cv2_img)
    
    # Save History
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cv2.imwrite(f"{HISTORY_FOLDER}/dog_{ts}.jpg", cv2_img)
    
    # Predict
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img_r = cv2.resize(roi_rgb, (224, 224))
    img_arr = np.expand_dims(img_r, axis=0) / 255.0
    
    p1 = models['resnet50'].predict(img_arr, verbose=0)
    p2 = models['efficientnetb0'].predict(img_arr, verbose=0)
    p3 = models['mobilenetv2'].predict(img_arr, verbose=0)
    final = mlp.predict(np.concatenate([p1, p2, p3], axis=1), verbose=0)[0]
    
    # Top 3
    top_3 = np.argsort(final)[-3:][::-1]
    top_name = dog_names[top_3[0]]
    top_conf = final[top_3[0]] * 100
    info = breed_info.get(top_name, {'life': '?', 'traits': '?'})
    
    plt.figure(figsize=(5, 6))
    plt.imshow(roi_rgb)
    if top_conf < 50:
        plt.title(f"⚠️ Uncertain ({top_conf:.1f}%)", color="red")
    else:
        plt.title(f"🏆 {top_name}\n{info['life']}", color="green", fontweight="bold")
    
    # Add Top 3 Text
    txt = "Top 3:\n"
    for i in top_3:
        txt += f"{dog_names[i]}: {final[i]*100:.1f}%\n"
    plt.xlabel(txt)
    plt.show()

# --- 5. GUI ---
class DogApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dog Classifier (Local)")
        self.root.geometry("400x300")
        
        Label(root, text="Select Mode", font=("Arial", 16)).pack(pady=20)
        Button(root, text="📂 Upload File", command=self.upload, width=20).pack(pady=10)
        Button(root, text="📷 Use Camera", command=self.cam, width=20).pack(pady=10)

    def upload(self):
        f = filedialog.askopenfilename()
        if f:
            img = cv2.imread(f)
            predict_frame(img)

    def cam(self):
        self.win = tk.Toplevel(self.root)
        self.win.title("Press SPACE to Snap")
        self.lbl = Label(self.win)
        self.lbl.pack()
        self.cap = cv2.VideoCapture(0)
        self.win.bind('<space>', lambda e: self.snap())
        self.update()

    def update(self):
        ret, frame = self.cap.read()
        if ret:
            self.last_frame = frame
            img = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.lbl.config(image=img)
            self.lbl.image = img
            self.win.after(10, self.update)

    def snap(self):
        predict_frame(self.last_frame)
        self.cap.release()
        self.win.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    DogApp(root)
    root.mainloop()