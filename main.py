import tkinter as tk
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename
import numpy as np
import cv2
import tensorflow as tf
import json

# -----------------------------
# Load Model and Classes
# -----------------------------
print("Loading TFLite model...")
interpreter = tf.lite.Interpreter(model_path="plant_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open("classes.json", "r") as f:
    class_names = json.load(f)
print("Classes loaded:", class_names)

# -----------------------------
# Disease Info Dictionary
# -----------------------------
disease_info = {
    # ---------------- Corn Diseases ----------------
    "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot": {
        "Name": "Cercospora Leaf Spot (Gray Leaf Spot)",
        "Causes": "Fungal disease caused by Cercospora zeae-maydis",
        "Prevention": "Crop rotation, resistant hybrids, and avoid high humidity.",
        "Treatment": "Use triazole or strobilurin fungicides; remove crop debris.",
        "Plant Health": "Diseased"
    },
    "Corn_(maize)___Common_rust_": {
        "Name": "Common Rust",
        "Causes": "Fungal disease caused by Puccinia sorghi",
        "Prevention": "Plant resistant hybrids, avoid dense planting.",
        "Treatment": "Apply fungicides early (triazoles or strobilurins).",
        "Plant Health": "Diseased"
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "Name": "Northern Corn Leaf Blight",
        "Causes": "Fungal disease caused by Exserohilum turcicum",
        "Prevention": "Use resistant hybrids, rotate crops, remove debris.",
        "Treatment": "Apply fungicides such as strobilurins or triazoles.",
        "Plant Health": "Diseased"
    },
    "Corn_(maize)___Healthy": {
        "Name": "Healthy Corn Leaf",
        "Causes": "None — the plant is healthy.",
        "Prevention": "Maintain good irrigation and balanced nutrition.",
        "Treatment": "No treatment required.",
        "Plant Health": "Healthy"
    },

    # ---------------- Tomato Diseases ----------------
    "Tomato_Bacterial_spot": {
        "Name": "Bacterial Spot",
        "Causes": "Caused by Xanthomonas campestris pv. vesicatoria.",
        "Prevention": "Use certified seeds, avoid overhead watering, rotate crops.",
        "Treatment": "Copper-based bactericides and removal of infected leaves.",
        "Plant Health": "Diseased"
    },
    "Tomato_Early_blight": {
        "Name": "Early Blight",
        "Causes": "Fungal disease caused by Alternaria solani.",
        "Prevention": "Crop rotation, avoid wet leaves, use resistant varieties.",
        "Treatment": "Apply fungicides (chlorothalonil, copper oxychloride).",
        "Plant Health": "Diseased"
    },
    "Tomato_Late_blight": {
        "Name": "Late Blight",
        "Causes": "Fungal-like pathogen Phytophthora infestans.",
        "Prevention": "Use disease-free seeds and avoid waterlogging.",
        "Treatment": "Use mancozeb or metalaxyl fungicides; remove infected plants.",
        "Plant Health": "Diseased"
    },
    "Tomato_Leaf_Mold": {
        "Name": "Leaf Mold",
        "Causes": "Fungal disease caused by Passalora fulva.",
        "Prevention": "Increase air circulation and avoid high humidity.",
        "Treatment": "Use sulfur or copper fungicides; prune infected leaves.",
        "Plant Health": "Diseased"
    },
    "Tomato_Septoria_leaf_spot": {
        "Name": "Septoria Leaf Spot",
        "Causes": "Fungal disease caused by Septoria lycopersici.",
        "Prevention": "Avoid overhead irrigation and rotate crops.",
        "Treatment": "Apply chlorothalonil or mancozeb fungicides.",
        "Plant Health": "Diseased"
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "Name": "Two-Spotted Spider Mite Infestation",
        "Causes": "Infestation by Tetranychus urticae mites.",
        "Prevention": "Maintain humidity and avoid drought stress.",
        "Treatment": "Use miticides or neem oil; remove infested leaves.",
        "Plant Health": "Diseased"
    },
    "Tomato_healthy": {
        "Name": "Healthy Tomato Leaf",
        "Causes": "None — plant is healthy.",
        "Prevention": "Ensure proper sunlight, watering, and nutrition.",
        "Treatment": "No treatment required.",
        "Plant Health": "Healthy"
    }
}

# -----------------------------
# GUI Setup
# -----------------------------
root = tk.Tk()
root.title("🌿 Plant Leaf Disease Detection")
root.geometry("1024x600")
root.configure(bg="gray25")

root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=3)
root.columnconfigure(2, weight=1)
root.rowconfigure(1, weight=1)

title = tk.Label(root, text="🌿 Plant Leaf Disease Detection",
                 bg="gray25", fg="white", font=("Helvetica", 18, "bold"))
title.grid(row=0, column=0, columnspan=3, pady=5)

canvas = tk.Label(root, bg="black", width=640, height=400)
canvas.grid(row=1, column=1, padx=10, pady=5)

disease_label = tk.Label(root, text="Status:\n\nAccuracy:\n\nRecommendation:",
                         bg="gray25", fg="white", font=("Helvetica", 12),
                         justify="left", wraplength=350)
disease_label.grid(row=1, column=2, sticky="n", padx=10, pady=5)

# -----------------------------
# Variables
# -----------------------------
cap = None
frame = None
captured_image = None
camera_running = False

# -----------------------------
# Functions
# -----------------------------
def exit_app():
    stop_camera()
    root.destroy()

def clear():
    global captured_image
    captured_image = None
    canvas.configure(image='')
    disease_label.config(text="Status:\n\nAccuracy:\n\nRecommendation:")

def preprocess_image(frame_to_analyse):
    img = cv2.resize(frame_to_analyse, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def analyse_frame(frame_to_analyse):
    img_input = preprocess_image(frame_to_analyse)
    interpreter.set_tensor(input_details[0]['index'], img_input)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    pred_idx = int(np.argmax(output_data))
    accuracy = round(float(np.max(output_data)) * 100, 2)
    class_name = class_names[pred_idx].strip()

    print("Predicted class:", class_name)

    # Flexible matching for similar names
    matched_key = None
    for key in disease_info.keys():
        if key.lower().replace(" ", "_") == class_name.lower().replace(" ", "_"):
            matched_key = key
            break
        # Partial match for robustness
        if class_name.lower() in key.lower() or key.lower() in class_name.lower():
            matched_key = key
            break

    info = disease_info.get(matched_key, {})

    if info:
        display_text = (
            f"Detected: {info.get('Name', class_name)}\n\n"
            f"Accuracy: {accuracy}%\n\n"
            f"Causes: {info.get('Causes')}\n\n"
            f"Prevention: {info.get('Prevention')}\n\n"
            f"Treatment: {info.get('Treatment')}\n\n"
            f"Plant Health: {info.get('Plant Health')}"
        )
    else:
        display_text = f"Detected: {class_name}\nAccuracy: {accuracy}%\n(No details found.)"

    disease_label.config(text=display_text)

def openphoto():
    global captured_image
    path = askopenfilename(filetypes=[("Image File", "*.jpg *.jpeg *.png")])
    if not path:
        return
    captured_image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    imgtk = ImageTk.PhotoImage(Image.fromarray(cv2.resize(captured_image, (640, 400))))
    canvas.imgtk = imgtk
    canvas.configure(image=imgtk)
    disease_label.config(text="Image loaded. Press Predict.")

def start_live():
    global cap, camera_running
    if not camera_running:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            disease_label.config(text="Cannot open camera")
            return
        camera_running = True
        update_frame()

def update_frame():
    global frame
    if camera_running and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imgtk = ImageTk.PhotoImage(Image.fromarray(rgb))
            canvas.imgtk = imgtk
            canvas.configure(image=imgtk)
        canvas.after(15, update_frame)
    else:
        stop_camera()

def capture_image():
    global captured_image, frame
    if frame is not None:
        captured_image = frame.copy()
        stop_camera()
        rgb = cv2.cvtColor(captured_image, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(Image.fromarray(rgb))
        canvas.imgtk = imgtk
        canvas.configure(image=imgtk)
        disease_label.config(text="Image captured. Press Predict.")

def predict():
    if captured_image is None:
        disease_label.config(text="Please capture or load an image first")
    else:
        analyse_frame(captured_image)

def stop_camera():
    global camera_running, cap
    camera_running = False
    if cap:
        cap.release()
        cap = None

# -----------------------------
# Buttons
# -----------------------------
button_frame = tk.Frame(root, bg="gray25")
button_frame.grid(row=2, column=0, columnspan=3, pady=10)

buttons = [
    ("Live", "green", start_live),
    ("Capture", "blue", capture_image),
    ("Predict", "orange", predict),
    ("Clear", "purple", clear),
    ("Stop", "red", stop_camera),
    ("Select Image", "brown", openphoto),
    ("Exit", "black", exit_app)
]

for text, color, cmd in buttons:
    tk.Button(button_frame, text=text, fg="white", bg=color,
              width=13, height=2, command=cmd).pack(side="left", padx=5)

# -----------------------------
# Run App
# -----------------------------
root.mainloop()
