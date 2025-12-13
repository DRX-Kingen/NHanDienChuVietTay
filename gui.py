import tkinter as tk
import numpy as np
import cv2
import os
from PIL import Image, ImageGrab
from tensorflow.keras.models import load_model


try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1)
except:
    pass


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.keras")
model = load_model(MODEL_PATH)


def predict_digit(img):
    img = cv2.resize(img, (28, 28))
    img = img.astype("float32") / 255.0
    img = img.reshape(1, 28, 28, 1)

    pred = model.predict(img)
    label = np.argmax(pred)
    conf = np.max(pred)
    return chr(label + 65), conf


window = tk.Tk()
window.title("Handwriting Recognition (A–Z)")
window.geometry("420x560")
window.configure(bg="#f2f2f7")


tk.Label(
    window,
    text="Handwritten Letter Recognition",
    font=("Arial", 16, "bold"),
    bg="#f2f2f7"
).pack(pady=(10, 5))



canvas_frame = tk.Frame(
    window, bg="#ffffff", highlightbackground="#bfbfbf", highlightthickness=1
)
canvas_frame.pack(pady=10)

canvas = tk.Canvas(canvas_frame, width=300, height=300, bg="white")
canvas.pack()

lastx, lasty = None, None


def xy(event):
    global lastx, lasty
    lastx, lasty = event.x, event.y


def addLine(event):
    global lastx, lasty
    canvas.create_line(
        (lastx, lasty, event.x, event.y),
        width=14, fill="black",
        capstyle=tk.ROUND, smooth=True
    )
    lastx, lasty = event.x, event.y


canvas.bind("<Button-1>", xy)
canvas.bind("<B1-Motion>", addLine)



def predict():
    x1 = window.winfo_rootx() + canvas.winfo_x()
    y1 = window.winfo_rooty() + canvas.winfo_y()
    x2 = x1 + canvas.winfo_width()
    y2 = y1 + canvas.winfo_height()

    img = ImageGrab.grab(bbox=(x1, y1, x2, y2)).convert("L")
    img = np.array(img)

    img = 255 - img  

    letter, conf = predict_digit(img)
    result_label.config(text=f"Prediction: {letter}\nAccuracy: {conf*100:.2f}%")


def clear():
    canvas.delete("all")
    result_label.config(text="Draw a letter (A–Z)")



button_frame = tk.Frame(window, bg="#f2f2f7")
button_frame.pack(pady=10)

btn_predict = tk.Button(
    button_frame, text="Predict", command=predict,
    bg="#007aff", fg="white",
    activebackground="#005ec2",
    font=("Arial", 12), width=10
)
btn_predict.grid(row=0, column=0, padx=10)

btn_clear = tk.Button(
    button_frame, text="Clear", command=clear,
    bg="#ff3b30", fg="white",
    activebackground="#c12722",
    font=("Arial", 12), width=10
)
btn_clear.grid(row=0, column=1, padx=10)



result_label = tk.Label(
    window,
    text="Draw a letter (A–Z)",
    font=("Arial", 14),
    bg="#f2f2f7"
)
result_label.pack(pady=15)

window.mainloop()
