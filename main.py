import cv2
from cvzone.HandTrackingModule import HandDetector      # Hand Detection
from cvzone.ClassificationModule import Classifier      # Importing Classifier
import webbrowser as wb
import numpy as np
import math
import tkinter as tk
from tkinter import Label, messagebox
from PIL import Image, ImageTk

# Initialize the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

detector = HandDetector(maxHands=1)  # Detect only one hand
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 10  # The value for spacing for exactly cropping the image
imgSize = 300  # The fixed size of the image

# URL dictionary for gestures
'''url_dict = {"A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"}'''

url_dict = {
    "A": "https://amazon.com", "B": "https://bookmyshow.com", "C": "https://chatgpt.com", "D": "https://duolingo.com", "E": "https://adobe.com",
    "F": "https://facebook.com", "G": "https://google.com", "H": "https://hotstar.com", "I": "https://instagram.com", "J":"https://jiocinema.com",
    "K": "https://primevideo.com", "L": "https://linkedin.com", "M": "https://myntra.com", "N": "https://netflix.com", "O": "https://outlook.com",
    "P": "https://pinterest.com", "Q": "https://glassdoor.com", "R": "https://rabbitresearch.com", "S": "https://spotify.com", "T": "https://twitter.com",
    "U": "https://udemy.com", "V": "https://sonyliv.com", "W": "https://whatsapp.com", "X": "https://nykaa.com", "Y": "https://www.youtube.com/", 
    "Z":"https://zee5.com"
}


# Initialize the main window
root = tk.Tk()
root.title("SIGNWEB")

window_height = 650
window_width = 654

def center_screen():
    """ gets the coordinates of the center of the screen """
    global screen_height, screen_width, x_cordinate, y_cordinate
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
        # Coordinates of the upper left corner of the window to make the window appear in the center
    x_cordinate = int((screen_width/2) - (window_width/2))
    y_cordinate = int((screen_height/2) - (window_height/2))
    root.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))
center_screen()

#Define the PhotoImage Constructor by passing the image file
img= ImageTk.PhotoImage(file='Images/app_bg.jpeg')
img_label= Label(root,image=img)
#Define the position of the image
img_label.place(x=0, y=0)

# Frame for Webcam Input
frame = tk.Label(root)
frame.pack()

# Label for displaying the prediction
label_prediction = tk.Label(root, text="SignWeb", font=("Comic Sans MS", 26),padx=15,pady=6, bg='bisque', borderwidth=2, relief="ridge")
label_prediction.pack()

# For handling the opening of url
def handle_url_open(label):
    """Open URL and ask if user wants to continue."""
    if label in url_dict:
        wb.open_new_tab(url_dict[label])
        if messagebox.askyesno("Continue", "Do you want to continue ?"):
            return True
        else:
            exit_application()
            return False
    return True

# Function to update the webcam and prediction
def update_frame():
    if not cap.isOpened():
        return                                  # Do not proceed if the webcam is not open
    success, img = cap.read()
    if success:
        img = cv2.flip(img, 1)                  # Flip the image for natural interaction
        hands, img = detector.findHands(img)    # Detect hands

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            aspectRatio = h / w

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            label = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"[index]
            label_prediction.config(text=f'Predicted Gesture: {label}')

            # Ask user after opening URL
            if not handle_url_open(label):
                return

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        frame.imgtk = imgtk
        frame.configure(image=imgtk)
    frame.after(10, update_frame)                 # Call the same function after 10 milliseconds

def exit_application():
    """Close all resources and exit."""
    if cap.isOpened():
        cap.release()
    root.quit()

# Button to start/continue recognition
btn_start = tk.Button(root, text="Start Recognition", command=lambda: update_frame(),fg="black",padx=10,pady=2, bg='bisque', activebackground='sky blue')
btn_start.pack(padx=20, pady=20)

# Button to exit the application
btn_exit = tk.Button(root, text="Exit", command=exit_application, bg='bisque', activebackground='red',padx=18)
btn_exit.pack(padx=20, pady=4)

# Start the GUI
root.mainloop()
