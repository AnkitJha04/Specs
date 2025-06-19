# Import necessary modules
import speech_recognition as sr  # Speech recognition
import os  # OS operations
import sys  # System functions
import time  # Delays
import threading  # Multi-threading for speech and video processing
from colorama import Fore, Style  # Colored terminal output
import pyttsx3  # Text-to-speech conversion
import cv2  # OpenCV for video feed and overlay text
import numpy as np
from tensorflow.keras.models import load_model
import os
from datetime import datetime
from collections import deque
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk

# Function to convert text to speech (non-blocking)
def speak(text):  # Uses pyttsx3 to convert text to speech in a separate thread to ensure it does not block execution of the program.
    def _speak():
        engine = pyttsx3.init()
        engine.setProperty("rate", 180)  # Adjust speech speed
        engine.setProperty("volume", 1.0)  # Set volume
        engine.say(text)
        engine.runAndWait()
    thread = threading.Thread(target=_speak) # Running speech synthesis in a separate thread
    thread.start()

# Function to display video feed with transcript overlay
def display_video(): # Captures video feed using OpenCV and overlays the live speech transcription on the screen with a black background for visibility.
    global current_transcription
    cap = cv2.VideoCapture(0)  # Open webcam
    while True:
        ret, frame = cap.read()  # Capture frame from webcam
        if not ret:
            # Display an error message if the camera is not accessible
            cv2.putText(frame, "Error Opening Camera", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 0, 255), 2, cv2.LINE_AA)
            continue
        # Text display settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_color = (0, 255, 0)  # Green text
        thickness = 2
        # Split transcript into multiple lines for better readability
        words = current_transcription.split()
        max_words_per_line = 8
        lines = [" ".join(words[i: i + max_words_per_line]) for i in range(0, len(words), max_words_per_line)]
        # Positioning of text on the screen
        x, y = 20, 50  # Starting position for the text
        box_padding = 10
        line_height = 30
        # Calculate dimensions of the black background box
        box_width = 500
        box_height = len(lines) * line_height + 20  # Adjust height based on number of lines
        frame = cv2.flip(frame, 1) # flipped video feed
        overlay = frame.copy()
        # Draw a black rectangle behind the text for better readability
        cv2.rectangle(overlay, (x - box_padding, y - box_padding),(x + box_width, y + box_height),(0, 0, 0), thickness=cv2.FILLED)
        # Blend the overlay with transparency
        alpha = 0.5  # Transparency level
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        # Overlay text on the video frame
        for line in lines[-5:]:  # Display last 5 lines only
            cv2.putText(frame, line, (x, y), font, font_scale, font_color, thickness)
            y += line_height  # Move text down for the next line
        cv2.imshow("Live Transcription", frame) # Display the updated video feed
        if cv2.waitKey(1) & 0xFF == ord("q"): # Press 'q' to exit the video feed
            break
    cap.release()
    cv2.destroyAllWindows()

# Function to perform real-time speech transcription
def transcriber():  # Uses the speech_recognition library to continuously listen for speech,transcribe it into text, display it, and store it in a log file.
    global current_transcription
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    print(Fore.YELLOW + "Real-Time Speech Transcription is starting!" + Style.RESET_ALL)
    time.sleep(1)
    os.system("cls" if os.name == "nt" else "clear")
    accumulated_transcription = ""  # Store full transcription history

    with mic as source:
        # Adjust for background noise to improve recognition
        recognizer.adjust_for_ambient_noise(source, duration=0.2)
        recognizer.pause_threshold = 0.5  # Controls when to stop listening
        recognizer.non_speaking_duration = 0.2  # Delay before stopping listening
        print(Fore.CYAN + "Listening... Speak now!" + Style.RESET_ALL)
        while True:
            try:
                # Listen to audio input (timeout ensures quick response)
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=8)
                # Convert speech to text using Google's API
                transcript = recognizer.recognize_google(audio).lower()
                # Clear previous output and print new transcription
                os.system("cls" if os.name == "nt" else "clear")
                print(Fore.GREEN + "Transcription:\n" + Style.RESET_ALL)
                print(Fore.LIGHTMAGENTA_EX + transcript + Style.RESET_ALL)
                # Update global variable for OpenCV overlay
                current_transcription = transcript
                speak(transcript)  # Convert text to speech
                # Store complete conversation
                accumulated_transcription += transcript + "\n"
                # Stop transcription if user says "stop transcribing"
                if "stop transcribing" in transcript:
                    print(Fore.YELLOW + "\nTranscription stopped. Exiting..." + Style.RESET_ALL)
                    with open("transcription_log.txt", "w") as log_file:
                        log_file.write(accumulated_transcription)  # Save log file
                    break
            except sr.WaitTimeoutError:
                print(Fore.YELLOW + "No speech detected. Try speaking again..." + Style.RESET_ALL)
            except sr.UnknownValueError:
                print(Fore.RED + "Didn't catch that, try again..." + Style.RESET_ALL)
            except sr.RequestError as e:
                print(Fore.RED + f"Speech recognition error: {e}" + Style.RESET_ALL)
                break
            except KeyboardInterrupt:
                print(Fore.YELLOW + "\nTranscription interrupted. Exiting..." + Style.RESET_ALL)
                sys.exit()
    # Store the entire transcription in the global variable
    current_transcription = accumulated_transcription
    print(Fore.GREEN + "\nFinal Transcription Log:" + Style.RESET_ALL)
    print(Fore.LIGHTBLUE_EX + accumulated_transcription + Style.RESET_ALL)

# Function for doing interpretations
def interpreter():
    # Load pre-trained CNN model
    model = load_model("C:/Users/ankit/OneDrive/Documents/Personal/Repls/Resource files/asl.h5")

    # Labels (A-Z)
    labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "nothing", "space"]
    labels = np.array(sorted(labels))

    # ROI settings
    ROI_START = (100, 100)
    ROI_END = (300, 300)
    IMG_SIZE = 64

    # Prediction log and word builder
    prediction_log = deque(maxlen=15)

    # GUI setup
    gui = tk.Tk()
    gui.title("ASL Recognition GUI")
    gui.geometry("900x500")

    # Webcam feed label
    video_label = Label(gui)
    video_label.pack(side="left", padx=10, pady=10)

    # Prediction labels
    frame_right = tk.Frame(gui)
    frame_right.pack(side="right", fill="both", expand=True, padx=10)

    letter_label = Label(frame_right, text="Predicted Letter: ", font=("Arial", 18))
    letter_label.pack(pady=20)

    word_label = Label(frame_right, text="Constructed Word: ", font=("Arial", 18))
    word_label.pack(pady=20)

    # Preprocess frame
    def preprocess(frame):
        # Crop the ROI in BGR (color) directly
        roi = frame[ROI_START[1]:ROI_END[1], ROI_START[0]:ROI_END[0]]
        resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
        normalized = resized.astype('float32') / 255.0
        # Convert BGR to RGB as model might expect RGB order
        rgb_img = cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB)
        return np.expand_dims(rgb_img, axis=0)  # shape (1, 64, 64, 3)


    # Update GUI labels
    def update_gui(predicted_letter, word):
        letter_label.config(text=f"Predicted Letter: {predicted_letter}")
        word_label.config(text=f"Constructed Word: {word}")

    # Update frames
    def show_frame():
        global current_word
        ret, frame = cap.read()
        if not ret:
            gui.after(10, show_frame)
            return

        frame = cv2.flip(frame, 1)
        cv2.rectangle(frame, ROI_START, ROI_END, (0, 255, 0), 2)

        # Predict
        preprocessed = preprocess(frame)
        prediction = model.predict(preprocessed, verbose=0)

        # Debug prints to check prediction shape and index
        print("Prediction shape:", prediction.shape)
        predicted_index = np.argmax(prediction)
        print("Predicted index:", predicted_index)
        print("Number of labels:", len(labels))

        # Safely get predicted label
        if predicted_index >= len(labels):
            print(f"Warning: predicted index {predicted_index} out of label range")
            predicted_label = "?"
            confidence = 0.0
        else:
            predicted_label = labels[predicted_index]
            confidence = np.max(prediction) * 100

        # Logic to build word
        prediction_log.append(predicted_label)
        if prediction_log.count(predicted_label) > 10:
            if not current_word.endswith(predicted_label):
                current_word += predicted_label

        update_gui(predicted_label, current_word)

        # Show predicted letter on ROI
        cv2.putText(frame, f"{predicted_label} ({confidence:.1f}%)",
                    (ROI_START[0], ROI_END[1] + 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 0, 0), 2)

        # Convert frame to ImageTk
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

        gui.after(10, show_frame)

    # Start webcam
    cap = cv2.VideoCapture(0)

    # Begin GUI loop and show video
    show_frame()
    gui.mainloop()

    # Release camera after GUI closes
    cap.release()
    cv2.destroyAllWindows()

# Initialize global variable for storing the current transcription text
current_transcription = ""
current_word = ""

# Main program execution
print("Hello! What would you like me to do?")
print("1. Real-time Transcribing \n2. Interpret Sign Language")
print("**ENTER ONLY THE SERIAL NUMBER OF THE TASK YOU WANT ME TO PERFORM**")
task = input("Your Input: ")
if task == "1":
    print("Starting real-time transcription...")
    time.sleep(1)
    os.system("cls" if os.name == "nt" else "clear")
    # Start the video display in a separate thread
    video_thread = threading.Thread(target=display_video, daemon=True)
    video_thread.start()
    transcriber()  # Run the transcription function
elif task == "2":
    print("Starting real-time interpretation...")
    time.sleep(2)
    os.system("cls" if os.name == "nt" else "clear")
    interpreter()   # run the interpretation function
    time.sleep(1)
    os.system("cls" if os.name == "nt" else "clear")
else:
    print("Invalid input, please retry.")
    time.sleep(1)
    os.system("cls" if os.name == "nt" else "clear")
    sys.exit() 