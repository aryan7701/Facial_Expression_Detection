import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import model_from_json, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# load json and create model
# json_file = open('model/emotion_model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# emotion_model = model_from_json(loaded_model_json)

# Loading the model
with open('emotion_model.json', 'r') as json_file:
    loaded_model_json = json_file.read()

emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights('model/emotion_model.h5')

# load weights into new model
emotion_model.load_weights("model/emotion_model.h5")
print("Loaded model from disk")

# start the webcam feed
#cap = cv2.VideoCapture(0)

# pass here your video path
# you may download one from here : https://www.pexels.com/video/three-girls-laughing-5273028/
# cap = cv2.VideoCapture("D:\GitUpload\Emotion_detection_with_CNN-main\couple-surprised-.mp4") #error one
# cap = cv2.VideoCapture(r"D:\GitUpload\Emotion_detection_with_CNN-main\couple-surprised-.mp4")
# or
# cap = cv2.VideoCapture("D:/GitUpload/Emotion_detection_with_CNN-main/man-laughing.mp4")


# while True:
#     # Find haar cascade to draw bounding box around face
#     ret, frame = cap.read()
#     # frame = cv2.resize(frame, (1280, 720))
#     cap = cv2.VideoCapture(r"D:\GitUpload\Emotion_detection_with_CNN-main\man-laughing.mp4")
#     if not cap.isOpened():
#         print("Error: Could not open video file.")
#         exit()

#     if not ret:
#         break
#     face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # detect faces available on camera
#     num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

#     # take each face available on the camera and Preprocess it
#     for (x, y, w, h) in num_faces:
#         cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
#         roi_gray_frame = gray_frame[y:y + h, x:x + w]
#         cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

#         # predict the emotions
#         emotion_prediction = emotion_model.predict(cropped_img)
#         maxindex = int(np.argmax(emotion_prediction))
#         cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

#     cv2.imshow('Emotion Detection', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

import cv2
import numpy as np


# Initialize video capture
cap = cv2.VideoCapture("D:/GitUpload/Emotion_detection_with_CNN-main/man-laughing.mp4")

# Check if video file opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Load the Haar cascade classifier for face detection
face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

while True:
    # Read frame from video
    ret, frame = cap.read()
    
    # Break the loop if no frame is returned (end of video)
    if not ret:
        break
    
    # Convert frame to grayscale for face detection
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # Process each face found
    for (x, y, w, h) in num_faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        
        # Extract the region of interest (ROI) for emotion prediction
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # Predict the emotion
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        
        # Put the predicted emotion text on the frame
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the frame with emotion detection
    cv2.imshow('Emotion Detection', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()


# Compile the model (same as when it was trained)
emotion_model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

# Prepare the test data generator
test_data_gen = ImageDataGenerator(rescale=1./255)

# Preprocess all test images
test_generator = test_data_gen.flow_from_directory(
    "D:/GitUpload/Emotion_detection_with_CNN-main/data/test",
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical')

# Evaluate the model
loss, accuracy = emotion_model.evaluate(test_generator)
print(f'Test accuracy: {accuracy * 100:.2f}%')

