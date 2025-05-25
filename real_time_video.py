from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np

# parameters for loading data and images
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'

# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry" ,"disgust","fear", "happy", "sad", "surprised", "neutral"]

# start video stream
cv2.namedWindow('Emotion Detector', cv2.WINDOW_NORMAL)
camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    frameClone = frame.copy()
    label = ""

    if len(faces) > 0:
        faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces

        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        preds = emotion_classifier.predict(roi)[0]
        label = EMOTIONS[preds.argmax()]

        # Draw creative face detection overlay (rectangle with rounded corners feel)
        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (147, 112, 219), 3)
        # Removed the nose circle

        cv2.putText(frameClone, label.upper(), (fX, fY - 15),
                    cv2.FONT_HERSHEY_PLAIN, 2.0, (0, 255, 255), 3)

    # Title Bar
    cv2.rectangle(frameClone, (0, 0), (500, 40), (20, 20, 20), -1)
    cv2.putText(frameClone, "Real-Time Emotion Recognition", (10, 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 0), 2)

    # Show single output window
    cv2.imshow('Emotion Detector', frameClone)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
