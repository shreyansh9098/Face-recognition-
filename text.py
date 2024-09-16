print("1")
import cv2

    
import face_recognition
import numpy as np
import os 
from datetime import datetime

# Path to the directory containing images of known people
KNOWN_FACES_DIR ='known_faces'
# Threshold for face recognition
TOLERANCE = 0.6
# Font for displaying names on the image
FONT = cv2.FONT_HERSHEY_SIMPLEX
# Scale factor for resizing the webcam image
SCALE_FACTOR =0.25

def get_encoded_faces():
    """
    Function to encode known faces
    """
    encoded = {}
    for dirpath, dnames, frames in os.walk(KNOWN_FACES_DIR):
        for f in frames:
            if f.endswith(".jpg") or f.endswith(".png"):
                face = face_recognition.load_image_file(f"{dirpath}/{f}")
                encoding = face_recognition.face_encodings(face)[0]
                encoded[f.split('.')[0]] = encoding 
    return encoded
def mark_attendance(name):
    """
    Function to mark attendance in a CSV file
    """
    with open('attendance.csv', 'a+') as f:
        data_list = f.readlines()
        names = []
        for line in data_list:
            entry = line.split('.')
            names.append(entry[0])
        if name not in names:
            now = datetime.now()
            date_string = now.strftime('%Y-%m-%d')
            time_string = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{date_string},{time_string}')
            

def recognize_faces(frame, known_faces):
    """
    Function to recognize faces in the frame
    """
    small_frame = cv2.resize(frame, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_small_frame)
    print(face_locations)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    for face_encoding in range(len(face_encodings)):
        matches = face_recognition.compare_faces(list(known_faces.values()), face_encodings[face_encoding], tolerance=TOLERANCE)
        name = "Unknown"
        face_distances = face_recognition.face_distance(list(known_faces.values()), face_encodings[face_encoding])
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = list(known_faces.keys())[best_match_index].upper()
            mark_attendance(name)
        text_size, _ = cv2.getTextSize(name, FONT, 1, 3)
        

        y1, x2, y2, x1 = [int(x / SCALE_FACTOR) for x in face_locations[face_encoding]]
        text_x = int(x1 + ((x2-x1) - text_size[0]) / 2)
      
        cv2.rectangle(frame, (x1-30, y1-30), (x2+10, y2+20), (0, 255, 0), 2)
        cv2.putText(frame, name, (text_x,y2), FONT, 1, (0,0, 255), 3,)
    return frame

if __name__ == "__main__":
    known_faces = get_encoded_faces()
    #print(known_faces)
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        try:
            frame = recognize_faces(frame, known_faces)
        except Exception as e:
            print(e)
        cv2.imshow('Face Recognition Based Attendance System', frame,)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
        
            
                                    