import cv2
import face_recognition
import os
import pandas as pd
from datetime import datetime, timedelta

# Path for known faces and attendance file
KNOWN_FACES_DIR = "known_faces"
ATTENDANCE_FILE = "attendance.csv"

# Ensure known_faces directory exists
if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)
    print(f"[+] Created folder: {KNOWN_FACES_DIR}")

# Create attendance file if not exists
if not os.path.exists(ATTENDANCE_FILE):
    pd.DataFrame(columns=["Name", "Date", "Time"]).to_csv(ATTENDANCE_FILE, index=False)
    print(f"[+] Created file: {ATTENDANCE_FILE}")

# Dictionary to track last attendance time
last_attendance_time = {}

# Load known faces function
def load_known_faces():
    encodings = []
    names = []
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(KNOWN_FACES_DIR, filename)
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)
            if encoding:
                encodings.append(encoding[0])
                names.append(os.path.splitext(filename)[0])
    return encodings, names

# Mark attendance function
def mark_attendance(name):
    df = pd.read_csv(ATTENDANCE_FILE)
    today = datetime.now().strftime("%Y-%m-%d")
    time_now = datetime.now().strftime("%H:%M:%S")
    df = pd.concat([df, pd.DataFrame([{"Name": name, "Date": today, "Time": time_now}])], ignore_index=True)
    df.to_csv(ATTENDANCE_FILE, index=False)
    print(f"[✔] Attendance marked for {name} at {time_now}")

# Load initial faces
known_face_encodings, known_face_names = load_known_faces()

# Start webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            name = known_face_names[matches.index(True)]
            now = datetime.now()

            # Cooldown check (1 minutes)
            if name in last_attendance_time:
                if now - last_attendance_time[name] < timedelta(minutes=1):
                    # In cooldown - show yellow message
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)
                    cv2.putText(frame, f"{name} - Wait 1 min",
                                (left, top - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                    continue

            # Mark attendance and update time
            mark_attendance(name)
            last_attendance_time[name] = now

            # Green box for recognized face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        else:
            # Unknown face - Red box
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow("Face Attendance System (Press 's' to save new face, 'q' to quit)", frame)

    key = cv2.waitKey(1) & 0xFF

    # Save unknown face
    if key == ord('s'):
        if face_locations:
            name_input = input("Enter name for this person: ").strip()
            if name_input:
                top, right, bottom, left = face_locations[0]
                face_image = frame[top:bottom, left:right]
                save_path = os.path.join(KNOWN_FACES_DIR, f"{name_input}.jpg")
                cv2.imwrite(save_path, face_image)
                print(f"[+] Saved new face: {name_input}")

                # Reload model with new face
                known_face_encodings, known_face_names = load_known_faces()
                print("[✔] Model updated with new face")

    # Quit
    if key == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
