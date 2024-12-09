import cv2
import numpy as np
import face_recognition
import requests
import os
import firebase_admin
from firebase_admin import credentials, db
import time
from test import test

cred = credentials.Certificate("smarthome-40f50-firebase-adminsdk-qq9ly-c96d530cd3.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://smarthome-40f50-default-rtdb.asia-southeast1.firebasedatabase.app/'
})


def unlock_door():
    ref = db.reference('OPIN_rZXLrMzEuJdnBO7e/devices/04/DoorLock_k5vj7/current')

    ref.update({"state": 1})
    print("Pintu terbuka.")

    connection_ref = db.reference('OPIN_rZXLrMzEuJdnBO7e/devices/04/DoorLock_k5vj7/connection')
    connection_ref.update({
        "timestamp": time.strftime('%m/%d/%Y, %I:%M:%S %p')
    })

path = 'images'
images = []
classNames = []
myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)
        if encode:
            encodeList.append(encode[0])
    return encodeList

encodeListKnown = findEncodings(images)
print('Encoding Complete')

url="http://192.168.151.161/capture"

cap = cv2.VideoCapture(0)  # Menggunakan webcam sebagai input
threshold = 0.55

while True:
    response = requests.get(url, stream=True)
    img_array = np.array(bytearray(response.content), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        print("Gagal memuat gambar dari ESP32-CAM")
        continue

    key = cv2.waitKey(1)
    # ret, img = cap.read()

    # if not ret:
    #     print("Gagal mengambil gambar")
    #     continue

    # cv2.imshow('Face Recognition', img)

    cv2.imwrite('src/img.png', img)

    img_for_spoofing = cv2.imread('src/img.png')

    spoofing_label = test(img_for_spoofing,
                          model_dir="resources/anti_spoof_models",
                          device_id=0
                          )

    if spoofing_label == 1:  # Jika wajah asli terdeteksi
        print("Wajah asli, melanjutkan ke pengenalan wajah.")
    else:
        print("Deteksi wajah palsu! Login ditolak.")
        continue

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        matchIndex = np.argmin(faceDis)

        if faceDis[matchIndex] < threshold:
            name = classNames[matchIndex].capitalize()
        else:
            name = "Unknown"

        print(name)

        if name != "Unknown":
            print(f"Wajah {name} dikenali. Membuka pintu...")
            unlock_door()

        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

        cv2.rectangle(img, (x1, y1 - 50), (x2, y2 + 30), color, 2)
        cv2.rectangle(img, (x1, y2), (x2, y2 + 30), color, cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 + 24), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        time.sleep(3)

cap.release()
cv2.destroyAllWindows()
