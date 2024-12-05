import cv2
import numpy as np
import face_recognition
import requests
import os
from test import test

path = 'images'
images = []
classNames = []
myList = os.listdir(path)

# Memuat gambar dan nama dari direktori 'images'
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

url="http://192.168.8.198/capture"

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

    cv2.imshow('Face Recognition', img)

    if key == ord('l'):  # 'l' untuk Login
        # Simpan gambar dari webcam untuk pengecekan anti-spoofing
        cv2.imwrite('img.png', img)

        img_for_spoofing = cv2.imread('img.png')

        # Implementasi pengecekan anti-spoofing
        spoofing_label = test(img_for_spoofing,
                              model_dir="F:/AI Computer Vision/OPIN/Silent-Face-Anti-Spoofing/resources/anti_spoof_models",
                              device_id=0
                              )

        if spoofing_label == 1:  # Jika wajah asli terdeteksi
            print("Wajah asli, melanjutkan ke pengenalan wajah.")
        else:
            print("Deteksi wajah palsu! Login ditolak.")
            continue  # Kembali ke loop awal jika wajah palsu terdeteksi

        # Proses Face Recognition setelah pengecekan anti-spoofing berhasil
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

            # Tampilkan bounding box
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

            cv2.rectangle(img, (x1, y1 - 50), (x2, y2 + 30), color, 2)
            cv2.rectangle(img, (x1, y2), (x2, y2 + 30), color, cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 + 24), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Face Recognition', img)
    elif key == ord('q'):  # 'q' untuk keluar dari aplikasi
        break

cap.release()
cv2.destroyAllWindows()
