import cv2
import numpy as np
import face_recognition
import requests
import os
import firebase_admin
from firebase_admin import credentials, db, storage
import threading
import time
from test import test

# Firebase setup
cred = credentials.Certificate("smarthome-40f50-firebase-adminsdk-qq9ly-c96d530cd3.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://smarthome-40f50-default-rtdb.asia-southeast1.firebasedatabase.app/',
    'storageBucket': 'smarthome-40f50.appspot.com'
})

bucket = storage.bucket()

# Fungsi untuk membuka pintu
def unlock_door():
    ref = db.reference('OPIN_rZXLrMzEuJdnBO7e/devices/04/DoorLock_k5vj7/current')
    ref.update({"state": 1})
    print("Pintu terbuka.")

    connection_ref = db.reference('OPIN_rZXLrMzEuJdnBO7e/devices/04/DoorLock_k5vj7/connection')
    connection_ref.update({
        "timestamp": time.strftime('%m/%d/%Y, %I:%M:%S %p')
    })

# Inisialisasi
path = 'imagesFirebase'
images = []
classNames = []
encodeListKnown = []
known_files = set()
os.makedirs(path, exist_ok=True)

# Timer interval (dalam detik)
TIMER_INTERVAL = 60
def periodic_update():
    print("Memulai sinkronisasi dengan Firebase...")
    updateEncodings()
    print("Sinkronisasi selesai. Menunggu 10 menit sebelum sinkronisasi berikutnya...")

    # Set timer untuk memanggil fungsi ini lagi setelah TIMER_INTERVAL
    threading.Timer(TIMER_INTERVAL, periodic_update).start()

def download_images_from_firebase():
    global known_files
    blobs = list(bucket.list_blobs(prefix='identitas/'))  # Ubah iterator menjadi list

    for blob in blobs:
        if not blob.name.endswith(('.jpg', '.png')):
            continue

        file_name = os.path.basename(blob.name)
        if file_name in known_files:
            continue

        local_path = os.path.join(path, file_name)
        blob.download_to_filename(local_path)
        known_files.add(file_name)
        print(f"File baru diunduh: {file_name}")

# Fungsi untuk menemukan encoding
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)
        if encode:
            encodeList.append(encode[0])
    return encodeList


def updateEncodings():
    global images, classNames, encodeListKnown
    download_images_from_firebase()
    images.clear()
    classNames.clear()

    new_images = []
    new_class_names = []

    for file_name in os.listdir(path):
        curImg = cv2.imread(f'{path}/{file_name}')
        if curImg is not None:
            new_images.append(curImg)
            new_class_names.append(os.path.splitext(file_name)[0])

    encodeListKnown = findEncodings(new_images)
    images = new_images
    classNames = new_class_names

    print(f"Encoding diperbarui untuk file: {new_class_names}")

print("Memulai program dengan sinkronisasi awal...")
updateEncodings()
periodic_update()

# url = "http://192.168.151.161/capture"  # Ganti IP sesuai dengan IP ESP32-CAM
cap = cv2.VideoCapture(0)  # Menggunakan webcam sebagai input

threshold = 0.55

# Face Recognition Loop
while True:
    time.sleep(3)
    # Jika menggunakan ESP32-CAM
    # response = requests.get(url, stream=True)
    # img_array = np.array(bytearray(response.content), dtype=np.uint8)
    # img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    #
    # if img is None:
    #     print("Gagal memuat gambar dari ESP32-CAM")
    #     continue

    # Jika menggunakan webcam
    ret, img = cap.read()

    if not ret:
        print("Gagal mengambil gambar")
        continue

    key = cv2.waitKey(1)

    # Simpan gambar sementara untuk deteksi spoofing
    cv2.imwrite('src/img.png', img)
    img_for_spoofing = cv2.imread('src/img.png')

    # Deteksi spoofing
    spoofing_label = test(
        img_for_spoofing,
        model_dir="resources/anti_spoof_models",
        device_id=0
    )

    if spoofing_label == 1:  # Wajah asli terdeteksi
        print("Wajah asli, melanjutkan ke pengenalan wajah.")
    else:
        print("Deteksi wajah palsu! Login ditolak.")
        continue

    # Pengenalan wajah
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

        # Bounding box
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

        cv2.rectangle(img, (x1, y1 - 50), (x2, y2 + 30), color, 2)
        cv2.rectangle(img, (x1, y2), (x2, y2 + 30), color, cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 + 24), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    # Jeda untuk menghindari pemeriksaan folder terlalu sering
    time.sleep(2)
