# -*- coding: utf-8 -*-
# @Time : 20-6-9 下午3:06
# @Author : zhuying
# @Company : Minivision
# @File : test.py
# @Software : PyCharm

import os
import cv2
import numpy as np
import warnings
import time

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

warnings.filterwarnings('ignore')

# 1 untuk wajah asli (bukan spoofing)
# 2 untuk wajah palsu (spoofing)

def check_image(image):
    height, width, channel = image.shape
    if not np.isclose(width / height, 3 / 4, atol=1e-2):
        print("Image is not appropriate!!!\nWidth/Height should be 3/4.")
        return False
    else:
        return True

def test(image, model_dir, device_id):
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()

    # Menyesuaikan ukuran gambar agar sesuai dengan rasio 3:4
    original_height, original_width = image.shape[:2]
    new_width = int(original_height * 3 / 4)
    image = cv2.resize(image, (new_width, original_height))

    # Memeriksa apakah gambar sesuai dengan rasio yang diinginkan
    result = check_image(image)
    if not result:
        return None  # Mengembalikan None jika rasio tidak sesuai

    # Memeriksa apakah model_dir valid
    if not os.path.exists(model_dir):
        print(f"Error: Direktori model tidak ditemukan di path {model_dir}")
        return None

    if not os.path.isdir(model_dir):
        print(f"Error: Path {model_dir} bukan sebuah direktori")
        return None

    # Mendapatkan bounding box dari wajah dalam gambar
    image_bbox = model_test.get_bbox(image)
    if image_bbox is None:
        print("Error: Tidak ada wajah yang terdeteksi dalam gambar.")
        return None

    prediction = np.zeros((1, 3))
    test_speed = 0

    # Menjumlahkan prediksi dari setiap model dalam direktori model_dir
    for model_name in os.listdir(model_dir):
        model_path = os.path.join(model_dir, model_name)
        if not os.path.isfile(model_path):
            continue  # Lewati jika bukan file

        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False

        img = image_cropper.crop(**param)
        if img is None:
            print(f"Warning: Tidak dapat crop gambar untuk model {model_name}")
            continue

        start = time.time()
        prediction += model_test.predict(img, model_path)
        test_speed += time.time() - start

    label = np.argmax(prediction)
    value = prediction[0][label] / 2
    if label == 1:
        print("Real Face detected. Score: {:.2f}.".format(value))
    else:
        print("Fake Face detected. Score: {:.2f}.".format(value))
    print("Prediction cost {:.2f} s".format(test_speed))

    return label