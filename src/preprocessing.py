import numpy as np
import cv2
from PIL import Image

####################아래는 원본이미지에만 적용할 전처리####################
# 엣지 보강 (샤프닝)
def enhance_edges(image):
    image = np.array(image)  # PIL Image -> NumPy 배열로 변환
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return Image.fromarray(cv2.filter2D(image, -1, kernel))  # NumPy 배열에서 필터링 후 PIL 이미지로 변환

# 채도 강화
def enhance_saturation(image, scale=1.5):
    image = np.array(image)  # PIL Image -> NumPy 배열로 변환
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # RGB -> BGR로 변환
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.multiply(s, scale)
    s = cv2.min(s, 255)
    hsv = cv2.merge((h, s, v))
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)  # BGR -> HSV로 변환 후 다시 BGR로
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR -> RGB 변환
    return Image.fromarray(image)  # 결과를 PIL 이미지로 변환

# CLAHE
def enhance_clahe(image):
    image = np.array(image)  # PIL Image -> NumPy 배열로 변환
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # RGB -> BGR로 변환
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(2, 2))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # LAB -> BGR로 변환 후
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR -> RGB 변환
    return Image.fromarray(image)  # 결과를 PIL 이미지로 변환