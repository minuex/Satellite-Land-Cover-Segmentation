import os
import numpy as np
import rasterio
from PIL import Image

"""
언리얼 프로세스에 사용될 형태로 가공하는 코드
[0,1,2,3,4] -> 각 클래스 별 [0,255]의 레이어 png 형태로 저장
"""

input_folder = "predictions"  # 분류된 래스터 데이터가 있는 폴더
output_folder = "binary_layers"  # 개별 레이어를 저장할 폴더
os.makedirs(output_folder, exist_ok=True)

raster_files = [f for f in os.listdir(input_folder) if f.endswith(".tif")]

# 클래스 목록 (0: 건물, 1: 도로, 2: 농경지, 3: 숲, 4: 배경)
class_names = ["Building", "Road", "Field", "Forest", "Background"]

for raster_file in raster_files:
    raster_path = os.path.join(input_folder, raster_file)

    with rasterio.open(raster_path) as src:
        classification = src.read(1)  # 첫 번째(유일한) 밴드 읽기

    # 각 클래스별 이진화된 PNG 저장
    for class_value, class_name in enumerate(class_names):
        binary_layer = (classification == class_value).astype(np.uint8) * 255

        # PNG 파일 저장 경로 설정하기
        output_path = os.path.join(output_folder, f"{os.path.splitext(raster_file)[0]}_{class_name}.png")

        # PNG로 저장
        image = Image.fromarray(binary_layer)
        image.save(output_path)

        print(f"{output_path} 저장 완료!")

