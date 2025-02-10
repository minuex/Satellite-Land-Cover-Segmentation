import torch
import os

# 클래스 매핑 설정
class_mapping = {
    10: 0,  # 건물
    30: 1,  # 도로
    50: 2,  # 경작지
    60: 2,  # 경작지
    71: 3,  # 숲
    75: 3,  # 숲
    70: 3,  # 숲
    90: 2,  # 농경지
    100: 4  # 배경
}

class_names = ["Building", "Road", "Field", "Forest", "Background"]

# train parameters
batch_size = 8
num_epochs = 100
num_classes = 5
learning_rate = 0.0001
early_stopping_patience = 10

save_dir = 'models/saved_checkpoints'
os.makedirs(save_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
