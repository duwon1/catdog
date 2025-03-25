import torch
from PIL import Image
import os
from torchvision import transforms
from models.cnn_model import CatDogCNN  # 모델 클래스 정의

# GPU 사용 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 이미지 폴더 경로 지정
image_folder = 'data/test'  # 예시: test 데이터가 있는 폴더 경로
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'))]

# 모델 불러오기
model = CatDogCNN()
model.load_state_dict(torch.load('cat_dog_model.pth'))  # 저장된 모델 불러오기
model = model.to(device)
model.eval()

# 각 이미지를 불러와 예측
for image_file in image_files:
    img_path = os.path.join(image_folder, image_file)

    # 이미지 로딩
    img = Image.open(img_path)

    # 이미지 전처리
    img = transform(img).unsqueeze(0)  # 배치 차원 추가

    # GPU로 이미지 이동
    img = img.to(device)

    # 예측
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)

    # 예측 결과 출력
    if predicted.item() == 0:
        print(f"{image_file}: 고양이")
    else:
        print(f"{image_file}: 개")
