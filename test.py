from torchvision import datasets

# 데이터셋 로드
train_data = datasets.ImageFolder('data/train')

# 클래스 인덱스 확인
print(train_data.class_to_idx)