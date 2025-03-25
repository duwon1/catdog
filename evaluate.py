import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.cnn_model import CatDogCNN

# 데이터 변환
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 검증 데이터셋 로딩
val_data = datasets.ImageFolder('data/val', transform=transform)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# 모델 불러오기
model = CatDogCNN()
model.load_state_dict(torch.load('cat_dog_model.pth'))
model.eval()

# 평가
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Final Accuracy: {accuracy:.2f}%')
