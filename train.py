import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.cnn_model import CatDogCNN

# GPU 사용 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터셋 로딩
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_data = datasets.ImageFolder('data/train', transform=transform)
val_data = datasets.ImageFolder('data/val', transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# 모델, 손실 함수, 옵티마이저 설정
model = CatDogCNN().to(device)  # 모델을 GPU로 이동
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 훈련 루프
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_batches = len(train_loader)  # 전체 배치 수

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)  # 데이터도 GPU로 이동
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 진행 상황 출력
        if batch_idx % 10 == 0:  # 매 10번 배치마다 출력
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{total_batches}], Loss: {loss.item():.4f}')

    # 에포크 끝나면 검증 진행
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # 데이터도 GPU로 이동
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Validation Accuracy: {accuracy:.2f}%')

# 학습이 끝난 후 모델 저장
torch.save(model.state_dict(), 'cat_dog_model.pth')
print("모델이 'cat_dog_model.pth'로 저장되었습니다.")
