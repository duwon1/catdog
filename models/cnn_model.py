import torch.nn as nn
import torch.nn.functional as F

class CatDogCNN(nn.Module):
    def __init__(self):
        super(CatDogCNN, self).__init__()
        # 첫 번째 합성곱층 (3채널 -> 32채널)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        # 두 번째 합성곱층 (32채널 -> 64채널)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 풀링층 (2x2)
        self.pool = nn.MaxPool2d(2, 2)
        # 완전연결층
        self.fc1 = nn.Linear(64 * 56 * 56, 512)  # 입력 이미지가 224x224일 때 기준
        self.fc2 = nn.Linear(512, 2)  # 개/고양이 분류는 2개 클래스

    def forward(self, x):
        # 합성곱 -> 활성화 -> 풀링
        x = self.pool(F.relu(self.conv1(x)))  # (batch, 32, 112, 112)
        x = self.pool(F.relu(self.conv2(x)))  # (batch, 64, 56, 56)
        # 차원 축소
        x = x.view(-1, 64 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # 최종 출력층
        return x
