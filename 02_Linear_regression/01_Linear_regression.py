# 딥러닝에서 가장 기초적인 선형 회귀 모델 만들기

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 네트웍 정의
        self.linear = nn.Linear(1, 1) # Wx+b nn.Linear(input, output)

    def forward(self, x):
        # 네트웍을 연결해서 전체 네특웍 형태를 정의
        y_pred = self.linear(x)
        return y_pred


# input/output 정의
x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[1.0], [2.0], [3.0]])

# 실제 뉴럴넷 실행
net = Net()

# loss 계산 방식 정의
criterion = nn.MSELoss()

# 최적화 계산 방식 정의
optimizer = optim.SGD(net.parameters(), lr=0.1)

# 실제 트레이닝
for step in range(100):
    y_pred = net(x_data)
    loss = criterion(y_pred, y_data)

    if(step%10==0):
        print(step, loss.data)
    
    # gradient descent 직전에 초기화 해주기.
    # gradient가 축적되기 때문에 해 주어야 함.
    optimizer.zero_grad()

    # 모델 파라메터별 gradient 계산
    loss.backward()

    # 위에서 계산된 gradient에 의해 정의한 모델 파라메터 업데이트
    optimizer.step()
