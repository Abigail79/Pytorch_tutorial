
# 털과 날개가 있는지 없는지에 따라, 포유류인지 조류인지 분류하는 신경망 모델을 만들어봅니다.
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# 네트웍 정의
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 네트웍 정의
        self.linear = nn.Linear(2, 3) # Wx+b nn.Linear(input, output화
        nn.init.uniform_(self.linear.weight, -1, 1) # weight를 -1.0~1.0로 uniform normalize로 초기화
        nn.init.zeros_(self.linear.bias) # bias를 0으로 초기화

    def forward(self, x):
        # 네트웍을 연결해서 전체 네특웍 형태를 정의
        L = self.linear(x)
        L = F.relu(L)
        y_pred = F.softmax(L)
        return y_pred


# 데이터 입력
# [털, 날개]
x_data = torch.Tensor(
    [[0, 0], [1, 0], [1, 1], [0, 0], [0, 0], [0, 1]])

# [기타, 포유류, 조류]
# 다음과 같은 형식을 one-hot 형식의 데이터라고 합니다.
y_data = torch.Tensor([
    [1, 0, 0],  # 기타
    [0, 1, 0],  # 포유류
    [0, 0, 1],  # 조류
    [1, 0, 0],
    [1, 0, 0],
    [0, 0, 1]
])

_, y_label = y_data.max(dim=1)

# 실제 뉴럴넷 실행
#net = nn.Sequential(Net())
net = Net()

# loss 계산 방식 정의
criterion = nn.CrossEntropyLoss()

# 최적화 계산 방식 정의
optimizer = optim.SGD(net.parameters(), lr=0.01)

# 실제 트레이닝
for step in range(100):

    y_pred = net(x_data)
    
    # cross-entropy를 직접 계산하는 경우
    # loss = torch.mean(-torch.sum(y_data * torch.log(y_pred), dim=1))
    # torch에서 제공하는 모듈을 사용하는 경우
    loss = criterion(input=y_pred, target=y_label) 

    if (step+1)%10==0:
        print(step+1, loss.data)
    
    # gradient descent 직전에 초기화 해주기.
    # gradient가 축적되기 때문에 해 주어야 함.
    optimizer.zero_grad()

    # 모델 파라메터별 gradient 계산
    loss.backward()

    # 위에서 계산된 gradient에 의해 정의한 모델 파라메터 업데이트
    optimizer.step()

#########
# 결과 확인
# 0: 기타 1: 포유류, 2: 조류
######
# tf.argmax: 예측값과 실제값의 행렬에서 tf.argmax 를 이용해 가장 큰 값을 가져옵니다.
# 예) [[0 1 0] [1 0 0]] -> [1 0]
#    [[0.2 0.7 0.1] [0.9 0.1 0.]] -> [1 0]
pred = torch.argmax(y_pred, dim=1)
print('예측값:', pred)
print('실제값:', y_label)

is_correct = torch.eq(pred, y_label)
accuracy = torch.mean(is_correct.type(dtype=torch.float32))
print('정확도: %.2f' % (accuracy * 100))

