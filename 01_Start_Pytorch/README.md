## Pytorch 설치


### 파이썬 및 필수 라이브러리 설치하기

#### 파이썬 다운로드 위치
* https://www.python.org/downloads/

#### pytorch 설치방법
```
# Python 3.x
C:\> pip3 install torch torchvision torchaudio
```

```
# Python 2.x
C:\> pip install torch torchvision torchaudio
```
(*)torchvision과 torchaudio는 영상과 음성을 위한 추가 패키지이다.

#### pytorch 설치확인
```
from __future__ import print_function
import torch
x = torch.rand(5, 3)
print(x)
```

위와 같이 입력하고 실행하면 아래와 같은 output이 출력되면 제대로 설치된 것인다.
(*) rand함수가 난수를 생성하기 때문에 실제 수치는 다를 수 있다.

```
tensor([[0.0816, 0.8901, 0.6653],
        [0.7125, 0.0324, 0.2095],
        [0.5819, 0.5938, 0.4170],
        [0.5810, 0.1716, 0.8959],
        [0.7225, 0.2252, 0.3958]])
```

anaconda 또는 소스를 직접 빌드해서 설치하고 싶다면 pytorch 공식 홈페이지에서 확인하세요.
```
    https://pytorch.org/get-started/locally/
```