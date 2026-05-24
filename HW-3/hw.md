[HW#3] MLP 신경망 성능에 영향을 주는 요소별 실험과 분석과제

1. 목표  
손실 함수, 활성화 함수, 최적화 알고리즘이 학습 결과에 미치는 영향을 구현과 실험을 통해 정량적으로 분석
실험 결과를 바탕으로 성능 비교 + 실험결과 분석 보고서 작성
 

2. 실험 구성 
1) 기본 네트워크 구조 (예시) :  다음과 같은 MLP 구조 사용 - 예시 일 뿐 변경 가능함.

(더 깊은 Layer (512 → 256 → 128 → Output)나 Shallow Layer(128 → 64 → Output)도 추가 실험해 보면 최적화 기법이 학습 속도에 미치는 영향이 더 잘 보일 것임)

model = nn.Sequential(

    nn.Linear(input_size, 256),   # 입력 → 첫 번째 은닉층 (input_size → 256)
    nn.ReLU(),                              # 활성화 함수 (ReLU)
    nn.Linear(256, 128),              # 두 번째 은닉층 (256 → 128)
    nn.ReLU(),                              # 활성화 함수 (ReLU)
    nn.Linear(128, num_classes)   # 출력층 (128 → 클래스 개수)
)

2) 데이터셋 선택 : 각 실험에 적합한 데이터셋 사용

 실험              추천 데이터셋                                   설명
---------------------------------------------------------------------------------------------------------------------------------
실험A  Fashion-MNIST, Digits Dataset(scikit-learn)     28x28 흑백 이미지 (Fashion-MNIST), 8x8 숫자 이미지 (Digits)
실험 B make_moons, make_circlest(scikit-learn)       2D 비선형 분류 문제, Dead ReLU 및 학습 경계 시각화에 유리
실험 C Fashion-MNIST, Digits Dataset                        Optimizer 성능 차이를 분석하기에 적합

 

3)  활용 기술 제한 및 구현 조건 

A. 프레임워크 사용 범위
   - PyTorch 또는 TensorFlow 사용 가능
   - 모델은 nn.Module 또는 tf.keras.Model을 통해 구성할 수 있음
   - 학습 루프는 for epoch in range(...) 형태로 직접 작성할 것

B. 자동 미분 (autograd) 사용 가능
   - backward() (PyTorch) 또는 tape.gradient() (TensorFlow) 사용 가능
   - 단, 손실함수와 활성화함수는 명시적으로 설정하고 변경해야 함

예시) loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

C.  손실 함수, 활성화 함수, 최적화기법을 실험별로 수동 설정 및 교체 가능하도록 구현
   - loss_fn = ... 형식으로 명확히 설정
   - optimizer = torch.optim.Adam(...) 등 명확히 기술
   - 각 실험에서 이 구성 요소를 바꿔가며 성능 차이를 비교해야 함

[구현 조건]
  - 실험 A, B, C에 model.eval()과 with torch.no_grad() 활용
  - 모델 성능 차이가 잘 드러나는 학습률 / 에폭 수 사용

## Learning Rate  가이드 ##
  - SGD(0.01~0.1) : 단순 경사 하강법이므로 비교적 큰 학습률을 사용
  - SGD+Momentum(0.01~0.05) : SGD보다 빠르게 수렴하지만, 진동을 줄이기 위해 약간 낮춤
  - Adam(0.001~0.01) : Adaptive Learning Rate를 사용하므로 작은 값이 효과적

## 에폭수 가이드 ##
  - Fashion-MNIST 20 ~ 50
  - Digits Dataset 20 ~ 50
  - make_moons 200 ~ 500
  - CIFAR-10 50 ~ 100

## 에폭 수 설정 가이드 ##
  - 실험 전에 Baseline Epoch 설정 
  - 적절한 시점에 학습 종료
  - 수렴 속도가 느리면 Learning Rate Decay 추가 적용


3. 실험 설계 및 실행


[실험 A] 손실 함수 비교 : CrossEntropy Loss vs MSE Loss (with softmax)

1) 실험 목표
  -  MSE와 CrossEntropy가 학습 성능에 미치는 차이 분석
  - 학습 곡선의 수렴 속도, 정확도, loss 안정성 비교

2) 추천 데이터셋 :  Fashion-MNIST, Digits Dataset

3) 실험 조건
  - 동일한 네트워크 구조 및 optimizer 하에 손실함수만 변경
  - MSE Loss 사용 시 softmax를 출력층에 명시적으로 적용. CrossEntropy Loss는내부적으로 softmax 포함

예시) # MSE Loss 사용 시
   loss_fn = nn.MSELoss()
   outputs = torch.softmax(model(inputs), dim=1)

 4) 분석 포인트
  - MSE 사용 시 Gradient Vanishing 문제 분석
  - CrossEntropy의 빠른 수렴 속도와 안정성 확인
  - 최종 정확도 차이, 학습 곡선 비교



[실험 B] 활성화 함수 비교 :  ReLU vs LeakyReLU vs Sigmoid

1) 실험 목표
  - ReLU, LeakyReLU, Sigmoid가 학습에 미치는 영향을 분석
  - Dead ReLU 발생 유도 및 LeakyReLU의 완화 효과 확인

2) 추천 데이터셋 : make_moons, make_circles 

3) 실험 조건
  - 동일 네트워크 및 손실함수(CrossEntropy), Optimizer(Adam 또는 SGD)
  - weight 초기값을 std=0.01 등 작게 설정하여 dead neuron 상황 유도

4) 분석 포인트
  - Layer별 출력값 시각화 (matplotlib 사용)
  - Sigmoid는 vanishing gradient 발생 → 학습 정체 확인
  - Dead ReLU 발생 시 어떤 지표가 정체되는지 분석
  - Dead ReLU가 발생한 neuron 비율 측정 → 히트맵 시각화
  - LeakyReLU가 Dead ReLU 문제를 얼마나 완화하는지 시각적 분석



[실험 C] 최적화 알고리즘 비교 : SGD, SGD+Momentum, Adam

1) 실험 목표
  - SGD, SGD+Momentum, Adam의 성능 비교
  - 학습률 변화가 미치는 영향 분석 (0.1, 0.01, 0.001)

2) 추천 데이터셋: Fashion-MNIST, Digits Dataset 

3) 실험 조건
  - 동일한 네트워크, 손실함수, 데이터셋
  - 학습률에 따른 3개의 optimizer 성능 비교 (표로 정리)
  - Learning rate를 지수 감소 (Exponential Decay) 적용하여 실험  

     scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

4) 분석 포인트
  - Overshooting, 느린 수렴, 안정성 확인
  - 학습률 변화에 따른 Loss 곡선과 Gradient 흐름 분석



4. 실험 결과 분석 보고서 구조 :  아래의 5번, 6번 항목 참조
[각 실험 별로]

     1) 실험 목표
     2) 코드 및 실행 결과 ==> 코드 안에 집어넣을것. (따로 보고서에 넣지 않아도 됩니다. 주석처리 구체적으로)
     3) 그래프 및 시각화 결과 
     4) 정량적 분석 (표 포함)
     5) 해설 및 분석 (질문에 대한 답변) :  분석은 단순 “정확도가 더 높았다” 수준이 아니라, 원인/기전 중심 서술
     6) 결론 및 개선 사항



5. 보고서 작성 요구사항
1) 그래프 시각화
 - Loss vs Epoch,  Accuracy vs Epoch 그래프 
 - 활성화 함수 실험 시: 중간 레이어 출력 분포(히스토그램 또는 BoxPlot)

2) 히트맵 시각화
 - Dead ReLU Neuron의 비율 시각화 : 각 레이어의 뉴런 활성화 값을 히트맵 형태로 시각화. 
 - Gradient 흐름 시각화→ 특정 Layer에서 gradient가 소멸하는지 분석
    : Dead ReLU가 발생하면 역전파(backpropagation)에서 gradient가 0이 됨. 이를 시각화하면, 어느 Layer에서 gradient가 소멸하는지 명확히 볼 수 있음.

3) 정량 비교 표

예시)

실험 항목 최종 정확도 (%) Loss 최솟값 수렴까지 걸린 epoch 수
-------------------------------------------------------------------------------------------------------------------------------------------------------
CrossEntropy
MSE (with softmax)
-------------------------------------------------------------------------------------------------------------------------------------------------------

활성화 함수 Dead ReLU 비율 (%) 최종 정확도 (%) 수렴 속도
-------------------------------------------------------------------------------------------------------------------------------------------------------
ReLU
LeakyReLU
Sigmoid
-------------------------------------------------------------------------------------------------------------------------------------------------------

Optimizer 최종 정확도 (%) 수렴 속도 안정성
-------------------------------------------------------------------------------------------------------------------------------------------------------
SGD
SGD+Momentum
Adam
-------------------------------------------------------------------------------------------------------------------------------------------------------



6. 해설 및 분석 질문
[공통질문 (모든 실험에 공통 적용)] 

1) 손실 함수, 활성화 함수, 최적화 알고리즘이 학습 곡선에 미치는 영향은 무엇인가?
  - 수렴 속도, 진동 발생 여부, 학습 정체 구간을 그래프로 설명하세요.

2) Loss 감소와 Accuracy 상승 간의 불균형이 발생한 경우, 원인은 무엇이며 어떻게 개선할 수 있을까?
  - 특정 시점에서 Loss는 감소하지만 Accuracy가 오르지 않을 때 원인을 분석하세요.

3) Layer 별 Activation의 분포가 학습 중 어떻게 변화하는가?
  - 초반, 중반, 후반 학습 시 활성화 함수 (ReLU, Sigmoid) 값의 분포를 시각화하고 분석하세요.

4) Gradient Flow가 특정 층에서 소멸되거나 폭발할 때 모델에 미치는 영향은 무엇인가?
  - Gradient Vanishing/Exploding이 발생한 구간을 찾아 원인을 분석하세요.



[실험 A: 손실 함수 비교 (CrossEntropy vs MSE)]

1) 왜 MSE는 CrossEntropy보다 학습이 느리고, 안정적이지 못한가?
  - 로지트(logits) 값의 분포 변화와 gradient scaling 문제를 설명하세요.

2) CrossEntropy가 MSE보다 수렴 속도가 빠른 이유는 무엇인가?
  - CrossEntropy가 Gradient Vanishing 문제를 어떻게 완화하는지 분석하세요.

3) 학습 초기 vs 후반의 gradient 분포 차이
  - 학습 초기에는 gradient가 크지만 후반에는 0에 수렴하는지 시각화하세요.

4) MSE 사용 시 softmax를 적용하지 않으면 학습이 왜 어려운가?



[실험 B: 활성화 함수 비교 (ReLU vs LeakyReLU vs Sigmoid)]

1) Dead ReLU가 발생한 비율이 높으면 학습에 미치는 영향은?
  - 특정 Layer에서 Dead ReLU 비율이 높으면 학습이 정체되는 이유를 분석하세요.

2) LeakyReLU가 Dead ReLU를 얼마나 완화하는가?
   - LeakyReLU 사용 시 Dead 상태에 빠진 뉴런이 감소하는지 확인하세요.

3) Sigmoid를 사용할 때 vanishing gradient가 발생하는 구간은 어디인가?
   - Sigmoid의 gradient가 0에 가까워지는 지점을 시각화하고 설명하세요.

4) 각 Layer가 학습에 기여하는 정도를 히트맵으로 분석하세요.


[실험 C: 최적화 알고리즘 비교 (SGD, SGD+Momentum, Adam)]

1) SGD, SGD+Momentum, Adam의 학습 곡선이 다른 이유는 무엇인가?
 - SGD+Momentum이 왜 SGD보다 빠른 수렴을 보여주는지 설명하세요.

2) 학습률 변화가 학습 안정성에 미치는 영향은?
  - 너무 큰 학습률은 overshooting, 너무 작은 학습률은 정체를 초래함을 분석하세요.

3) Adam이 초기 학습에 빠르게 수렴하는 이유는 무엇인가?
  - 적응형 학습률(Adaptive Learning Rate) 덕분에 초기에 빠르게 최적점으로 수렴함을 설명하세요.

4) 지수 감소(Exponential Decay)를 사용했을 때 성능 변화는?
  - 학습 후반부에 과적합이 줄어들거나 수렴 안정성이 증가하는지 분석하세요.

5) Gradient 흐름이 특정 Optimizer에서 사라지거나 급격히 변할 때, 어떤 문제가 발생하는가?
  - Vanishing Gradients, Exploding Gradients의 원인 분석

6) Optimizer가 동일한 네트워크에서 서로 다른 학습 패턴을 보이는 이유는?

  - SGD, Adam, SGD+Momentum의 학습 곡선이 왜 다르게 움직이는지, 수식적 근거를 포함하여 설명하세요.



7. 제출항목 
1) 보고서 (PDF 형식)
2) 실험에 사용된 코드 (Jupyter Notebook 또는 .py 파일) : 주석처리 확실히 할것




<평가기준>
1) 실험 정확성 및 재현성 20% : 실험 조건에 맞춰 정확하게 수행하고, 동일 조건에서 재현 가능한지 확인
  - 동일한 조건에서 실험이 재현 가능한가?
  - 학습률, 에폭 수, Optimizer 설정을 명확히 기록했는가?

2) 학습곡선 분석 능력 20% : Loss vs Epoch, Accuracy vs Epoch 그래프를 통한 수렴 속도, 진동, 학습 정체 구간 해석
  - Loss와 Accuracy의 진동 구간을 해석했는가?
  - 학습이 빠르게 진행되는 시점과 정체되는 시점을 시각적으로 명확히 나타냈는가?

3) Activation 분포 분석 20% 각 Layer의 Activation 변화를 시각화하고, Dead ReLU, Vanishing Gradient의 발생 여부 분석
  - Dead ReLU가 발생한 Layer를 히트맵으로 정확히 시각화했는가? 
  - Sigmoid 사용 시 Vanishing Gradient가 발생한 구간을 설명했는가?
  - LeakyReLU가 Dead ReLU 문제를 완화하는 효과를 명확히 분석했는가?

4) Optimizer 비교 분석 20% SGD, Momentum, Adam의 학습 곡선과 수렴 패턴 차이를 해석하고, 그 이유를 설명
  - SGD, SGD+Momentum, Adam의 수렴 속도 차이를 시각화했는가?
  - Local Minima에 빠지지 않도록 하는 최적화 기법의 차이를 설명했는가?
  - Exponential Decay가 학습 안정성에 미친 영향을 분석했는가?

5) 문제 해결 접근법 20% : 문제 발생 시 (Gradient Vanishing, Overshooting 등) 개선 방법을 제안하고, 이를 실험으로 증명
  - 학습 중 진동이 심하거나 수렴이 어려울 때 어떻게 해결할 수 있을까?
  - Dead ReLU가 심각할 때 어떻게 해결할 수 있을까?

