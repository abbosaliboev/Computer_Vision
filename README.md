# Computer Vision
2026-1 semester — 충북대학교 소프트웨어학과

**이름:** ALIBOEV ABBOS  
**학번:** 2023041080  
**교수님:** 최경주 교수님

---

## 디렉토리 구조

```
Computer_Vision/
├── 1_week.ipynb                           # 1주차 실습
├── 2_week.ipynb                           # 2주차 실습
├── 8_Transformers.ipynb                   # Transformer 실습
└── HW-3/                                  # HW#3 과제
    ├── hw3_mlp_experiments.ipynb          # 실험 코드 (실험 A, B, C)
    ├── hw3_report.pdf                     # 보고서 (PDF)
    ├── expA_learning_curves.png           # 실험A: 학습 곡선
    ├── expA_gradient_flow.png             # 실험A: Gradient 흐름
    ├── expB_learning_curves.png           # 실험B: 학습 곡선
    ├── expB_dead_relu_heatmap.png         # 실험B: Dead ReLU 히트맵
    ├── expB_activation_hist.png           # 실험B: 활성화 분포 히스토그램
    ├── expB_activation_time_evolution.png # 실험B: 활성화 시간 변화
    ├── expB_gradient_flow.png             # 실험B: Gradient 흐름
    ├── expC_fmnist_optimizer_curves.png   # 실험C: Fashion-MNIST 학습 곡선
    ├── expC_digits_optimizer_curves.png   # 실험C: Digits 학습 곡선
    ├── expC_fmnist_best_comparison.png    # 실험C: 최적 성능 비교 (FMNIST)
    ├── expC_digits_best_comparison.png    # 실험C: 최적 성능 비교 (Digits)
    └── expC_gradient_flow.png             # 실험C: Gradient 흐름
```

---

## HW#3 — MLP 신경망 실험

### 실험 A: 손실 함수 비교 (CrossEntropy vs MSE)
- 데이터셋: Fashion-MNIST (10,000 subset), Digits Dataset
- Optimizer: Adam (lr=0.001), Epochs: 25
- 결과: CrossEntropy가 MSE보다 빠른 수렴과 높은 정확도 달성

### 실험 B: 활성화 함수 비교 (ReLU vs LeakyReLU vs Sigmoid)
- 데이터셋: make_moons + make_circles
- 가중치 초기화: std=0.01 (Dead ReLU 유도)
- Optimizer: Adam (lr=0.01), Epochs: 200
- 결과: LeakyReLU가 Dead ReLU 문제를 완화하며 최고 성능 달성

### 실험 C: 최적화 알고리즘 비교 (SGD vs SGD+Momentum vs Adam)
- 데이터셋: Fashion-MNIST, Digits Dataset
- LR Scheduler: ExponentialLR (gamma=0.9)
- Epochs: 25
- 결과: Adam이 가장 빠르고 안정적인 수렴 달성
