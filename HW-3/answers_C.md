# 실험 C - Q1 답변

## SGD, SGD+Momentum, Adam 학습 곡선이 다른 이유

---

**SGD (확률적 경사 하강법)**

업데이트 규칙:
  새 파라미터 = 이전 파라미터 - 학습률 × Gradient

각 스텝이 완전히 독립적으로 수행되어 이전 방향을 기억하지 않는다.
Noisy Gradient로 인해 방향이 계속 바뀌므로 진동이 심하고 수렴이 느리다.

---

**SGD + Momentum**

업데이트 규칙:
  속도(v) = 모멘텀계수 × 이전속도 + 현재Gradient
  새 파라미터 = 이전 파라미터 - 학습률 × 속도(v)

과거 Gradient 방향을 누적하여 관성(momentum)을 부여한다.
같은 방향으로 계속 업데이트되면 속도가 붙고,
방향이 바뀌면 서로 상쇄되어 진동이 줄어든다.
덕분에 평탄한 구간(saddle point)을 빠르게 통과할 수 있다.

---

**Adam (Adaptive Moment Estimation)**

업데이트 규칙:
  1차 Moment (m): m = beta1 × 이전m + (1 - beta1) × Gradient
  2차 Moment (v): v = beta2 × 이전v + (1 - beta2) × Gradient²
  보정값:         m_hat = m / (1 - beta1^t),  v_hat = v / (1 - beta2^t)
  새 파라미터   = 이전 파라미터 - 학습률 × m_hat / (sqrt(v_hat) + epsilon)

- 1차 Moment: 과거 Gradient의 평균 방향 추적 (SGD+Momentum과 유사)
- 2차 Moment: 과거 Gradient의 분산 추적 → Gradient가 자주 큰 파라미터는 학습률을 자동으로 낮춤
- 결과적으로 파라미터마다 독립적인 학습률이 적용되어 가장 안정적으로 수렴

---

**세 방법 비교 요약:**

| Optimizer      | 과거 기억 | 적응형 학습률 | 수렴 속도 | 안정성 |
|----------------|-----------|--------------|-----------|--------|
| SGD            | 없음      | 없음         | 느림      | 낮음   |
| SGD+Momentum   | 방향만    | 없음         | 보통      | 보통   |
| Adam           | 방향+크기 | 있음         | 빠름      | 높음   |

세 방법이 동일한 네트워크에서 서로 다른 학습 패턴을 보이는 근본 원인은
각 방법이 활용하는 정보의 양이 다르기 때문이다.
SGD는 현재 Gradient만, Momentum은 방향의 역사를, Adam은 방향과 크기의 역사를 모두 활용한다.
