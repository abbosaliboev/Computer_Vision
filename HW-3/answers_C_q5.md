# 실험 C - Q5 답변

## Gradient 흐름이 특정 Optimizer에서 사라지거나 급변할 때 발생하는 문제

---

**SGD - Gradient Exploding 문제**

SGD에서 학습률이 너무 크면 파라미터 업데이트량이 과도하게 커진다.
  업데이트량 = 학습률 x Gradient

Gradient가 이미 큰 상황에서 학습률까지 크면 Loss가 최솟값을 넘어
반대 방향으로 튕겨나가는 Overshooting이 반복된다.
결국 Loss 값이 계속 증가하거나 NaN(계산 불능)이 된다.
해결 방법: 학습률을 줄이거나, Gradient Clipping 적용 (임계값 초과 시 Gradient를 잘라냄)

---

**SGD - Vanishing Gradient 문제**

SGD는 Gradient를 그대로 사용하기 때문에,
깊은 네트워크에서 역전파 시 Gradient가 레이어를 거칠수록 곱셈으로 줄어든다.
초기 레이어에 도달하면 Gradient가 거의 0이 되어 해당 레이어의 가중치가 거의 업데이트되지 않는다.
결과적으로 깊은 레이어일수록 학습이 제대로 이루어지지 않는다.

---

**Adam - Gradient Exploding 완화 구조**

Adam의 업데이트 규칙에서 분모 항은 다음과 같다:

  실제 업데이트량 = 학습률 x m_hat / (sqrt(v_hat) + epsilon)

여기서 epsilon = 1e-8 (0.00000001)이 핵심 안전장치 역할을 한다.
v_hat이 매우 작아져 sqrt(v_hat)이 거의 0이 되더라도
epsilon이 분모를 0이 되지 않도록 막아준다.
따라서 업데이트량이 무한히 커지는 상황을 방지한다.

또한 v_hat 자체가 과거 Gradient 제곱의 평균이므로,
Gradient가 자주 큰 파라미터는 분모가 커져 자동으로 업데이트량이 제한된다.

---

**세 Optimizer의 안정성 비교:**

| 문제 상황             | SGD            | SGD+Momentum   | Adam                    |
|-----------------------|----------------|----------------|-------------------------|
| Gradient Exploding    | 매우 취약      | 취약           | epsilon으로 완화        |
| Vanishing Gradient    | 매우 취약      | 부분 완화      | 적응형 학습률로 완화    |
| 학습률 민감도         | 높음           | 높음           | 낮음 (자동 조정)        |
