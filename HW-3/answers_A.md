# 실험 A 질문 답변

---

## Q1. 왜 MSE는 CrossEntropy보다 학습이 느리고, 안정적이지 못한가?

MSE Loss를 분류 문제에 사용할 때 핵심 문제는 **Gradient Scaling**이다.

MSE의 Gradient를 계산하면 다음과 같다:

  dL/dz_i = (y_hat_i - y_i) * y_hat_i * (1 - y_hat_i)

여기서 y_hat_i = softmax(z_i)이고, 뒤에 붙는 y_hat_i * (1 - y_hat_i) 항이 문제의 핵심이다.

**Gradient Scaling 문제:**

- 학습 초기에 모델이 잘못된 예측을 할 때 logit z_i의 절댓값이 크면, softmax 출력 y_hat_i가 0 또는 1에 가까워진다.
- 이 경우 y_hat_i * (1 - y_hat_i) ≈ 0 이 되어 Gradient가 매우 작아진다.
- 즉, **가장 틀렸을 때 오히려 Gradient가 작아져** 수정이 느려지는 역설적 상황이 발생한다.

**Logit 분포 변화 문제:**

- MSE는 각 출력 뉴런을 독립적으로 최소화하려 한다.
- 정답 클래스의 logit을 키우는 것보다 **오답 클래스의 출력값을 0에 가깝게 만드는 방향**으로도 Loss를 줄일 수 있어, 클래스 간 상대적 margin이 작아도 Loss가 낮아진다.
- 결과적으로 logit 분포가 결정 경계 근방에 몰려 학습 진동이 잦아진다.

---

## Q2. CrossEntropy가 MSE보다 수렴 속도가 빠른 이유는 무엇인가?

CrossEntropy Loss의 Gradient는 다음과 같다:

  dL/dz_i = y_hat_i - y_i

**Gradient Vanishing 완화 구조:**

- Softmax의 미분 항이 분모에서 약분되어 사라진다.
- 따라서 MSE처럼 y_hat_i * (1 - y_hat_i) 같은 포화 항이 없다.
- 모델이 정답 클래스를 0.01로 예측했을 때: Gradient = 0.01 - 1 = -0.99 → 매우 큰 수정 신호
- 모델이 정답 클래스를 0.99로 예측했을 때: Gradient = 0.99 - 1 = -0.01 → 수렴 신호로 작아짐

**정보이론적 특성:**

- CrossEntropy는 L = -log(y_hat_correct) 형태이므로, 예측이 틀릴수록 Loss가 **로그적으로 급증**한다.
- 이는 가장 잘못된 예측에 대해 가장 큰 Gradient를 자동으로 부여하는 구조다.
- 반면 MSE는 제곱 오차로 증가하여 같은 상황에서 Gradient가 상대적으로 작다.

**요약 비교표:**

| 상황                        | MSE Gradient       | CrossEntropy Gradient |
|-----------------------------|--------------------|-----------------------|
| 예측 매우 틀림 (y_hat≈0, y=1) | ≈ 0 (포화, 수정 안됨) | ≈ -1 (크고 안정)       |
| 예측 거의 맞음 (y_hat≈1, y=1) | ≈ 0               | ≈ 0 (수렴 신호)        |

CrossEntropy는 구조적으로 Gradient Vanishing이 발생하지 않아 분류 문제에서 일관되게 빠른 수렴을 보장한다.
