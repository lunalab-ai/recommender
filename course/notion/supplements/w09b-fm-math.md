# W09B · FM 수학과 코드 보강

> 딥러닝응용I(추천시스템) · 동덕여자대학교 · 유원상 교수 · 2026-2 · 2026-10-29

출처: 주교재 5.1–5.4; [Rendle(2010)](https://jame-zhang.github.io/assets/algo/Factorization-Machines-Rendle2010.pdf) III-A–C; [NumPy sum 공식 문서](https://numpy.org/doc/stable/reference/generated/numpy.sum.html). 항등식의 단계 전개, 수치 미분 설명과 함수 설계는 수업용 보충입니다.

본문의 효율식과 미분을 작은 단계로 다시 읽습니다. $n$은 입력 열 수, $k$는 잠재 차원 수이며 $V$는 $(n,k)$입니다. 모든 계산은 정답을 입력에 섞지 않은 한 관측에서 시작합니다.

## 1. 합의 순서를 바꾸기

내적을 먼저 펼치면 다음과 같습니다.

$$
\sum_{i<j}\left(\sum_f v_{if}v_{jf}\right)x_ix_j
=\sum_f\sum_{i<j}(v_{if}x_i)(v_{jf}x_j).
$$

한 $f$를 고정하고 $a_i=v_{if}x_i$라고 잠시 부릅니다.

$$
\left(\sum_i a_i\right)^2=\sum_i a_i^2+2\sum_{i<j}a_ia_j.
$$

양쪽에서 자기항을 빼고 2로 나눈 뒤, 모든 $f$의 값을 더하면 본문의 효율식입니다. 합성 입력에서도, 연속·음수 입력에서도 성립합니다. $x_i^2=x_i$는 0과 1일 때만 성립하므로 일반 FM 입력에는 사용할 수 없습니다.

## 2. 효율식을 잠재 계수 하나로 미분하기

한 차원의 기여를 $I_f=\frac12(S_f^2-D_f)$, $S_f=\sum_j v_{jf}x_j$, $D_f=\sum_j v_{jf}^2x_j^2$라 둡니다.

$$
\frac{\partial S_f}{\partial v_{if}}=x_i,\qquad
\frac{\partial D_f}{\partial v_{if}}=2v_{if}x_i^2.
$$

$$
\frac{\partial I_f}{\partial v_{if}}
=\frac12(2S_fx_i-2v_{if}x_i^2)
=x_i(S_f-v_{if}x_i).
$$

다른 잠재 차원은 이 $v_{if}$에 의존하지 않고, 절편과 선형항에도 $v_{if}$가 없으므로 이것이 전체 예측의 $v_{if}$ 미분입니다.

## 3. 손실 미분과 갱신 부호

$$
L=\frac12(y-\hat y)^2,\quad e=y-\hat y,\quad
\frac{\partial L}{\partial v_{if}}=-e\,x_i(S_f-v_{if}x_i).
$$

경사하강은 이전 모수에서 $\alpha$배 손실 미분을 뺍니다. 위 미분의 음수와 경사하강의 음수가 만나 본문 갱신의 $+\alpha e$가 됩니다. 예측 미분 $\partial\hat y/\partial v_{if}$만 빼는 코드는 정답의 방향을 반영하지 못합니다. $w_i$도 같은 원리로 $w_i+\alpha e x_i$입니다.

이번에는 절편 고정·정규화 0입니다. 실제 전체 학습에서는 절편 학습 여부, 정규화 형태, 관측 평균 손실의 계수 등 정의를 먼저 확인해야 합니다. 전부 0인 $V$로 시작하면 이 상호작용의 데이터 미분도 0이므로 잠재벡터가 움직이지 않을 수 있습니다. 이를 일반적인 좋은 초기화로 삼지 않습니다.

## 4. 식과 실제 함수 연결

계산 모듈은 [fm_core.py](https://github.com/lunalab-ai/recommender/blob/2026-fall-w09b/src/luna_recsys/fm_core.py)입니다. 다음 설명은 새로 만든 교육용 구현에 관한 것으로 출판사 코드를 복사하지 않았습니다.

| 함수 | 입력 → 출력 | 수식과 확인할 것 |
|---|---|---|
| `encode_observation` | 사용자·아이템·맥락 → $(5,)$ x | 원핫 블록과 별도 y |
| `toy_parameters` | 인자 없음 → 새 w,V,절편,이름 | 고정 합성 모수, 실제 장르 의미 없음 |
| `pair_contributions` | x,V → 쌍별 행 목록 | 활성 $i<j$, 기여가 0인 쌍도 표시 |
| `fm_score_naive` | x,w,V,절편 → 실수 | 쌍을 직접 열거한 독립 검산 |
| `fm_score` | x,w,V,절편 → 항별 dict | weighted_factors, S, D, 차원별 기여 |
| `to_sparse`, `from_sparse` | 밀집 ↔ index/value | Python 0기반, 비영 값 보존 |
| `fm_score_sparse` | index/value,w,V,절편 → 실수 | 비영 행으로 같은 효율식 계산 |
| `prediction_gradients` | x,V → w/V 예측 미분 | 손실 미분이 아닌 예측 미분 |
| `fm_sgd_step` | x,y,w,V,절편,학습률 → 새 상태와 전후 값 | 같은 이전 상태, 원본 배열 불변 |

실제 선언 줄로 이동하는 링크는 노트북의 API 확인 셀에서 고정 태그를 기준으로 생성합니다. 공개 태그가 배포되기 전에는 링크가 열리지 않을 수 있습니다. 계산 모듈은 다운로드·파일 저장·네트워크 호출이 없으며 모든 앱 비교는 고정 모수에서 다시 시작합니다.

```python
import numpy as np
from luna_recsys.fm_core import toy_parameters, encode_observation, fm_score
p = toy_parameters()
x = encode_observation('A', 'P', 1)
result = fm_score(x, p['w'], p['V'], p['bias'])
assert np.isclose(result['prediction'], 3.25)
print(result['factor_sums'])  # E05의 차원별 합과 비교
```

## 5. 수치 미분은 어떤 검산인가

모수 하나 $\theta$를 아주 조금 양쪽으로 움직여 손실 차이를 구합니다.

$$
\frac{\partial L}{\partial\theta}\approx
\frac{L(\theta+h)-L(\theta-h)}{2h}.
$$

이는 해석 미분을 독립적으로 확인하는 도구입니다. $h$를 무조건 작게 하면 반올림 오차가 커질 수 있습니다. 노트북에서는 고정 합성 사례와 $h=10^{-6}$에서 비교하고 오차 허용 범위를 명시합니다. 유한한 몇 사례의 통과를 모든 데이터에서의 성능 보장으로 해석하지 않습니다.
