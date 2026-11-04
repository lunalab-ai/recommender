# W10B · AE 실습 API 길잡이

> 딥러닝응용I(추천시스템) · 동덕여자대학교 · 유원상 교수 · 2026-2 · 2026-11-05

함수 이름을 외우기보다 **입력, 반환값의 모양, 상태 변경**을 확인하세요. 계산은 `luna_recsys.autoencoder_recommenders`, 화면은 `luna_recsys.autoencoder_explainer`에서 가져옵니다. 먼저 강의의 작은 숫자를 손으로 계산한 뒤 같은 함수의 중간값과 비교합니다.

## 1. 작은 계산과 실제 자료 준비

| 함수 | 입력과주요 옵션 | 반환값 | 상태 변경 |
|---|---|---|---|
| `toy_ratings()` | 없음 | 독자 4×4 평점배열, 빈칸NaN | 새사본, 학습·다운로드 없음 |
| `fixed_ae_forward(raw=None, mean=3)` | 길이4의1–5 평점/NaN, 주어진평균 | 입력·마스크·가중치·가중합·잠재값·중심화/원척도예측 사전 | 고정순전파, 학습 없음 |
| `masked_metrics(y, p, m)` | 같은shape의정답·예측·관측 마스크 | `sse, n, mse, rmse` 사전 | 없음 |
| `loss_comparison(y, p, m)` | 유한한중심화배열과마스크 | 관측/전체 칸/정답!=0의오차표 | 없음 |
| `prepare_movielens(mode='real')` | 공식다운로드/캐시, 선택한모드 | `PreparedMovieLens`와실제모드 | 캐시파일준비 가능 |
| `prepare_autoencoder_data(prepared, seed=20261105)` | 준비된자료, 선택적`max_rows` | 동일관측분할의`AutoencoderData` | 입력표변경·학습 없음 |

`masked_metrics`는 관측된 항의 제곱오차 합을 관측 수로 나눕니다. 예를 들어 오차 제곱의 합이 6이고 관측이 3개라면 MSE는 2, RMSE는 √2입니다. 예측과 정답의 모양이 다르거나 관측이 0개이면 오류를 냅니다. 관측 밖의 NaN은 먼저 제외합니다.

`mode='real'`은 공식 MovieLens 자료를 준비합니다. 실패를 합성 자료로 숨기지 않습니다. `mode='synthetic'`은 독자 합성 자료 80명·18편을 명시적으로 준비합니다. 실제 자료와 합성 자료의 결과를 구별해서 기록하세요.

## 2. 준비된 자료의 구조 읽기

```python
data.shape                 # (사용자 수, 영화 수)
data.mean                  # 훈련 관측만으로 계산한 평균
data.values['train']       # 모델 입력: 관측값-평균, 나머지는 0
data.masks['train']        # 중심화 전에 보존한 관측 여부
data.masks['validation']   # 모형 선택에 사용할 관측
data.masks['test']         # 마지막 평가에 사용할 관측
data.frames['train']       # 분할된 관측표와 row_id
data.warm                  # 사용자와 영화에 모두 훈련 관측이 있는 위치
```

`data.ratings`는 원척도 평가용 배열입니다. 이를 그대로 모델에 넣으면 보류 정답이 입력에 섞일 수 있습니다. 모델 입력은 계속 **`data.values['train']`**입니다. 내부 배열을 바꾸는 실험에는 먼저 `copy()`를 사용하세요.

사용자·영화 사전은 카탈로그 ID를 정렬한 뒤 0부터 시작하는 위치에 연결합니다. 카탈로그에 있어도 훈련 관측이 없을 수 있습니다. 이번 구현은 그런 미학습 축을 평균으로 예측하고 해당 평가 관측 수를 `cold_n`으로 보고합니다.

## 3. 모형 생성·학습·예측·평가

| 함수 | 하는 일과인수 | 반환값 |
|---|---|---|
| `oriented_arrays(data, direction)` | `user`는NxM, `item`은MxN;값·마스크를함께전치 | 방향에맞는배열사전두개 |
| `build_fixed_autoencoder()` | 설명용4→2→4, 주어진가중치 | Keras모형, 모수22개 |
| `build_autoencoder(M, hidden=(32,8), reg=.0003)` | M→32→8→32→M구조생성, 학습은안함 | Keras모형 |
| `fit_autoencoder(data, direction='user', epochs=8)` | Adamax실제 학습, 검증 RMSE로선택 | `AutoencoderFit` |
| `predict_autoencoder(fitted, data)` | 훈련입력으로예측, 평균 한 번복원 | 원척도NxM예측배열 |
| `evaluate_autoencoder(data, pred, split='test')` | 지정관측에서만평가 | SSE·n·MSE·RMSE·`cold_n` |
| `recommend_autoencoder(data, pred, user_id, k=5)` | 알려진관측이력제외후점수정렬 | 영화ID·제목·예측표 |

`fit_autoencoder`에서 추가로 바꿀 수 있는 설정은 `hidden, reg, learning_rate, batch_size, patience, loss_kind, seed`입니다. 기본 학습 목적은 **관측 MSE + L1**입니다. `loss_kind='rmse'`는 안정화를 위해 `sqrt(MSE+1e-8)+L1`을 사용합니다. 보고하는 RMSE에는 이 작은 상수를 더하지 않습니다.

```python
fitted.model          # 검증에서 가장 좋았던 가중치를 복원한 모형
fitted.history        # 각 epoch 끝의 전체 훈련·검증 관측 평가
fitted.best_epoch     # 검증 RMSE로 선택한 epoch
fitted.train_seconds  # 초기 import/모형 초기 호출을 제외한 학습·평가 시간
fitted.updates        # 실제 미니배치 optimizer 갱신 수
fitted.config         # 은닉폭, L1, 학습률, seed, 모수 수, 자료 모드
```

학습 함수는 시험 정답으로 모형을 선택하지 않습니다. 모형 생성·학습은 난수 상태와 가중치를 바꿉니다. 예측은 가중치를 갱신하지 않으며 결과를 1–5로 자르지도 않습니다. 학습률과 L1 계수는 우리가 정하는 설정이고, optimizer가 학습하는 것은 가중치와 편향입니다.

## 4. 앱에서 입력 → 함수 → 표 → CSV 연결

`ae_view(r1,r2,r3,r4,o1,o2,o3,o4)`는 평점 4개와 관측 체크 4개를 받습니다. 기본값이 있어 인수 없이도 호출할 수 있습니다. 훈련 평균 3과 설명용 가중치는 고정되어 있습니다. 반환값의 순서는 다음과 같습니다.

1. 계산 요약문
2. 평점표: `movie, rating, observed, centered_input, reconstructed, observed_squared_error`
3. 잠재값 표: `node, before_tanh, after_tanh`
4. 세 계산 대상의 오차표
5. Matplotlib 그림

`build_autoencoder_explainer(view_fn=...)`에 자신의 함수를 연결합니다. 그 함수는 같은 8개 인수를 받고 같은 5개 항목을 반환해야 합니다. 첫 화면 준비를 위해 인수 없이 호출해도 작동해야 합니다. 반환 평점표에 열을 추가하면 화면과 CSV에 같은 열이 표시됩니다.

`export_ae_csv(...,view_fn=...)`는 그 표를 임시 CSV로 저장하고 경로를 반환합니다. 다운로드 파일은 현재 런타임에서 생성됩니다. 워크북 L08과 실습 A에서 관측된 칸의 절대오차 열을 직접 추가합니다.

`train_synthetic_comparison(epochs=5)`는 독자 합성 자료에서 평균·사용자·아이템 모형을 실제로 학습해 비교합니다. 결과표와 검증 곡선을 반환합니다. 이것을 MovieLens 100K 실험으로 보고하지 않습니다. 앱 공유는 호출자가 `launch(...)`로 실행하며 공유 주소의 수명은 해당 런타임에 따릅니다.

## 5. 실행이 막힐 때 확인할 순서

- 이름을 찾지 못하면 앞의 정의·입력 준비 셀을 실행했는지 확인합니다.
- 설치 뒤 이전 라이브러리가 메모리에 남아 있으면 런타임을 재시작합니다.
- 자료 준비 실패와 모델 학습 실패를 구별하고 오류를 숨기지 않습니다.
- 관측 0개, 서로 다른 shape, 잘못된 모드 오류에서는 입력과 마스크를 확인합니다.
- 앱에서 입력을 바꾸고 계산 버튼을 누릅니다. 함수를 수정했다면 앱 생성·실행 셀도 다시 실행합니다.

고정 모수의 계산, 합성 자료 학습, 실제 자료 학습을 구별해서 기록하세요. 수학 설명과 교재 대응은 강의노트에서 확인합니다.
