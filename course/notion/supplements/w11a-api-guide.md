# W11A · 실습 코드를 본문에서 읽는 길잡이

**딥러닝응용I(추천시스템) · 동덕여자대학교 · 유원상 교수 · 2026-2**

공통 구현은 `luna_recsys`에 있고, 이번 네 노트북에는 핵심 정의를 실행 가능한 셀로 펼쳐 놓았습니다. 초기 셀은 모듈에서 추출한 같은 정의입니다. 직접 수정한 뒤 **정의 셀 → 후속 계산 셀 → 앱 조립 셀**을 다시 실행하세요.

## 7장 보강

| 함수 | 입력 → 반환 | 계산과 상태 |
|---|---|---|
| `fixed_ae_forward(raw, mean=3)` | 평점4개/NaN → 입력·관측·가중합·잠재값·복원 사전 | 주어진 가중치의 순전파, 학습 없음 |
| `observed_reconstruction_loss(y,p,m,loss_kind='mse')` | 같은 크기 텐서3개 → 손실 스칼라 | 실제 관측 수가 분모, 관측0 유지 |
| `autoencoder_train_step(model,opt,y,m,loss_fn=...)` | 모형·optimizer·배치 → 목적함수·기울기 크기 | 실제 모수와 optimizer 갱신 |
| `oriented_arrays(data,direction)` | 같은 데이터 + user/item → 값·마스크 사전 | item이면 둘 다 전치, 원본 불변 |
| `fit_autoencoder(...,step_fn=...,loss_fn=...)` | data·설정·본문 함수 → 학습 결과 | 전달한 실제 함수로 학습, 검증 RMSE 선택 |

7A는 본문의 `fixed_ae_forward`를 앱 계산 콜백에도 전달합니다. 7B는 본문의 손실·학습 함수를 `fit_autoencoder`에 전달합니다. 함수를 다시 정의하기만 하고 모듈 안의 다른 함수가 자동으로 바뀐다고 가정하지 마세요.

## 8장 계산과 전처리

| 함수 | 입력 → 반환 | 읽을 부분 |
|---|---|---|
| `scaled_attention(q,k,v,allowed)` | `(T,D),(S,D),(S,V)` → 점수·가중치·출력 사전 | 내적/√D → 허용 위치 softmax → V 가중합 |
| `pad_history(h,maxlen)` | 양의 내부 ID 이력 → `(maxlen,)` int32 | 최근 이력 절단·뒤0패딩 |
| `prefix_pairs(h,maxlen,min_prefix)` | 내부 ID 이력 → `(N,T)` 입력, `(N,)` 정답 | 정답 직전까지만 입력 |
| `prepare_sequence_data(prepared,pair_fn=...)` | 공식 데이터 객체·본문 함수 → `SequenceData` | 사용자 분리 후 본문 접두 함수 호출 |

`SequenceData.x/y`는 훈련·검증 배열입니다. `histories/targets`는 시험 사용자의 입력 이력과 끝10개 보류 정답입니다. `encode/decode`는 원본↔내부 ID, `popularity`는 훈련 사용자만 센 기준선입니다. `protocol`에는 seed·표본 제한·동시각 처리·후보 규칙을 기록합니다. 원자료 다운로드는 기존 `prepare_movielens(mode='real')`이 담당하며 실패를 합성 결과로 숨기지 않습니다.

## 8장 모형·학습·평가

| 함수 | 입력 → 반환 | 실제로 바뀌는 것 |
|---|---|---|
| `build_sequence_model(vocabulary,...)` | 설정 → 분류 모형·첫 블록 attention probe | 임베딩·위치·블록·분류 모수 초기화 |
| `sequence_train_step(model,opt,x,y,loss_fn=...)` | 입력`(B,T)`, 정답`(B,)` → 손실·기울기 크기 | 모수·optimizer를 한 번 갱신 |
| `fit_sequence(data,model_fn=...,step_fn=...)` | 데이터·본문 함수 → 모형·검증 이력·선택epoch | 실제 학습, 검증 CE 최소 가중치 복원 |
| `select_candidates(scores,h,k)` | 점수`(V,)`·알려진 이력 → 내부 ID 목록 | 0·비유한 점수·알려진 이력 제외, 입력 불변 |
| `rollout(score_fn,h,maxlen,n,candidate_fn)` | 예측 함수·이력 → 생성 ID n개 | 사본에 자체 예측만 추가 |
| `ranking_metrics(rec,target)` | 추천 목록·보류 정답 → hits/P/R/F1 | 집합 적중, 순서 정확도와 구별 |
| `evaluate_sequence(data,score_fn,...)` | 모형/기준선 → 사용자별 지표 표 | 각 사용자의 정답에만 비교 |

함수의 `maxlen/epochs/heads/key_dim`을 바꾸면 다른 실험 조건이 됩니다. 전체 데이터·분할·후보·학습량을 기록하고 한 조건씩 비교하세요. 같은 epoch가 같은 계산량을 뜻하지는 않습니다.

## 앱과 CSV에 내 함수 연결

`build_transformer_explainer(attention_fn=scaled_attention, data=data, fitted=fitted, candidate_fn=select_candidates)`로 앱을 조립합니다. 8A는 attention 함수만 연결해도 실행됩니다. 8B는 실제로 학습한 모형을 전달합니다.

`attention_view`와 `export_attention`은 같은 attention 함수를, `sequence_view`와 `export_sequence`는 같은 후보 함수를 사용합니다. CSV 파일 경로는 임시 파일이며 다운로드 버튼에 연결됩니다. 앱 공유 주소는 런타임이 켜진 동안만 유효합니다. 누적 앱의 FM·MF·MLP·AE 기능은 이전 구현을 재사용합니다.

실습이 중단되면 첫 오류를 읽고, 환경 설치가 갱신된 경우 런타임을 재시작한 뒤 처음부터 실행하세요. `runtime-check`는 기본 예제의 수치·모형 동작을 검사합니다. 학생의 빈 응답이나 자신의 설명을 대신 평가하지 않습니다.

## 구현과 참고

- [순차 추천 공통 구현](https://github.com/lunalab-ai/recommender/blob/2026-fall-w11a/src/luna_recsys/sequential_recommenders.py)
- [앱·다운로드 연결 구현](https://github.com/lunalab-ai/recommender/blob/2026-fall-w11a/src/luna_recsys/transformer_explainer.py)
- [Keras 직접 학습 루프](https://keras.io/guides/writing_a_custom_training_loop_in_tensorflow/)
- [Keras MultiHeadAttention](https://keras.io/api/layers/attention_layers/multi_head_attention/)

주교재8장 전체의 구현 흐름을 바탕으로 한 독자 보완 구현입니다. 교재 코드와 사용자 분리·ID·마스크·평가·정답 표현이 다른 부분은 강의노트에서 구분했습니다.
