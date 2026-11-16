# W12A · Colab 실습 길잡이

**딥러닝응용I(추천시스템) 01 · 2026-11-17**

## A · 영화제 편성

[실습 A 열기](https://colab.research.google.com/github/lunalab-ai/recommender/blob/2026-fall-w12a/notebooks/student/w12a-festival-llm.ipynb)

처음에는 팀당 한 대에서 Colab 기본 CPU로 실행합니다. 생성 모델 준비 중에는 10편의 가상 카드에서 후보를 손으로 고릅니다. 실제 생성이 준비되면 기본 CF, 기본 CB, 사전 필터 CB의 원응답과 검사 열을 차례대로 비교합니다.

| 본문 함수 | 입력 | 출력에서 볼 것 |
| --- | --- | --- |
| build_ballot_prompt | 영화 카드, 관객 조건, CF 또는 CB | 후보 ID, 근거 사전과 실제 요청 |
| eligibility | 영화 한 편과 관객 조건 | 시간·장르·이미 본 영화 위반 |
| validate_ballot | 원응답, 카드, 후보, 근거 사전 | 행별 통과 여부와 이유 |
| assemble_features | 사용자·영화와 훈련 벡터 | 원핫 30칸, 필요하면 벡터 16칸 |

처음 준비가 길다면 `MODE = "replay"`를 직접 선택하세요. 준비된 실제 호출 기록으로 검수와 판정을 실습할 수 있습니다. 새 모델 응답이 필요한 실험은 `MODE = "live"`에서 실행합니다. 9.3 셀에서 임베딩 모델을 따로 준비하며 라이브 실습에서는 실제 검색문 두 개를 바꿔 봅니다.

## B · 하이브리드 점수와 앱

[실습 B 열기](https://colab.research.google.com/github/lunalab-ai/recommender/blob/2026-fall-w12a/notebooks/student/w12a-hybrid-recommendation.ipynb)

| 본문 함수 | 입력 | 출력에서 볼 것 |
| --- | --- | --- |
| align_predictions | 실제 평점·CF 예측·MF 예측의 키 표 | 같은 사용자·영화의 세 열 |
| blend_scores | 맞춘 표와 MF 비중 하나 | 1–5점 척도의 결합 예측 |
| select_weight | 검증 표와 후보 비중 | 선택된 비중과 검증 RMSE 곡선 |

수정 후 **함수 정의 셀 → 후속 표/검증 셀 → 앱 조립 셀** 순으로 다시 실행합니다. 시험 셀은 비중을 기록한 다음 확인합니다. 앱은 Colab 셀 안의 조작 위젯을 제공하고 현재 Top-3를 CSV로 저장합니다. 같은 계산을 Gradio 앱의 콜백에서도 사용합니다.

**10-1**은 동일 모델과 데이터를 쓰는 기본 실습 뒤에 실행할 수 있습니다. MLP는 사용자·영화 임베딩 및 작은 은닉층을 실제로 학습합니다. 새 비중은 검증에서 고릅니다. 처음 패키지 준비에는 시간이 걸릴 수 있습니다.

### 자료 출처

가상 영화·관객·리뷰는 수업용입니다. 실습 B의 원평점은 실행 시 GroupLens에서 받습니다. [MovieLens 100K와 출처 조건](https://grouplens.org/datasets/movielens/100k/)
