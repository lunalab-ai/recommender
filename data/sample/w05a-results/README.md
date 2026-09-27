# W05A 실측 참조 결과

MovieLens100K 원본 평점을 재배포하지 않습니다. 아래 파일은 수업 코드로 직접 계산한 집계입니다. 데이터 준비는 공식 GroupLens 다운로드를 사용하는 실습 B에서 수행합니다.

| 파일 | 읽는 목적 |
|---|---|
| protocol.json | 실제 실행 시각, 분할 seed, 코드 해시, 관측 집합 식별자, 순위 표본 |
| validation-quick.csv | 같은 100명 순위 표본에서 38개 설정 탐색; 평점 오차는 20,000건 전체 |
| validation-full.csv | 12개 기본 설정을 전체 validation 사용자에서 비교 |
| selected-configurations.json | test를 보기 전에 모델군별 NDCG/RMSE 목적 선택을 고정한 기록 |
| test-ranking.csv | 순위 목적에서 고른 설정의 최종 test 성과 |
| test-rating.csv | 평점 목적에서 고른 설정의 최종 test 성과 |
| test-final.csv | 위 두 목적의 선택을 합친 17개 고유 설정의 최종 결과 |
| validation-activity.csv | 학습 평가 수 중앙값을 기준으로 나눈 두 집단의 평균 |
| validation-distribution.csv | 개인 NDCG의 평균·중앙값·범위 |
| validation-n-sensitivity.csv | 인기 모델과 같은 100명에서 N=5/10/20의 별도 비교 |

train/validation/test는 60,000/20,000/20,000건이며, test 직전에는 train+validation 80,000건으로 재학습합니다. 무작위 관측 분할 한 번의 결과입니다. 시간순 미래 추천·신규 사용자·온라인 만족도를 측정하지 않습니다.

추천 후보는 1,682편 중 해당 학습 자료에서 사용자가 평가한 영화를 제외한 전체입니다. 관련 영화는 heldout 평점 4 이상입니다. 후보 점수 내림차순, 동점 영화 ID 오름차순입니다. 순위 지표는 관련 영화가 있는 사용자마다 계산한 뒤 동일 비중 평균을 사용합니다. 제외 사용자 수를 결과 열에 남깁니다. 미관측 영화를 실제 비선호라고 단정하지 않습니다.

인기 개수와 내용 코사인에는 평점 MAE/RMSE가 없어 CSV에서 비어 있습니다. 실패를 뜻하지 않습니다. 평균 대체 예측도 전체 평점 오차의 분모에 포함합니다. `rating_fallback`과 `cf_support`, `catalog_coverage`는 서로 다른 질문이므로 함께 읽습니다. 실행 시간은 장비와 캐시에 따라 달라집니다.

38개 탐색 설정과 12개 기본 설정 및 17개 최종 설정은 같은 개수를 뜻하지 않습니다. 전체 표와 100명 표본의 수치를 바로 빼서 개선량으로 보고하지 마세요. test 결과를 보고 설정을 다시 선택하지 마세요.

[실습 B](https://colab.research.google.com/github/lunalab-ai/recommender/blob/2026-fall-w05a-v2/notebooks/student/w05a-model-comparison.ipynb) · [공개 구현](https://github.com/lunalab-ai/recommender/blob/2026-fall-w05a-v2/src/luna_recsys/model_comparison.py) · [GroupLens](https://grouplens.org/datasets/movielens/100k/)


2026-09-28 호환성 수정: 관측쌍 식별자는 UTF-8 CSV의 CRLF 줄바꿈으로 고정하여 Windows/Linux에서 동일하게 계산합니다. 기존 Windows 실측 해시와 분할·성과 수치는 그대로입니다. protocol.json의 source_sha256는 원실험 당시 코드 해시이며, 호환성 수정 내역을 별도로 기록했습니다.
