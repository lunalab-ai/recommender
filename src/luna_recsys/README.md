# 누적 추천 코드 안내

이 패키지는 강의마다 재사용할 기능을 쌓는 공간이다. notebook은 설명·실험·문제에 집중하고 같은 공통 구현을 복사하지 않는다.

| 모듈 | API | 역할 |
|---|---|---|
| `data` | `load_movielens_100k`, `download_movielens_100k` | 세 파일 로딩·검사·캐시 다운로드 |
| `datasets` | `prepare_movielens`, `PreparedMovieLens` | 한 번의 호출로 데이터 준비, 실제 모드 표시 |
| `datasets` | `synthetic_movielens` | 설치된 패키지에서도 가능한 독자적 CC0 대안 |
| `baselines` | `popularity_ranking`, `mean_rating_recommendations` | W01A/W01B API 유지 |
| `baselines` | `baseline_recommendations` | 평점 수·평균·집단별 평균 목록 |
| `rating_models` | `MeanRatingPredictor.fit/predict/predict_details` | 학습 통계와 예측 대체 수준 |
| `evaluation` | `split_ratings`, `evaluate_means` | 공통 분할과 RMSE 비교 |
| `demo_app` | `build_movie_recommender_app`, `build_baseline_lab` | 기존 앱과 W02A 반응형 앱 |

```python
from luna_recsys import prepare_movielens, split_ratings, MeanRatingPredictor

prepared = prepare_movielens()
split = split_ratings(prepared.data.ratings, seed=42)
model = MeanRatingPredictor("group", group_col="occupation")
model.fit(split.train, prepared.data.users)
estimates = model.predict(split.test[["user_id", "movie_id"]])
```

상태를 학습하는 객체에는 `fit/predict` 인터페이스를 사용한다. 단순 변환·정렬에는 함수를 사용한다. 새 API에는 타입 힌트·docstring·예제와 작은 합성 데이터 테스트를 제공한다. 기존 함수 인자와 공개 태그를 보존하고 새 기능을 확장한다.

`prepare_movielens(local_dir=...)`는 이미 가진 데이터 사본으로 검증할 때 사용한다. W02A 학생 Colab은 `mode="real"`로 캐시·공식 서버·해시가 일치하는 고정 HTTPS 대체 경로를 사용한다. 모두 실패하면 오류를 내며 합성 데이터로 바꾸지 않는다. 기존 API의 `auto` 및 명시적 `synthetic` 모드는 유지한다. 명시적으로 지정한 로컬 사본이 잘못되면 오류를 내므로 경로를 확인한다. 합성 모드의 숫자를 MovieLens 결과로 기록하지 않는다.

그룹 목록 전체가 비면 전체 평균 목록으로 전환하는 것과, 개별 평점 예측에서 그룹→영화→전체 평균으로 대체하는 것은 서로 다른 동작이다. 목록의 `basis`, 예측의 `level`로 확인한다.

W02A에서 추가한 기능은 데이터 준비 통합, 집단별 추천과 상태 있는 평균 예측기, holdout 평가, 모바일 카드다. 내용 기반·협업 필터링 등은 이후 수업에서 추가할 예정이며 현재 구현되었다고 표시하지 않는다.
