# W03A · 공통 코드 사용 안내

공개 수업 버전 `2026-fall-w03a` 기준. 공개 전에는 로컬 저작 패키지로 검증한다.

기본 흐름: `model = UserCF().fit(train)` → `model.predict(pairs)` → `model.recommend(user_id, movies, 5)`.

학습 관측은 user_id/movie_id/rating 열이며 평점은 1–5, ID 쌍은 유일해야 한다. predict에는 평점 정답이 필요 없다.

## [toy_ratings](https://github.com/lunalab-ai/recommender/blob/2026-fall-w03a/src/luna_recsys/collaborative.py#L15)

직접 설계한 5명×5편의 관측 18행을 새 DataFrame으로 반환한다.

인자 없음. 열은 user_id(int), movie_id(int), rating(float,1–5).
없는 행은 미관측이며 실제 0점이 아니다. MovieLens의 부분집합이 아니다.
사용 예: toy_ratings().pivot(index='user_id', columns='movie_id', values='rating').
파일/네트워크 접근이나 기존 객체 변경은 없다.

## [binary_jaccard](https://github.com/lunalab-ai/recommender/blob/2026-fall-w03a/src/luna_recsys/collaborative.py#L31)

두 행동 집합 a,b의 교집합/합집합 유사도를 float로 반환한다.

set 또는 frozenset만 허용한다. 두 집합이 비면 근거 없음인 np.nan.
예: binary_jaccard({1,2},{2,3}) == 1/3. 집합을 수정하지 않는다.
scipy.spatial.distance.jaccard의 거리 반환과 구별한다.

## [UserCF](https://github.com/lunalab-ai/recommender/blob/2026-fall-w03a/src/luna_recsys/collaborative.py#L44)

모든 유효 평가자의 양의 유사도를 사용하는 원평점 가중평균 CF.

metric='cosine'(기본) 또는 'pearson'; min_common=3은 Pearson에만 적용한다.
생성자는 설정만 저장한다. fit은 train의 평점/관측 마스크/유사도/평균과
빠른 조회용 예측 배열을 준비한다. SGD나 별도 손실 최적화는 없다.
학습 후 rating_matrix_, similarity_, common_counts_, seen_, global_mean_,
movie_means_를 확인할 수 있다. 원본 입력 표는 수정하지 않는다.

예: model = UserCF().fit(toy_ratings())
    model.predict(pd.DataFrame({'user_id':[1], 'movie_id':[4]}))
Pearson의 음수·정의 불가 유사도는 예측 가중치로 쓰지 않는다.
근거 부족은 영화 평균→전체 평균으로 대체하며 basis에 남긴다.

## [UserCF.fit](https://github.com/lunalab-ai/recommender/blob/2026-fall-w03a/src/luna_recsys/collaborative.py#L71)

train 관측 표로 학습 상태를 만들고 self를 반환한다.

ratings: user_id/movie_id/rating 필수, 중복 쌍 없음, 결측 없음,
rating은 유한한 1–5점. 사용자/영화 ID는 정수다. test를 넣지 않는다.
출력 행렬의 사용자·영화 순서는 ID 오름차순이다. 유사도는 U×U,
평점은 U×I이며 원래 NaN을 보존한다. 예상 메모리는 O(U²+UI)라서
MovieLens100K 규모의 교육용이며 대규모 서비스용 구현은 아니다.
상태는 모든 계산 성공 후 갱신한다. 유효하지 않은 입력은 ValueError.

## [UserCF.predict_details](https://github.com/lunalab-ai/recommender/blob/2026-fall-w03a/src/luna_recsys/collaborative.py#L133)

사용자/영화 쌍의 예측과 근거를 입력과 같은 행 순서·인덱스로 반환한다.

pairs: user_id/movie_id 필수 DataFrame; rating 등 추가 열은 무시한다.
출력 열: prediction(1–5점), basis('cf','movie-mean','global-mean'),
n_contributors(양의 가중치 평가자 수), weight_sum(유사도 합).
빈 입력도 허용한다. 미등록 사용자→영화 평균, 미등록 영화→전체 평균.
fit 전/필수 열 부재/결측 ID는 ValueError. 학습 상태나 입력은 변경하지 않는다.

## [UserCF.predict](https://github.com/lunalab-ai/recommender/blob/2026-fall-w03a/src/luna_recsys/collaborative.py#L164)

predict_details와 같은 쌍 입력에서 예측 평점만 (N,) float 배열로 반환한다.

출력은 입력 행 순서이며 rating 열은 읽지 않는다. 상태 변경 없음.
사용 예: model.predict(test[['user_id','movie_id']]).

## [UserCF.explain](https://github.com/lunalab-ai/recommender/blob/2026-fall-w03a/src/luna_recsys/collaborative.py#L172)

한 예측에 기여한 평가자의 가중치·평점·기여도를 반환한다.

user_id/movie_id는 정수. 출력은 유사도 내림차순·user_id 오름차순의
user_id, common_items, similarity, rating, weight, contribution 표.
weight는 해당 영화의 유사도 합으로 정규화하며 contribution=weight*rating.
모든 행의 contribution 합이 CF 예측값이다. 대체 예측이면 빈 표다.
전체 계산 근거를 반환하며, 화면에서 앞부분만 보여줄 때 생략을 표시해야 한다.
원본/학습 상태는 변경하지 않고 fit 전에는 ValueError.

## [UserCF.recommend](https://github.com/lunalab-ai/recommender/blob/2026-fall-w03a/src/luna_recsys/collaborative.py#L198)

학습 이력을 제외한 카탈로그에서 예측 평점이 높은 영화를 반환한다.

user_id: 정수. movies: 유일하고 결측 없는 movie_id와 title의 DataFrame.
top_n: 1 이상 정수, 기본10. 출력은 최대top_n행, movie_id/title과
predict_details의 4열. 예측 내림차순, 동점 movie_id 오름차순.
미등록 사용자도 평균 대체로 반환하며 basis로 표시한다. 상태/입력 변경 없음.

## [cf_view](https://github.com/lunalab-ai/recommender/blob/2026-fall-w03a/src/luna_recsys/cf_app.py#L11)

CF 화면의 요약 HTML, 추천 표, 첫 추천의 근거 표를 반환한다.

model은 학습된 UserCF, movies는 movie_id/title 카탈로그, user_id는 정수,
top_n은 양의 정수(기본10). model.recommend와 explain을 호출하며 재학습하지 않는다.
추천 표는 최대top_n행, 근거 표는 전체 기여자 중 유사도 상위20행이다.
요약에는 전체/표시 기여자 수와 전체 기여도의 합을 명시한다. HTML의
영화명은 escape 처리한다. 이 함수는 서버/다운로드 없이도 직접 검사할 수 있다.
예: summary, rows, evidence = cf_view(model, movies, 1, 5).

## [build_cf_app](https://github.com/lunalab-ai/recommender/blob/2026-fall-w03a/src/luna_recsys/cf_app.py#L39)

학습된 모델을 입력받아 실행 전 Gradio Blocks 객체를 반환한다.

model: 학습된 UserCF. movies: movie_id/title 카탈로그. comparison: 기본None;
기존 학습된 FourMethodRecommender를 넘기면 이전 네 방법 탭도 유지한다.
사용자 선택과 추천 수(1–20)를 cf_view에 연결한다. 반환 앱은 호출자가
launch(share=True)로 Colab에서 실행한다. 이 함수는 서버를 띄우거나
데이터를 내려받지 않는다. Gradio [apps] 의존성이 필요하다.
