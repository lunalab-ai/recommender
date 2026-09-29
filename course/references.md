# 전체 참고자료

[강의 허브](../README.md) · [Python 복습 길잡이](python-basics.md)

## 공통 보충 자료

수업을 위해 추가한 보충 읽기이며 외부 원문은 각 링크에서 확인합니다. 공통 보충 링크 확인일: 2026-09-07.

| 분야 | 자료 | 언어 | 활용 방법 |
|---|---|---|---|
| Python 기초 | [이 수업을 위한 Python 복습 길잡이](../course/python-basics.md) | 한국어 | Colab 실행부터 변수·반복문·함수·표 읽기까지, 짧은 예제로 복습 |
| Python 기초 | [Python 한국어 공식 자습서](https://docs.python.org/ko/3/tutorial/) | 한국어·일부 영어 | 문법을 배운 적이 있다면 3·4·5·8장으로 복습 |
| Python 기초 | [University of Helsinki — Python Programming MOOC 2026](https://programming-26.mooc.fi/) | 영어 | 프로그래밍이 처음이라면 Introduction의 Part 1부터 연습 |
| 실습 도구 | [Google Colab 시작 notebook](https://colab.research.google.com/notebooks/intro.ipynb) | 영어 중심 | 브라우저에서 notebook을 열고 셀 실행 연습 |
| 실습 도구 | [NumPy — Absolute basics for beginners](https://numpy.org/doc/stable/user/absolute_beginners.html) | 영어 | 배열·shape·인덱싱을 이해하고 평점 행렬 준비 |
| 실습 도구 | [pandas — Getting started tutorials](https://pandas.pydata.org/docs/getting_started/intro_tutorials/index.html) | 영어 | 표 읽기·행 선택·요약 통계를 MovieLens 실습과 연결 |
| 실습 도구 | [Matplotlib — Pyplot tutorial](https://matplotlib.org/stable/tutorials/pyplot.html) | 영어 | 평점 분포와 실험 결과를 그래프로 표현 |
| 추천시스템·머신러닝 | [Google — Recommendation systems](https://developers.google.com/machine-learning/recommendation) | 영어 | 추천 문제와 주요 추천 방법을 개괄 |
| 추천시스템·머신러닝 | [scikit-learn — Getting Started](https://scikit-learn.org/stable/getting_started.html) | 영어 | 학습·예측·전처리·평가 API 복습; 머신러닝 기초 이후 |
| 추천시스템·머신러닝 | [Keras — Collaborative Filtering for Movie Recommendations](https://keras.io/examples/structured_data/collaborative_filtering_movielens/) | 영어 | Embedding 기반 영화 추천 예제; 딥러닝 추천 단원에서 참고 |
| 이전 수업 버전 보관 | [w01a 원 수업 실습 버전](https://github.com/lunalab-ai/recommender/tree/2026-fall-w01a-r3/notebooks/student) | 한국어 | 기존 수업 링크는 보존합니다. 설명과 풀이를 보완한 현재 실습은 위 수업 표에서 엽니다. |
| 이전 수업 버전 보관 | [w01b 원 수업 실습 버전](https://github.com/lunalab-ai/recommender/tree/2026-fall-w01b/notebooks/student) | 한국어 | 기존 수업 링크는 보존합니다. 설명과 풀이를 보완한 현재 실습은 위 수업 표에서 엽니다. |
| 이전 수업 버전 보관 | [w02a 원 수업 실습 버전](https://github.com/lunalab-ai/recommender/tree/2026-fall-w02a/notebooks/student) | 한국어 | 기존 수업 링크는 보존합니다. 설명과 풀이를 보완한 현재 실습은 위 수업 표에서 엽니다. |
| W05A 복습과 실험 | [W05A 실습 A · 모델과 지표 복습](https://colab.research.google.com/github/lunalab-ai/recommender/blob/2026-fall-w05a-v2/notebooks/student/w05a-review-metrics.ipynb) | 한국어 | 작은 표의 계산부터 지표 해석과 미니 웹 앱까지 |
| W05A 복습과 실험 | [W05A 지표 보강 노트](../course/notion/supplements/w05a-metrics-remediation.md) | 한국어 | MAE/RMSE, Precision/Recall, NDCG를 그림과 계산으로 복습 |
| W05A 복습과 실험 | [W05A 실험 기록 양식](../course/notion/supplements/w05a-experiment-record.md) | 한국어 | 목적·고정 조건·변경 설정·수치·목록·한계를 남기는 양식 |
| W05B MF와 SGD | [W05B 실습 A · SGD 한 단계](https://colab.research.google.com/github/lunalab-ai/recommender/blob/2026-fall-w05b/notebooks/student/w05b-sgd-step.ipynb) | 한국어 | 관측·내적·편향·기울기를 숫자로 계산 |
| W05B MF와 SGD | [W05B SGD 손계산 워크북](https://fancy-ballcap-a15.notion.site/W05B-SGD-3ea7fd00109f81858004ec041a8e151d) | 한국어 | 미분을 한 줄씩 읽고 여섯 모수의 갱신을 따라가기 |

## 차시별 출처와 참고 링크

각 강의의 참고자료 절과 본문 외부 링크를 모았습니다. 출처 구분 설명은 강의 원문을 유지합니다.

### W01A · 오리엔테이션과 추천시스템의 세계

[강의 원문](notion/sessions/w01a-orientation-and-recommenders.md)

- 강의 소개와 평가: 2026학년도 2학기 공식 강의계획서 및 `course/course.yml`
- 추천시스템 정의, 방법의 범주, Netflix·Amazon 사례: 임일, 『AI 에이전트를 위한 개인화 추천 알고리즘: Python, 머신러닝, AI, LLM 활용』, 도서출판청람, 2025, 1장, pp. 2–9
- 실제 서비스 관찰 링크와 신호 설명: YouTube 고객센터, Amazon Science, Spotify Support, Google 뉴스 고객센터의 공식 공개 페이지(2026-09-01 확인)
- 발견과 선택을 지원한다는 표현, 가상 화면과 네 요소 활동: 수업을 위한 추가 교육적 해석과 독자적 예시
- 이 페이지의 모든 도식은 수업을 위해 새로 제작했으며 교재 그림이나 서비스 화면을 복제하지 않았습니다.

본문에서 함께 소개한 링크:

- [YouTube 홈](https://www.youtube.com/)
- [맞춤 동영상 공식 설명](https://support.google.com/youtube/answer/16089387?hl=ko)
- [Amazon](https://www.amazon.com/)
- [추천시스템 공식 연구 소개](https://www.amazon.science/publications/two-decades-of-recommender-systems-at-amazon-com)
- [Spotify Web Player](https://open.spotify.com/)
- [Made For You 공식 설명](https://support.spotify.com/us/article/find-playlists/)
- [Google 뉴스의 내 뉴스](https://news.google.com/foryou?hl=ko&gl=KR&ceid=KR%3Ako)
- [기사 선택 공식 설명](https://support.google.com/googlenews/answer/9005749?hl=ko)

### W01B · Colab 환경 구축과 첫 추천 앱

[강의 원문](notion/sessions/w01b-colab-and-first-recommender-app.md)

- MovieLens 100K 데이터와 사용 조건: [GroupLens 공식 데이터 페이지](https://grouplens.org/datasets/movielens/100k/)와 [공식 README](https://files.grouplens.org/datasets/movielens/ml-100k-README.txt)
- Colab runtime, notebook 공유와 GitHub 연동: [Google Colab FAQ](https://research.google.com/colaboratory/intl/en-GB/faq.html)
- Repository clone: [GitHub 공식 문서](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository)
- Python 복습: [Python 한국어 자습서](https://docs.python.org/ko/3/tutorial/)의 3장과 4장
- pandas 보충: [pandas Getting started tutorials](https://pandas.pydata.org/docs/getting_started/intro_tutorials/index.html)의 표 읽기, 선택, 요약 통계
- Gradio 앱 구조: [Gradio Blocks 문서](https://www.gradio.app/docs/gradio/blocks)와 [share link 설명](https://www.gradio.app/guides/understanding-gradio-share-links)
- 교재 기반 범위: 임일, 『AI 에이전트를 위한 개인화 추천 알고리즘: Python, 머신러닝, AI, LLM 활용』, 도서출판청람, 2025, 2장, pp. 12–17
- 교재에서 가져온 것은 MovieLens 세 파일의 구조와 평균 평점 기반 추천의 출발점입니다. 최소 평점 수 비교, 앱 계층, 오류 진단, GUI와 활동은 수업을 위해 새로 구성했습니다.
- 이 페이지의 모든 도식과 코드는 수업을 위해 독자적으로 제작했으며 교재 페이지, 교재 코드 또는 서비스 화면을 복제하지 않았습니다.

본문에서 함께 소개한 링크:

- [W01B 학생용 Colab 열기](https://colab.research.google.com/github/lunalab-ai/recommender/blob/2026-fall-w01b/notebooks/student/w01b-colab-and-first-recommender-app.ipynb)
- [학생용 GitHub 저장소](https://github.com/lunalab-ai/recommender)
- [API: arguments, results and examples](https://github.com/lunalab-ai/recommender/blob/2026-fall-w02b/src/API.md)
- [download_movielens_100k](https://github.com/lunalab-ai/recommender/blob/2026-fall-w02b/src/luna_recsys/data.py#L222)

### W02A · 기본적인 추천 방법 (1)

[강의 원문](notion/sessions/w02a-basic-recommendation-methods-1.md)

- 임일, 『AI 에이전트를 위한 개인화 추천 알고리즘: Python, 머신러닝, AI, LLM 활용』, 청람, 2025, 2.2–2.4, pp.16–26. 인기제품·집단별 추천·RMSE의 수업 전개와 용어를 참고했다. 평점 수 비교, 대체 규칙, 앱과 연습 예는 수업용 추가 구성이다.
- [GroupLens MovieLens 100K](https://grouplens.org/datasets/movielens/100k/): 데이터 출처. [README](https://files.grouplens.org/datasets/movielens/ml-100k-README.txt)의 사용 조건을 따른다. 원자료를 수업 저장소에 재배포하지 않는다.
- [pandas: 요약 통계](https://pandas.pydata.org/docs/getting_started/intro_tutorials/06_calculate_statistics.html): `groupby`·평균·개수 집계 복습.
- [pandas: 표 결합](https://pandas.pydata.org/docs/getting_started/intro_tutorials/08_combine_dataframes.html): 공통 키와 `merge` 복습.
- [scikit-learn: 데이터 누수](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage): 학습 통계와 평가값의 분리.
- [scikit-learn: RMSE](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.root_mean_squared_error.html): 평점 예측 오차 계산 API.
- [Gradio: 화면 배치](https://gradio.app/guides/controlling-layout): 줄바꿈 가능한 열과 탭 구성.

시각자료는 수업을 위해 새로 제작했다. 개념 그림은 생성 이미지이며 도식과 성능 그래프는 재현 가능한 코드로 작성했다.

본문에서 함께 소개한 링크:

- [재사용 코드 안내](https://github.com/lunalab-ai/recommender/blob/main/src/luna_recsys/README.md)
- [준비와 분할 API 안내](https://github.com/lunalab-ai/recommender/blob/2026-fall-w02b/src/luna_recsys/README.md)
- [API: arguments, results and examples](https://github.com/lunalab-ai/recommender/blob/2026-fall-w02b/src/API.md)
- [build_comparison_app](https://github.com/lunalab-ai/recommender/blob/2026-fall-w02b/src/luna_recsys/comparison_app.py#L51)
- [prepare_movielens](https://github.com/lunalab-ai/recommender/blob/2026-fall-w02b/src/luna_recsys/datasets.py#L62)
- [split_ratings](https://github.com/lunalab-ai/recommender/blob/2026-fall-w02b/src/luna_recsys/evaluation.py#L28)
- [evaluate_means](https://github.com/lunalab-ai/recommender/blob/2026-fall-w02b/src/luna_recsys/evaluation.py#L60)
- [FourMethodRecommender](https://github.com/lunalab-ai/recommender/blob/2026-fall-w02b/src/luna_recsys/ranking.py#L18)
- [evaluate_rankings](https://github.com/lunalab-ai/recommender/blob/2026-fall-w02b/src/luna_recsys/ranking.py#L186)
- [MeanRatingPredictor](https://github.com/lunalab-ai/recommender/blob/2026-fall-w02b/src/luna_recsys/rating_models.py#L29)
- [MeanRatingPredictor.fit](https://github.com/lunalab-ai/recommender/blob/2026-fall-w02b/src/luna_recsys/rating_models.py#L77)
- [MeanRatingPredictor.predict](https://github.com/lunalab-ai/recommender/blob/2026-fall-w02b/src/luna_recsys/rating_models.py#L147)

### W02B · 기본적인 추천 방법 (2)

[강의 원문](notion/sessions/w02b-basic-recommendation-methods-2.md)

강의 원문에 참고자료 절이 아직 작성되지 않았습니다.

본문에서 함께 소개한 링크:

- [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/)
- [준비와 분할 API 안내](https://github.com/lunalab-ai/recommender/blob/2026-fall-w02b/src/luna_recsys/README.md)
- [scikit-learn의 누수 안내](https://scikit-learn.org/stable/common_pitfalls.html)
- [Google의 내용 기반 추천 설명](https://developers.google.com/machine-learning/recommendation/content-based/basics)
- [공식 API](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [TF–IDF 공식 계산 설명](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html)
- [NDCG 공식 API 설명](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html)
- [API: arguments, results and examples](https://github.com/lunalab-ai/recommender/blob/2026-fall-w02b/src/API.md)
- [build_comparison_app](https://github.com/lunalab-ai/recommender/blob/2026-fall-w02b/src/luna_recsys/comparison_app.py#L51)
- [prepare_movielens](https://github.com/lunalab-ai/recommender/blob/2026-fall-w02b/src/luna_recsys/datasets.py#L62)
- [split_ratings](https://github.com/lunalab-ai/recommender/blob/2026-fall-w02b/src/luna_recsys/evaluation.py#L28)
- [evaluate_means](https://github.com/lunalab-ai/recommender/blob/2026-fall-w02b/src/luna_recsys/evaluation.py#L60)
- [FourMethodRecommender](https://github.com/lunalab-ai/recommender/blob/2026-fall-w02b/src/luna_recsys/ranking.py#L18)
- [evaluate_rankings](https://github.com/lunalab-ai/recommender/blob/2026-fall-w02b/src/luna_recsys/ranking.py#L186)
- [MeanRatingPredictor](https://github.com/lunalab-ai/recommender/blob/2026-fall-w02b/src/luna_recsys/rating_models.py#L29)
- [MeanRatingPredictor.fit](https://github.com/lunalab-ai/recommender/blob/2026-fall-w02b/src/luna_recsys/rating_models.py#L77)
- [MeanRatingPredictor.predict](https://github.com/lunalab-ai/recommender/blob/2026-fall-w02b/src/luna_recsys/rating_models.py#L147)

### W03A · 협업 필터링의 기본 원리와 유사도 기반 추천

[강의 원문](notion/sessions/w03a-collaborative-filtering-basics.md)

- 임일(2025), 『AI 에이전트를 위한 개인화 추천 알고리즘: Python, 머신러닝, AI, LLM 활용』, 청람, 3.1–3.3, pp.34–41. 표·그림은 개념을 참고해 수업용으로 독자 재구성했다.
- [scikit-learn cosine_similarity](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html): 정규화 내적과 입력/출력 차원.
- [SciPy pearsonr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html): 중심화와 정의 불가 조건.
- [SciPy Jaccard distance](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jaccard.html): 거리와 집합 유사도.
- [Surprise 기본 이웃 알고리즘](https://surprise.readthedocs.io/en/stable/knn_inspired.html): 양의 유사도 가중평균의 보충 참고. 해당 패키지를 실습 의존성으로 추가하지 않는다.
- [GroupLens MovieLens 100K](https://grouplens.org/datasets/movielens/100k/): 실제 평점 데이터의 공식 출처.
- [scikit-learn 데이터 누수](https://scikit-learn.org/stable/common_pitfalls.html): train/test 분리.

외부 문서 확인일: 2026-09-14. 성능 실험은 2026-09-15 한국시간에 실행했다.

본문에서 함께 소개한 링크:

- [W03A 코드 사용 안내](https://github.com/lunalab-ai/recommender/blob/2026-fall-w03a/src/W03A-API.md)

### W03B · 협업 필터링 따라가기: 평점 행렬부터 추천 근거까지

[강의 원문](notion/sessions/w03b-cf-step-by-step.md)

- 임일(2025), 『AI 에이전트를 위한 개인화 추천 알고리즘』, 청람, 3.1–3.3, 인쇄 34–41쪽. 원본 스캔과 출판사 코드는 이 자료에 포함하지 않았다.
- [W03A 강의와 기존 실습](https://github.com/lunalab-ai/recommender/tree/2026-fall-w03a): 기존 수업용 합성 표와 독자 구현을 재사용했다.
- [scikit-learn cosine_similarity](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html): 정규화 내적과 입력·출력 차원. 확인 2026-09-16.
- [Gradio 6.27.0](https://pypi.org/project/gradio/6.27.0/): 함수와 앱 입력·출력 연결, 실행 중 런타임의 공유 방식. 확인 2026-09-17.

본문에서 함께 소개한 링크:

- [기존 W03A 강의](https://fancy-ballcap-a15.notion.site/W03a-3db7fd00109f8184ad2df949b899097c)
- [보충 Colab](https://colab.research.google.com/github/lunalab-ai/recommender/blob/2026-fall-w03b/notebooks/student/w03b-cf-step-by-step.ipynb)
- [기존 MovieLens Colab](https://colab.research.google.com/github/lunalab-ai/recommender/blob/2026-fall-w03a/notebooks/student/w03a-collaborative-filtering-basics.ipynb)
- [W03A 코드 안내](https://github.com/lunalab-ai/recommender/blob/2026-fall-w03a/src/W03A-API.md)

### W04A · 이웃 기반 CF의 개선

[강의 원문](notion/sessions/w04a-neighborhood-cf.md)

- 주교재 3.4–3.6: 이웃 선택, 이웃 크기, 사용자 평가 경향 보정의 용어와 설명 흐름을 따른다. 그림3-2/3-3은 개념만 참고했으며 본문의 도식·합성 예·코드·측정 그래프는 수업용 독립 제작이다. 교재 스캔과 출판사 코드를 배포하지 않는다.
- [Surprise 공식 k-NN 문서](https://surprise.readthedocs.io/en/stable/knn_inspired.html): 원평점/사용자 평균 보정 수식 교차 확인. 이 실습은 Surprise 패키지 실행이 아닌 자체 구현이다.
- [scikit-learn 공식 교차검증 안내](https://scikit-learn.org/stable/modules/cross_validation.html): 검증과 최종 테스트의 역할을 구분하는 보충 설명.
- [GroupLens MovieLens 100K](https://grouplens.org/datasets/movielens/100k/): 실제 데이터의 공식 배포처. 원본을 이 저장소에 재배포하지 않는다.
- [W03B 계산 복습](https://github.com/lunalab-ai/recommender/blob/2026-fall-w03b/course/notion/sessions/w03b-cf-step-by-step.md): 관측 마스크·코사인·가중평균의 연결.

외부 문서와 측정 조건 확인: 2026-09-22. 검증 데이터 분리, 명시적 대체/clip 정책, 앱의 근거 표시는 이해와 재현성을 위한 수업 보충이다.

공통 API 상세: [NeighborCF 생성·학습](https://github.com/lunalab-ai/recommender/blob/2026-fall-w04a-v2/src/luna_recsys/neighborhood.py#L11), [예측 상세](https://github.com/lunalab-ai/recommender/blob/2026-fall-w04a-v2/src/luna_recsys/neighborhood.py#L57), [이웃 근거](https://github.com/lunalab-ai/recommender/blob/2026-fall-w04a-v2/src/luna_recsys/neighborhood.py#L100), [검증 비교](https://github.com/lunalab-ai/recommender/blob/2026-fall-w04a-v2/src/luna_recsys/neighborhood.py#L124), [데이터 준비](https://github.com/lunalab-ai/recommender/blob/2026-fall-w04a-v2/src/luna_recsys/datasets.py#L62), [관측 분할](https://github.com/lunalab-ai/recommender/blob/2026-fall-w04a-v2/src/luna_recsys/evaluation.py#L28).

표준 API 참고: [pandas concat](https://pandas.pydata.org/docs/reference/api/pandas.concat.html), [DataFrame.mean](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.mean.html), [RMSE](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.root_mean_squared_error.html), [Gradio Blocks](https://www.gradio.app/docs/gradio/blocks).

본문에서 함께 소개한 링크:

- [학생 Colab 실습](https://colab.research.google.com/github/lunalab-ai/recommender/blob/2026-fall-w04a-v2/notebooks/student/w04a-neighborhood-cf.ipynb)
- [강의 PDF](https://github.com/lunalab-ai/recommender/blob/2026-fall-w04a-v2/course/handouts/w04a-neighborhood-cf.pdf)
- [점검 퀴즈 해설 PDF](https://github.com/lunalab-ai/recommender/blob/2026-fall-w04a-v2/course/handouts/w04a-neighborhood-cf-quiz.pdf)
- [앱과 callback 정의](https://github.com/lunalab-ai/recommender/blob/2026-fall-w04a-v2/src/luna_recsys/neighborhood_app.py#L8)

### W04B · 협업 필터링 총정리

[강의 원문](notion/sessions/w04b-cf-synthesis.md)

CF는 관측 행렬에서 비교 가능한 이웃을 찾고 그들의 평가를 집계하는 과정이다. 공통수·이웃수·평균 보정·축·clip은 서로 다른 부분을 바꾼다. 평가할 때는 평점의 오차, 목록의 적중과 순서, 추천 근거와 노출 범위를 구별하고 데이터 분리와 분모를 함께 제시한다.

주교재 3.7–3.9(인쇄면 53–62)와 3장 제공 예제의 용어·알고리즘을 기준으로 설명했다. 도표는 스캔을 복사하지 않고 독자 값과 배치로 다시 만들었다. NDCG 복습, 연속 shrinkage 비교, 검증 절차, 명시적 fallback·coverage 정의 및 웹 구현은 수업을 위한 추가 설명이다.

- [Surprise KNN 알고리즘](https://surprise.readthedocs.io/en/stable/knn_inspired.html): 최소 이웃과 예측식 비교. 라이브러리의 fallback·유사도 정의를 이 실습과 동일하다고 가정하지 않는다.
- [Surprise 유사도 정의](https://surprise.readthedocs.io/en/stable/similarities.html): 공통 평가 좌표의 코사인과 이번 전체 차원 코사인의 차이를 확인한다.
- [Stanford IR: Precision과 Recall](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-unranked-retrieval-sets-1.html), [순위 평가](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html): 관련성과 순위 평가의 구분.
- [GroupLens MovieLens 100K](https://grouplens.org/datasets/movielens/100k/): 데이터 출처와 이용 조건. 원자료는 공개 저장소에 넣지 않는다.
- [scikit-learn: 교차 검증과 모델 선택](https://scikit-learn.org/stable/modules/cross_validation.html): 설정 선택과 최종 평가의 분리.

- 구현 문법: [NumPy 행렬곱 `@`](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html), [pandas `pivot`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.pivot.html), [scikit-learn `cosine_similarity`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html).

확인일: 2026-09-23. 공개 고정 버전 설치·실제 Colab·Notion 가져오기는 로컬 검증과 별도로 배포 단계에서 확인한다.

본문에서 함께 소개한 링크:

- [학생 Colab 실습](https://colab.research.google.com/github/lunalab-ai/recommender/blob/2026-fall-w04b/notebooks/student/w04b-cf-synthesis.ipynb)
- [predict_details(pairs)](https://github.com/lunalab-ai/recommender/blob/2026-fall-w04b/src/luna_recsys/cf_synthesis.py#L110)
- [explain(user_id, movie_id)](https://github.com/lunalab-ai/recommender/blob/2026-fall-w04b/src/luna_recsys/cf_synthesis.py#L155)
- [recommend(user_id, movies, top_n)](https://github.com/lunalab-ai/recommender/blob/2026-fall-w04a-v2/src/luna_recsys/collaborative.py#L198)
- [configured(**settings)](https://github.com/lunalab-ai/recommender/blob/2026-fall-w04b/src/luna_recsys/cf_synthesis.py#L72)
- [rating_report(model, heldout)](https://github.com/lunalab-ai/recommender/blob/2026-fall-w04b/src/luna_recsys/cf_synthesis.py#L175)
- [evaluate_cf(..., top_n=10, ranking_users=None)](https://github.com/lunalab-ai/recommender/blob/2026-fall-w04b/src/luna_recsys/cf_synthesis.py#L197)
- [EvidenceCF와 평가 함수](https://github.com/lunalab-ai/recommender/blob/2026-fall-w04b/src/luna_recsys/cf_synthesis.py#L13)
- [CFLab·callback·앱](https://github.com/lunalab-ai/recommender/blob/2026-fall-w04b/src/luna_recsys/synthesis_app.py#L52)
- [데이터 준비 함수](https://github.com/lunalab-ai/recommender/blob/2026-fall-w04a-v2/src/luna_recsys/datasets.py#L62)
- [분리 함수](https://github.com/lunalab-ai/recommender/blob/2026-fall-w04a-v2/src/luna_recsys/evaluation.py#L28)

### W05A · 추천시스템 핵심 복습과 동일 데이터 모델 비교·튜닝·평가

[강의 원문](notion/sessions/w05a-model-comparison.md)

강의 원문에 참고자료 절이 아직 작성되지 않았습니다.

본문에서 함께 소개한 링크:

- [실습 A: 모델·지표 복습](https://colab.research.google.com/github/lunalab-ai/recommender/blob/2026-fall-w05a-v2/notebooks/student/w05a-review-metrics.ipynb)
- [실습 B: 비교·튜닝·웹 앱](https://colab.research.google.com/github/lunalab-ai/recommender/blob/2026-fall-w05a-v2/notebooks/student/w05a-model-comparison.ipynb)
- [전체 실험 CSV와 분할·선택 기록](https://github.com/lunalab-ai/recommender/tree/2026-fall-w05a-v2/data/sample/w05a-results)
- [ModelSpec](https://github.com/lunalab-ai/recommender/blob/2026-fall-w05a-v2/src/luna_recsys/model_comparison.py#L21)
- [ComparisonSuite](https://github.com/lunalab-ai/recommender/blob/2026-fall-w05a-v2/src/luna_recsys/model_comparison.py#L111)
- [ComparisonSuite.model](https://github.com/lunalab-ai/recommender/blob/2026-fall-w05a-v2/src/luna_recsys/model_comparison.py#L131)
- [ComparisonSuite.recommend](https://github.com/lunalab-ai/recommender/blob/2026-fall-w05a-v2/src/luna_recsys/model_comparison.py#L150)
- [ComparisonSuite.evaluate](https://github.com/lunalab-ai/recommender/blob/2026-fall-w05a-v2/src/luna_recsys/model_comparison.py#L172)
- [compare](https://github.com/lunalab-ai/recommender/blob/2026-fall-w05a-v2/src/luna_recsys/model_comparison.py#L226)
- [select_best](https://github.com/lunalab-ai/recommender/blob/2026-fall-w05a-v2/src/luna_recsys/model_comparison.py#L241)
- [split_three](https://github.com/lunalab-ai/recommender/blob/2026-fall-w05a-v2/src/luna_recsys/model_comparison.py#L81)
- [ranking_metrics](https://github.com/lunalab-ai/recommender/blob/2026-fall-w05a-v2/src/luna_recsys/ranking.py#L163)
- [ComparisonLab](https://github.com/lunalab-ai/recommender/blob/2026-fall-w05a-v2/src/luna_recsys/comparison_lab.py#L16)
- [ComparisonLab.metrics](https://github.com/lunalab-ai/recommender/blob/2026-fall-w05a-v2/src/luna_recsys/comparison_lab.py#L32)
- [comparison_view](https://github.com/lunalab-ai/recommender/blob/2026-fall-w05a-v2/src/luna_recsys/comparison_lab.py#L39)
- [build_comparison_app](https://github.com/lunalab-ai/recommender/blob/2026-fall-w05a-v2/src/luna_recsys/comparison_lab.py#L87)
- [GroupLens MovieLens100K](https://grouplens.org/datasets/movielens/100k/)
- [공식 README](https://files.grouplens.org/datasets/movielens/ml-100k-README.txt)
- [Stanford IR: 집합 기반 평가](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-unranked-retrieval-sets-1.html)
- [순위 기반 평가](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html)
- [scikit-learn: 검증과 test 분리](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Surprise: 이웃 기반 알고리즘](https://surprise.readthedocs.io/en/stable/knn_inspired.html)
- [Gradio Blocks](https://gradio.app/docs/gradio/blocks)

### W05B · 행렬요인화(MF)와 SGD

[강의 원문](notion/sessions/w05b-mf-sgd.md)

강의 원문에 참고자료 절이 아직 작성되지 않았습니다.

본문에서 함께 소개한 링크:

- [실험 설정과 epoch별 수치](https://github.com/lunalab-ai/recommender/tree/2026-fall-w05b/data/sample/w05b-results)
- [Google MF 설명](https://developers.google.com/machine-learning/recommendation/collaborative/matrix)
- [Surprise MF 문서](https://surprise.readthedocs.io/en/stable/matrix_factorization.html)
- [GroupLens 공식 데이터 설명](https://files.grouplens.org/datasets/movielens/ml-100k-README.txt)
- [Gradio Blocks](https://www.gradio.app/docs/gradio/blocks)
- [Colab A · 한 단계 손계산](https://colab.research.google.com/github/lunalab-ai/recommender/blob/2026-fall-w05b/notebooks/student/w05b-sgd-step.ipynb)
- [Colab B · MF 학습과 웹 앱](https://colab.research.google.com/github/lunalab-ai/recommender/blob/2026-fall-w05b/notebooks/student/w05b-mf-sgd.ipynb)
- [MFSGD](https://github.com/lunalab-ai/recommender/blob/2026-fall-w05b/src/luna_recsys/mf_sgd.py#L54)
- [MFSGD.fit](https://github.com/lunalab-ai/recommender/blob/2026-fall-w05b/src/luna_recsys/mf_sgd.py#L73)
- [MFSGD.predict](https://github.com/lunalab-ai/recommender/blob/2026-fall-w05b/src/luna_recsys/mf_sgd.py#L150)
- [MFSGD.explain](https://github.com/lunalab-ai/recommender/blob/2026-fall-w05b/src/luna_recsys/mf_sgd.py#L133)
- [MFSGD.rmse](https://github.com/lunalab-ai/recommender/blob/2026-fall-w05b/src/luna_recsys/mf_sgd.py#L154)
- [MFSGD.recommend](https://github.com/lunalab-ai/recommender/blob/2026-fall-w05b/src/luna_recsys/mf_sgd.py#L165)
- [sgd_step](https://github.com/lunalab-ai/recommender/blob/2026-fall-w05b/src/luna_recsys/mf_sgd.py#L15)
- [step_view](https://github.com/lunalab-ai/recommender/blob/2026-fall-w05b/src/luna_recsys/mf_lab.py#L10)
- [training_view](https://github.com/lunalab-ai/recommender/blob/2026-fall-w05b/src/luna_recsys/mf_lab.py#L29)
- [build_mf_app](https://github.com/lunalab-ai/recommender/blob/2026-fall-w05b/src/luna_recsys/mf_lab.py#L45)
