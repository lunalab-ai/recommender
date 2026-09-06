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
