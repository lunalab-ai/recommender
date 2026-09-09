# 수업 공통 코드 사용 안내

입력·기본값·출력·상태 설명은 구현과 대조한 docstring에서 가져옵니다. 정의 링크는 같은 공개 버전의 실제 선언 줄을 가리킵니다.

## popularity_ranking

[src/luna_recsys/baselines.py · definition](https://github.com/lunalab-ai/recommender/blob/2026-fall-w02b/src/luna_recsys/baselines.py#L11)

```python
popularity_ranking(ratings: pd.DataFrame | Iterable[Mapping[str, Any]], *, item_col: str='item_id', rating_col: str='rating', min_count: int=1)
```

Rank items by interaction count, then mean rating.

Parameters
----------
ratings:
    A pandas DataFrame or iterable of row mappings.
item_col:
    Item identifier column.
rating_col:
    Explicit rating column.
min_count:
    Minimum number of ratings required for an item.

Returns
-------
pandas.DataFrame
    Columns: item identifier, ``rating_count``, ``mean_rating``.

## mean_rating_recommendations

[src/luna_recsys/baselines.py · definition](https://github.com/lunalab-ai/recommender/blob/2026-fall-w02b/src/luna_recsys/baselines.py#L58)

```python
mean_rating_recommendations(ratings: pd.DataFrame, movies: pd.DataFrame, *, genre: str | None=None, min_ratings: int=50, top_n: int=10)
```

영화 평균 평점의 상위 목록을 반환하는 W01B 함수.

ratings: movie_id/rating 열의 DataFrame. movies: movie_id/title과 필요한
장르 열의 DataFrame. genre: 기본None이면 전체 장르, 문자열이면 해당 열1만.
min_ratings: 최소 관측 수, 기본50, 1이상. top_n: 최대 목록 길이, 기본10.

영화별 개수/평균 집계→최소 표본 필터→영화 제목/장르 연결→평균 내림차순,
개수 내림차순, 제목/ID 오름차순으로 정렬한다. 출력은 최대top_n행의
movie_id/title/mean_rating/rating_count 표이며 마지막 표시에서만 평균을
소수3자리로 반올림한다. 조건에 맞는 영화가 없으면 빈 표다. 사용자 이력
제외/학습평가분할은 수행하지 않는다. 입력 표를 수정하지 않는다. 잘못된
열/장르/최소값은 ValueError. W02B 공통 순위평가 API와 동점/후보 규칙이 다르다.

## baseline_recommendations

[src/luna_recsys/baselines.py · definition](https://github.com/lunalab-ai/recommender/blob/2026-fall-w02b/src/luna_recsys/baselines.py#L114)

```python
baseline_recommendations(ratings: pd.DataFrame, movies: pd.DataFrame, *, method: str='mean', users: pd.DataFrame | None=None, group_col: str='sex', group_value: str='F', genre: str | None=None, min_ratings: int=5, top_n: int=10)
```

W02A 앱의 평점수/영화평균/집단평균 목록을 공통 형식으로 반환한다.

ratings: user_id/movie_id/rating DataFrame. movies: movie_id/title/장르 표.
method: mean(기본)/count/group. users: 기본None, group에서 사용자속성필수.
group_col: 집단 열 기본sex; group_value: 선택집단 기본F, 문자열로 비교.
genre: 기본None(전체), 지정한 장르0/1열로 필터. min_ratings: 기본5, top_n:
기본10, 모두 양수. 입력 표는 변경하지 않는다.

반환 최대top_n행은 movie_id/title/mean_rating/rating_count/basis 열.
count는 개수→평균 내림차순→ID, mean/group은 mean_rating_recommendations의
동점 규칙을 따른다. group은 선택집단 관측만 집계하고 조건에 맞는 목록이
전혀 없으면 전체 사용자 영화평균 목록으로 전환해 basis에 표시한다.
개별 평점의 group→movie→global 대체와 다르다. 학습 이력 제외/holdout은
이 함수 자체에서 하지 않으며 W02B 성능표는 FourMethodRecommender로 계산한다.

## comparison_table

[src/luna_recsys/comparison_app.py · definition](https://github.com/lunalab-ai/recommender/blob/2026-fall-w02b/src/luna_recsys/comparison_app.py#L8)

```python
comparison_table(model: FourMethodRecommender, user_id: int, k: int=10)
```

동일 user_id의 네 상위 목록을 합쳐 화면용 DataFrame을 반환한다.

model: fit이 끝난 FourMethodRecommender. user_id: 카탈로그에 추천할
사용자 ID(정수). k: 방법별 최대 추천 수, 기본10인 양의 정수다.
반환 열은 method/rank/movie_id/title/genres/score/basis이다.
rank는 방법마다1부터 시작한다. 최대4k행이며 score는 원래 단위를 유지.
입력 모델을 재학습/변경하지 않는다. 잘못된 k나 미학습 모델의 오류는
숨기지 않고 전달한다. Gradio import 없이도 callback을 검사할 수 있다.

## comparison_cards

[src/luna_recsys/comparison_app.py · definition](https://github.com/lunalab-ai/recommender/blob/2026-fall-w02b/src/luna_recsys/comparison_app.py#L28)

```python
comparison_cards(table: pd.DataFrame)
```

comparison_table 결과에서 방법별 상위 3개를 반응형 HTML로 표시한다.

table: method/rank/title/genres/score/basis 열이 있는 DataFrame.
반환값: gradio.HTML에 넣을 문자열. 원본 table과 점수는 변경하지 않는다.
제목과 근거는 HTML 이스케이프하여 텍스트로 표시한다. 작은 화면에서는
카드가 한 열로 배치되며 전체 순위는 별도 표에서 확인한다.

## build_comparison_app

[src/luna_recsys/comparison_app.py · definition](https://github.com/lunalab-ai/recommender/blob/2026-fall-w02b/src/luna_recsys/comparison_app.py#L51)

```python
build_comparison_app(model: FourMethodRecommender)
```

학습된 model을 사용하는 gradio.Blocks 객체를 만들고 반환한다.

model: fit이 끝난 FourMethodRecommender. 사용자 선택은 학습 사용자 ID,
목록 길이는1–20이다. 반환 앱의 launch(share=True)는 호출자가 Colab에서
실행한다. 이 함수 자체는 네트워크 서버를 띄우거나 데이터를 다운로드하지
않는다. [apps] 선택 의존성인 Gradio가 없으면 ImportError가 발생한다.
화면 입력은 callback에서 정수로 변환하여 comparison_table에 전달한다.

## MovieLens100K

[src/luna_recsys/data.py · definition](https://github.com/lunalab-ai/recommender/blob/2026-fall-w02b/src/luna_recsys/data.py#L58)

```python
MovieLens100K
```

사용자·영화·평점 세 표를 묶는 dataclass.

users: user_id/age/sex/occupation/zip_code의 DataFrame.
movies: movie_id/title/날짜/URL과19개 장르0·1열의 DataFrame.
ratings: user_id/movie_id/rating/timestamp의 DataFrame.
생성자 자체는 읽기/검사를 수행하지 않는다. load_movielens_100k가 파일을
읽고 validate_movielens_100k가 검사한다. 개인 속성/원본 평점을 공개 출력하지 않는다.

## course_root

[src/luna_recsys/data.py · definition](https://github.com/lunalab-ai/recommender/blob/2026-fall-w02b/src/luna_recsys/data.py#L72)

```python
course_root()
```

인자 없이 편집 가능한 checkout의 저장소 루트 Path를 반환한다.

현재 모듈 위치에서 계산하며 파일 생성/다운로드를 하지 않는다. wheel로
설치한 경우 같은 위치에 저장소 샘플이 있다고 보장하지 않으므로 학생은
prepare_movielens의 다운로드/독자 합성 생성기를 사용한다.

## data_path

[src/luna_recsys/data.py · definition](https://github.com/lunalab-ai/recommender/blob/2026-fall-w02b/src/luna_recsys/data.py#L81)

```python
data_path(filename: str, *, local_only: bool=False)
```

filename의 데이터 파일 Path를 확인하여 반환한다.

filename: 상대 파일 이름 문자열. local_only: 기본False이면 data/sample,
True이면 data/local에서 찾는다. RECOMMENDER_DATA_DIR 환경변수가 있으면
그 폴더가 우선한다. 파일이 없으면 FileNotFoundError이며 자동 다운로드를
하지 않는다. data_path("ratings_tiny.csv")처럼 사용한다.

## load_movielens_100k

[src/luna_recsys/data.py · definition](https://github.com/lunalab-ai/recommender/blob/2026-fall-w02b/src/luna_recsys/data.py#L118)

```python
load_movielens_100k(path: str | Path, *, expect_full: bool=False)
```

path의 세 파일을 읽고 검사한 MovieLens100K를 반환한다.

path: u.user/u.item/u.data 또는 ml-100k 하위 폴더를 가진 폴더 경로.
expect_full: 기본False이면 작은 형식 예제도 허용, True이면943명·1682편·
100000평점 크기까지 요구한다. user/item은 |, data는 탭 구분자이며 헤더가
없으므로 정해진 열 이름을 사용한다. latin-1로 읽는다. 원본 파일은 수정하지
않는다. 파일 부재/형식/참조ID/평점 범위 오류를 전달한다.
예: data = load_movielens_100k("data/local/ml-100k", expect_full=True).

## validate_movielens_100k

[src/luna_recsys/data.py · definition](https://github.com/lunalab-ai/recommender/blob/2026-fall-w02b/src/luna_recsys/data.py#L151)

```python
validate_movielens_100k(dataset: MovieLens100K, *, expect_full: bool=False)
```

dataset 세 표의 키·범위·참조 관계를 검사하며 반환값은 None이다.

dataset: MovieLens100K. expect_full: 기본False, True이면 공식 전체 크기
943/1682/100000도 요구한다. 중복 사용자/영화ID, 중복 사용자–영화 평점,
잘못된1–5점 범위나 참조할 수 없는 ID를 거부한다. 파일 쓰기/표 수정은 없고
유효하지 않으면 ValueError. 성공은 데이터 사용 조건의 법적 인증이 아니다.

## extract_movielens_100k_archive

[src/luna_recsys/data.py · definition](https://github.com/lunalab-ai/recommender/blob/2026-fall-w02b/src/luna_recsys/data.py#L183)

```python
extract_movielens_100k_archive(archive_path: str | Path, cache_dir: str | Path, *, expected_md5: str=MOVIELENS_100K_MD5)
```

archive_path ZIP을 검증해 필요한3파일만 cache_dir 아래에 꺼낸다.

archive_path/cache_dir: 문자열 또는 Path. expected_md5: 기본 공식ZIP의
MD5 문자열(MOVIELENS_100K_MD5). 일치하는 archive에서 u.user/u.item/u.data만
사용하며 다른 멤버를 임의 경로로 풀지 않는다. 반환은 로딩할 데이터 폴더 Path.
디렉터리 생성/파일 쓰기가 발생한다. 잘못된 checksum·필수 멤버 누락·ZIP
오류는 예외로 전달한다. 학생 수동 업로드도 같은 검증 원리를 사용한다.

## download_movielens_100k

[src/luna_recsys/data.py · definition](https://github.com/lunalab-ai/recommender/blob/2026-fall-w02b/src/luna_recsys/data.py#L222)

```python
download_movielens_100k(cache_dir: str | Path, *, url: str=MOVIELENS_100K_URL, expected_md5: str=MOVIELENS_100K_MD5, timeout: float=60)
```

cache_dir에서 재사용하거나 검증된 ZIP을 받아 데이터 폴더 Path를 반환한다.

cache_dir: 저장 폴더 문자열/Path. url: 기본 GroupLens 공식ZIP 주소.
expected_md5: 기본 공식MD5, timeout: 요청 제한 초(기본60). 테스트용 URL과
검증값 변경은 가능한 API지만 일반 실습은 기본 주소/해시를 유지한다.
기존 필수3파일이 있으면 재사용한다. 공식 경로 실패 시 공식기본설정에서만
고정 HTTPS 미러를 시도하고 SHA256/MD5를 확인한다. TLS 검증을 끄지 않는다.
다운로드/임시파일/캐시 쓰기가 발생한다. 모든 경로 실패는 OSError이며
반환 후 전체 데이터 크기/키는 load_movielens_100k(expect_full=True)로 검사한다.

## PreparedMovieLens

[src/luna_recsys/datasets.py · definition](https://github.com/lunalab-ai/recommender/blob/2026-fall-w02b/src/luna_recsys/datasets.py#L16)

```python
PreparedMovieLens
```

준비된 데이터와 실제 모드를 함께 전달하는 dataclass.

data: MovieLens100K(users/movies/ratings DataFrame). mode: movielens는
다운로드/캐시, local은 명시한 전체 로컬 사본, synthetic은 독자 합성 자료.
description: 화면에 보여 줄 설명 문자열. frozen=True는 속성 재할당을
막지만 내부 DataFrame을 자동으로 불변으로 만들지는 않는다.

## synthetic_movielens

[src/luna_recsys/datasets.py · definition](https://github.com/lunalab-ai/recommender/blob/2026-fall-w02b/src/luna_recsys/datasets.py#L29)

```python
synthetic_movielens(seed: int=2026)
```

독자적인 CC0 가상 MovieLens 형식 자료를 메모리에 생성한다.

seed: 난수 정수, 기본2026. 반환 MovieLens100K는80명·18편·960평점이다.
실제 MovieLens의 사용자·영화·행을 추출하거나 변환한 데이터가 아니다.
파일이나 네트워크가 필요 없고 입력 데이터 변경도 없다.
synthetic_movielens(seed=7).ratings로 생성 평점 표를 얻는다.

## prepare_movielens

[src/luna_recsys/datasets.py · definition](https://github.com/lunalab-ai/recommender/blob/2026-fall-w02b/src/luna_recsys/datasets.py#L62)

```python
prepare_movielens(cache_dir: str | Path='data/local', *, local_dir: str | Path | None=None, mode: str='auto', timeout: float=20)
```

캐시/다운로드/명시한 사본을 준비하고 데이터 종류까지 반환한다.

cache_dir: 폴더 문자열 또는 Path, 기본 data/local. 다운로드 파일 저장 위치.
local_dir: 기본None. 이미 가진 전체3파일의 폴더를 명시할 때만 사용한다.
mode: auto(기본)는 다운로드 실패 때 합성 대안을 명확히 표시한다. real은
실제 자료 준비 실패를 오류로 전달한다. synthetic은 다운로드 없이 가상
자료를 생성한다. 명시한 잘못된 local_dir는 합성으로 대체하지 않는다.
timeout: 네트워크 요청 제한 초, 기본20, 양수. 전체 함수 총시간 상한은 아니다.

반환 PreparedMovieLens의 data.users/data.movies/data.ratings는 표이며
mode/description은 출처를 설명한다. 실제 전체 자료는943명/1682편/100000평점.
폴더 생성과 다운로드/캐시 사용이 부작용이다. 실제 비교 실습은 반드시
prepare_movielens("data/local", mode="real")로 호출한다.
설정 오류는 ValueError, real 다운로드 실패는 OSError 등으로 보고한다.

## recommend_for_app

[src/luna_recsys/demo_app.py · definition](https://github.com/lunalab-ai/recommender/blob/2026-fall-w02b/src/luna_recsys/demo_app.py#L11)

```python
recommend_for_app(ratings: pd.DataFrame, movies: pd.DataFrame, genre_label: str, min_ratings: int, top_n: int)
```

W01B 화면 입력을 추천 함수로 전달하고 한국어 열 이름의 표를 반환한다.

ratings/movies: movie_id/rating과 영화메타데이터 DataFrame. genre_label:
화면의 전체 또는 장르문자열. min_ratings/top_n: 양수 정수형으로 전달할
최소평점수/목록길이. 출력 열 영화 ID/영화 제목/평균 평점/평점 수.
mean_rating_recommendations에 위임하며 모델학습/서버실행/원본변경은 없다.

## build_movie_recommender_app

[src/luna_recsys/demo_app.py · definition](https://github.com/lunalab-ai/recommender/blob/2026-fall-w02b/src/luna_recsys/demo_app.py#L42)

```python
build_movie_recommender_app(ratings: pd.DataFrame, movies: pd.DataFrame, users: pd.DataFrame | None=None, *, data_mode: str='데이터 모드 미지정')
```

W01B Gradio Blocks 앱 객체를 생성해 반환한다.

ratings/movies: 로딩한 평점/영화 DataFrame, users: 선택 사용자표(기본None).
data_mode: 화면에 표시할 실제 데이터 설명문자열. 장르/최소관측수/Top-N
입력을 callback으로 연결한다. 함수 자체는 launch하지 않는다. 반환객체의
launch(share=True)는 Colab 서버 실행, close()는종료다. Gradio [apps] 설치가
필요하다. 입력에 원본 사용자속성을 공개표로 내보내지 않는다.

## build_baseline_lab

[src/luna_recsys/demo_app.py · definition](https://github.com/lunalab-ai/recommender/blob/2026-fall-w02b/src/luna_recsys/demo_app.py#L106)

```python
build_baseline_lab(dataset: MovieLens100K, *, data_mode: str='데이터 모드 미지정')
```

W02A 추천·평가 탭을 가진 Gradio Blocks를 반환한다.

dataset: MovieLens100K(users/movies/ratings 표). data_mode: 화면의 실제
데이터 설명문자열. 기존 인기목록의 장르/최소수/Top-N과 집단선택, 공통
holdout의 평균모형RMSE평가를 연결한다. 앱 준비 시 split_ratings(seed42)를
사용한다. 전체 앱 객체를 반환하며 launch는 호출자가 수행한다.
목록필터와 개별평점대체의 규칙은 각 탭의basis/level 설명을 따른다.
서버네트워크나데이터다운로드를 이 함수 자체에서 시작하지 않는다.

## RatingSplit

[src/luna_recsys/evaluation.py · definition](https://github.com/lunalab-ai/recommender/blob/2026-fall-w02b/src/luna_recsys/evaluation.py#L17)

```python
RatingSplit
```

분할 결과를 묶는 dataclass.

train/test: 원본 관측에서 선택한 독립 DataFrame 사본. method는
user-stratified 또는 random-small-data 문자열. frozen 속성이 내부표를
불변으로 만들지는 않으므로 평가 중 train/test를 덮어쓰지 않는다.

## split_ratings

[src/luna_recsys/evaluation.py · definition](https://github.com/lunalab-ai/recommender/blob/2026-fall-w02b/src/luna_recsys/evaluation.py#L28)

```python
split_ratings(ratings: pd.DataFrame, *, test_size: float=0.25, seed: int=42)
```

ratings의 행 위치를 한 번 나누어 RatingSplit(train,test,method)를 반환한다.

ratings: user_id/movie_id/rating을 포함한 관측 DataFrame, 최소2행.
test_size: 평가 비율0과1사이(기본0.25), seed: 난수 정수(기본42).
가능하면 user_id로 층화한다. 너무 작은 표의 대체는 random-small-data로
명시한다. 출력은 원본 인덱스/열을 보존하는 사본이며 원본을 변경하지 않는다.
MovieLens100K의 기본 설정은75000/25000이다. 같은 사용자가 양쪽에 존재할
수 있는 무작위 관측평가이며 시간분할/새 사용자평가가 아니다.
중복 관측쌍은 분할 전 전체 로더에서 검사한다. 비율/행 수 오류는 ValueError.

## evaluate_means

[src/luna_recsys/evaluation.py · definition](https://github.com/lunalab-ai/recommender/blob/2026-fall-w02b/src/luna_recsys/evaluation.py#L60)

```python
evaluate_means(split: RatingSplit, users: pd.DataFrame, *, group_col: str='sex', min_group_ratings: int=1)
```

동일한 split에서 세 평균 회귀모형을 학습·평가해 DataFrame을 반환한다.

split: RatingSplit의 train/test. users: 사용자ID와 집단속성 표.
group_col: 기본sex, min_group_ratings: 기본1인 최소 집단×영화 표본.
전체/영화/집단모형을 각각train에fit하고 test의ID로 예측한 뒤 test평점과
RMSE를 계산한다. 출력 열 method/rmse/test_count/group_used/movie_used/
global_used; 대체 건수의 합은 각 행의test_count다. 반올림 전에 평가하며
입력이나분할을변경하지않는다. 낮은RMSE가 순위만족도 개선을보증하지않는다.

## FourMethodRecommender

[src/luna_recsys/ranking.py · definition](https://github.com/lunalab-ai/recommender/blob/2026-fall-w02b/src/luna_recsys/ranking.py#L18)

```python
FourMethodRecommender
```

평점 수·영화 평균·집단 평균·장르 프로필을 같은 후보에서 비교한다.

Parameters
----------
group_col : str, default 'sex'
    users의 집단 열 이름. 'occupation'도 가능하다. 개인 취향과 집단을
    동일시하지 않는다. 이 데이터에 주어진 집단 속성만 사용한다.
min_group_ratings : int, default 1
    집단×영화 평균을 사용할 최소 학습 평점 수. 부족하면 영화 평균,
    영화 학습 이력도 없으면 전체 평균으로 대체한다.
like_threshold : float, default 4
    내용 프로필에 넣을 학습 평점의 하한(1–5). 평가의 관련성 기준과
    구별되는 모델 설정이며 test 결과를 보고 조정하지 않는다.

Notes
-----
생성자는 설정만 저장한다. fit(train, users, movies)이 학습을 수행한다.
movies의 전체 장르는 추천 시 이미 알려진 카탈로그 메타데이터로 가정한다.
genre_vectors_는 unknown을 뺀 18개 장르의 L2 정규화 행렬(M×18),
profiles_는 사용자 ID→정규화 선호 벡터 사전이다. 원자료 변경이 학습
상태를 바꾸지 않도록 복사한다. 이력 없는 사용자는 내용 방식에서 영화
평균 목록으로 대체한다. 장르 없는 영화는 정상 프로필에서 유사도 0이다.

## FourMethodRecommender.__init__

[src/luna_recsys/ranking.py · definition](https://github.com/lunalab-ai/recommender/blob/2026-fall-w02b/src/luna_recsys/ranking.py#L43)

```python
__init__(self, *, group_col: str='sex', min_group_ratings: int=1, like_threshold: float=4)
```

group_col/min_group_ratings/like_threshold 설정만 보관한다.

반환값은 없으며 아직 추천할 수 없다. 유효하지 않은 평점 하한이나
최소 집단 표본 수에는 ValueError가 발생한다.

## FourMethodRecommender.fit

[src/luna_recsys/ranking.py · definition](https://github.com/lunalab-ai/recommender/blob/2026-fall-w02b/src/luna_recsys/ranking.py#L59)

```python
fit(self, train: pd.DataFrame, users: pd.DataFrame, movies: pd.DataFrame)
```

train 레이블로 통계와 프로필을 학습하고 self를 반환한다.

train: N행 DataFrame, user_id/movie_id/rating(1–5) 필수, 쌍은 유일.
users: user_id와 group_col 필수, 한 사용자 한 행.
movies: movie_id/title과 MovieLens 장르 18열 필수, 한 영화 한 행.
train의 모든 ID는 users/movies에 있어야 한다. 추가 열은 허용한다.

counts_는 영화별 학습 관측 수, mean_model_/group_model_은 학습된
평균 예측기, seen_은 사용자별 학습 영화 집합이다. profiles_는
4점 이상(기본값) 학습 영화의 단위 장르 벡터 평균을 다시 단위벡터로
정규화한다. 빈/영벡터 프로필은 저장하지 않아 대체 규칙이 적용된다.
입력 표를 수정하지 않는다. 잘못된 스키마·중복·범위는 ValueError.

## FourMethodRecommender.score_catalog

[src/luna_recsys/ranking.py · definition](https://github.com/lunalab-ai/recommender/blob/2026-fall-w02b/src/luna_recsys/ranking.py#L110)

```python
score_catalog(self, user_id: int, method: str='content')
```

한 user_id의 전체 카탈로그 점수 표를 movie_id 순서로 반환한다.

method는 count/mean/group/content 중 하나. 반환 M행의 열은
movie_id, score(float), basis(str)이며 아직 본 영화도 포함한다.
count는 학습 평점 개수(없으면0), mean/group은 예측 평점,
content는 코사인 유사도(0–1)다. 내용 프로필이 없으면 이 사용자
목록 전체를 영화 평균으로 대체하고 basis에 'empty-profile/…'를
표시한다. 정상 프로필에서 장르 없는 영화는 'zero-genres'다.
fit 전 또는 알 수 없는 method에는 ValueError. 상태 변경은 없다.

## FourMethodRecommender.recommend

[src/luna_recsys/ranking.py · definition](https://github.com/lunalab-ai/recommender/blob/2026-fall-w02b/src/luna_recsys/ranking.py#L143)

```python
recommend(self, user_id: int, method: str='content', k: int=10)
```

user_id의 학습 영화만 제외하고 상위 k개를 반환한다.

method: 네 기준 중 하나, k: 양의 정수(기본10). score 내림차순,
동점은 movie_id 오름차순으로 정한다. 후보가 k개 미만이면 전부 반환.
출력 열: movie_id/score/basis/title/genres(장르 이름을 |로 연결).
score를 반올림해서 정렬하지 않는다. test에서 본 영화를 제외하면
정답 후보를 숨기는 오류가 되므로 test 이력을 받지 않는다.

## ranking_metrics

[src/luna_recsys/ranking.py · definition](https://github.com/lunalab-ai/recommender/blob/2026-fall-w02b/src/luna_recsys/ranking.py#L163)

```python
ranking_metrics(recommended: list[int], relevant: set[int], *, k: int=10)
```

사용자 한 명의 이진 관련성 순위 지표를 계산한다.

recommended: 점수순 영화 ID 목록, 중복 불허. 앞의 k개만 평가한다.
relevant: 후보에 포함되는 test 평점>=4 영화 ID 집합(비어 있으면 오류).
k: 양의 정수. Precision=적중/k, Recall=적중/관련 영화 수.
DCG=sum(hit_at_rank/log2(rank+1)), NDCG=DCG/IDCG.
IDCG는 min(k, 관련 영화 수)개를 맨 위에 놓은 이상적 이진 순위다.
후보 부족으로 추천이 k개보다 짧으면 빈 자리도 Precision 분모 k에
포함한다. 반환 키 precision/recall/ndcg/hits, float 값. 평균은 호출자가
사용자별 결과를 모아서 계산한다. 입력을 수정하지 않는다.

## evaluate_rankings

[src/luna_recsys/ranking.py · definition](https://github.com/lunalab-ai/recommender/blob/2026-fall-w02b/src/luna_recsys/ranking.py#L186)

```python
evaluate_rankings(model: FourMethodRecommender, test: pd.DataFrame, *, k: int=10, relevance_threshold: float=4)
```

학습 완료 model을 고정하고 동일 test/후보/사용자에서 네 방법을 평가한다.

test: user_id/movie_id/rating의 관측 표. 중복 쌍과 train 쌍 겹침을 거부.
k: 공통 목록 길이(10), relevance_threshold: 평가 관련성 하한(4).
관련 test 영화가 하나도 없는 사용자는 네 방법 모두에서 제외하며
metadata에 수를 기록한다. test는 모델 재학습이나 후보 제거에 사용하지
않는다. 평균은 평가 사용자별 단순평균(macro), coverage는 추천된 고유
영화 수/전체 카탈로그 수다. 대체율은 반환된 상위 목록 항목 기준이다.

Returns
-------
summary : DataFrame
    method, precision, recall, ndcg, catalog_coverage, fallback_fraction,
    zero_genre_fraction, users. 점수는 반올림하지 않는다.
per_user : DataFrame
    user_id/method와 precision/recall/ndcg/hits/relevant_count.
    사용자 식별자와 평가 결과를 담으므로 저작 검증용으로만 저장한다.
metadata : dict
    test_users/evaluated_users/excluded_no_positive/catalog_size/k 및
    relevance_threshold. 관측되지 않은 영화는 비선호가 아니라 관련성
    미확인이다. 이 holdout 결과는 온라인 클릭/만족도의 보증이 아니다.

## validate_ratings

[src/luna_recsys/rating_models.py · definition](https://github.com/lunalab-ai/recommender/blob/2026-fall-w02b/src/luna_recsys/rating_models.py#L9)

```python
validate_ratings(ratings: pd.DataFrame)
```

평점 학습 표의 필수 열과 값 범위를 검사한다.

ratings: pandas DataFrame. user_id/movie_id/rating 열, 1행 이상,
세 열에 결측 없음, rating은 유한한 숫자 1–5여야 한다. 추가 열은 허용.
반환 None; 성공해도 표를 바꾸거나 출력하지 않는다. 위반은 ValueError.
이 함수만으로 ID 참조 관계·중복 관측을 검사하지는 않는다. 원본 파일은
validate_movielens_100k, 순위 비교는 FourMethodRecommender.fit도 확인한다.

## MeanRatingPredictor

[src/luna_recsys/rating_models.py · definition](https://github.com/lunalab-ai/recommender/blob/2026-fall-w02b/src/luna_recsys/rating_models.py#L29)

```python
MeanRatingPredictor
```

관측 평점을 이용해 범주별 상수 회귀모형을 학습하는 클래스.

Parameters
----------
mode : str, default 'movie'
    'global': 모든 행에 같은 학습 전체 평균. 'movie': 영화별 학습 평균.
    'group': 사용자의 집단과 영화가 같은 학습 행의 평균.
group_col : str, default 'sex'
    mode='group'일 때 users에서 읽는 집단 속성 열. 'occupation'도 가능.
min_group_ratings : int, default 1
    집단×영화 평균의 최소 학습 표본 수. 부족하면 영화 평균으로 대체.

학습은 L(theta)=sum((rating-theta)**2)를 최소화하는 theta=평균을
계산하는 것이다. 정답 rating을 쓰는 지도학습이며 경사하강법은 불필요하다.
생성자는 설정만 보관하고 fit이 통계 모수를 추정한다.

Attributes after fit
--------------------
global_mean_ : float, 학습 전체 평균.
movie_means_ : Series, movie_id가 인덱스인 영화별 평균.
user_groups_ : Series, group 모드의 user_id→집단 매핑.
group_means_ : Series, (집단, movie_id) 다중 인덱스 평균; 최소 표본 충족만.
밑줄 접미사는 학습 후 생기는 속성이라는 수업 코드의 명명 관례다.

Example
-------
model = MeanRatingPredictor('group', group_col='occupation', min_group_ratings=2)
model.fit(train, users)
y_hat = model.predict(test[['user_id', 'movie_id']])

집단 평균→영화 평균→전체 평균 순으로 대체하므로 새 집단/영화도 처리한다.
원본 표를 나중에 수정해도 이미 학습한 통계는 바뀌지 않는다. 재학습하려면
fit을 다시 호출한다. predict는 test 정답을 계산에 쓰지 않는다.

## MeanRatingPredictor.__init__

[src/luna_recsys/rating_models.py · definition](https://github.com/lunalab-ai/recommender/blob/2026-fall-w02b/src/luna_recsys/rating_models.py#L64)

```python
__init__(self, mode: str='movie', *, group_col: str='sex', min_group_ratings: int=1)
```

mode/group_col/min_group_ratings 설정을 검사해 보관한다.

mode는 global/movie/group, group_col은 집단 열 이름(기본sex),
min_group_ratings는 1 이상 정수(기본1)다. 데이터 인자를 받지 않고
평균도 계산하지 않는다. 반환값 None; 잘못된 설정은 ValueError.
객체를 만드는 것과 학습하는 것을 분리해 같은 설정을 재사용한다.

## MeanRatingPredictor.fit

[src/luna_recsys/rating_models.py · definition](https://github.com/lunalab-ai/recommender/blob/2026-fall-w02b/src/luna_recsys/rating_models.py#L77)

```python
fit(self, ratings: pd.DataFrame, users: pd.DataFrame | None=None)
```

ratings의 정답 평점으로 평균 모수를 추정하고 self를 반환한다.

ratings: 학습 N행 DataFrame(user_id/movie_id/rating 필수). test를 넣지
않는다. users: 기본 None; group 모드에서는 user_id와 group_col을 가진
사용자 메타데이터가 필수다. ID는 유일하고 결측이 없어야 하며 ratings의
모든 사용자에 집단 정보가 있어야 한다. 다른 모드에서는 users를 무시한다.

1. 전체 rating 평균을 global_mean_에 저장한다.
2. movie_id별 rating 평균을 movie_means_에 저장한다.
3. group 모드는 user_id로 속성을 연결해 (집단,movie_id)별 평균/개수를
   구하고 min_group_ratings 이상인 평균만 group_means_에 저장한다.
실제 코드는 입력 검사를 먼저 끝낸 뒤 이 통계를 갱신한다. 입력 표는
수정하지 않는다. 반환 self이므로 MeanRatingPredictor().fit(train)처럼
연결 호출할 수 있다. 잘못된 표/집단 매핑에는 ValueError가 발생한다.

## MeanRatingPredictor.predict_details

[src/luna_recsys/rating_models.py · definition](https://github.com/lunalab-ai/recommender/blob/2026-fall-w02b/src/luna_recsys/rating_models.py#L115)

```python
predict_details(self, pairs: pd.DataFrame)
```

pairs의 각 사용자–영화 쌍에 예측값과 실제 사용 기준을 반환한다.

pairs: DataFrame, user_id/movie_id 필수. rating 열은 필요 없으며 있어도
사용하지 않는다. 반환은 같은 행 수·순서·인덱스의 DataFrame이며
prediction(float,1–5)과 level('group'/'movie'/'global') 열을 갖는다.
학습이 된 경우 빈 입력도 빈 출력을 반환한다. 원자료나 학습 상태는 수정하지
않는다. fit 전 또는 필수 열 누락은 ValueError.

먼저 전체 평균으로 배열을 채운다. 영화 평균이 있으면 덮어쓰고, group
모드에서 최소 표본을 충족하는 집단×영화 평균이 있으면 다시 덮어쓴다.
이 순서가 집단→영화→전체 대체 규칙을 구현한다. 평균은 반올림하지 않는다.
level을 세면 집단 모형이 실제로 집단 평균을 쓴 비율을 알 수 있다.

## MeanRatingPredictor.predict

[src/luna_recsys/rating_models.py · definition](https://github.com/lunalab-ai/recommender/blob/2026-fall-w02b/src/luna_recsys/rating_models.py#L147)

```python
predict(self, pairs: pd.DataFrame)
```

pairs(user_id/movie_id 표)의 평점 예측 Series를 반환한다.

predict_details(pairs)의 prediction 열만 꺼낸 편의 메소드다. 출력 길이는
입력 행 수이고 순서와 중복 인덱스까지 보존한다. 학습 후 여러 번 호출할 수
있고 상태를 갱신하지 않는다. test 정답과의 RMSE/MAE는 호출자가 별도로
계산한다. fit 전/필수 열 누락은 predict_details와 같은 ValueError.
