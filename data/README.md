# Data policy and setup

## 공개 저장소에 포함되는 것

- 라이선스가 명확한 작은 sample 또는 synthetic data
- 파일명·출처·라이선스·준비 방법을 기록한 `registry.yml`

## 로컬에서만 사용하는 것

출판사 제공 `data.zip`의 전체 파일은 약 403 MB이므로 Git에 넣지 않습니다.
예상 파일명은 다음과 같습니다.

- `Amazon_embeddings_i.csv`
- `Amazon_embeddings_u.csv`
- `Amazon_ratings.csv`
- `Amazon_reviews.csv`
- `movies_metadata.csv`
- `ratings-20m.csv`
- `u.data`
- `u.item`
- `u.user`

실제 파일은 `data/local/`에 두고, notebook은 환경변수 또는 helper function으로 경로를 찾게 합니다.
Colab 수업에서는 재배포가 허용된 데이터의 공식 download URL이나 별도 수업용 저장 위치를 사용합니다.

## MovieLens 100K · W01B

W01B는 `u.user`, `u.item`, `u.data`를 저장소에 넣지 않습니다. 오래된 MovieLens 100K의
README에는 별도 허가 없는 재배포 제한이 있으므로, notebook이 GroupLens의 공식 archive를
Colab runtime에 직접 내려받거나 학생이 공식 페이지에서 받은 ZIP을 자기 runtime에 업로드합니다.

- 공식 안내: https://grouplens.org/datasets/movielens/100k/
- 공식 archive: https://files.grouplens.org/datasets/movielens/ml-100k.zip
- 공식 README: https://files.grouplens.org/datasets/movielens/ml-100k-README.txt
- archive MD5: `0e33842e24a9c977be4e0107933c0723`

다운로드한 파일은 Colab의 일시적인 `/content` 저장 공간에만 존재합니다. runtime을 재시작하면
사라질 수 있으므로 notebook의 다운로드 셀부터 다시 실행합니다. 과제 제출물에 원본 데이터나
사용자별 인구통계 표를 포함하지 않습니다.

Notebook의 `DATA_MODE`는 `auto`, `upload`, `synthetic`을 지원합니다. `auto`는 공식 다운로드가
실패하면 `data/sample/ml100k_tiny/`로 전환합니다. `upload`는 학생이 직접 받은 `ml-100k.zip`의
공식 MD5와 내부 파일을 검사한 뒤 필요한 세 파일만 풉니다. `synthetic`은 네트워크 없이 작은
가상 데이터로 바로 시작합니다. Synthetic 폴더는 실제 MovieLens 원본의 일부나 변환본이 아니며,
같은 세 파일명·구분자·열 구조를 연습하도록 이 강의를 위해 새로 만든 CC0 데이터입니다.
Clone한 공개 저장소에 이 폴더가 아직 없으면 student notebook이 같은 크기·형식의 CC0 데이터를
`/content/data/ml100k_tiny_generated`에 결정적으로 생성하므로 배포 전 Colab 시험도 진행할 수 있습니다.

인증서 검증을 끄거나 출처 불명 mirror를 사용하지 않습니다. MovieLens 원본 ZIP을 GitHub,
Notion ZIP, SmartClass 또는 교수자 공유 Drive로 재배포하지 않습니다.

## W02A 이후 자동 준비

W02A Colab은 `luna_recsys.prepare_movielens(mode="real")`로 실제 데이터를 준비한다.
캐시 재사용 → 공식 서버 → 고정된 HTTPS 대체 경로 순서이며 파일 업로드가 필요 없다.
모든 경로가 실패하면 원인을 표시하고 중단한다. 이전 API의 `auto` 합성 대안은 유지한다.
현재 모드는 `prepared.mode`와 `prepared.description`으로 확인한다.
이미 가진 사본은 `local_dir`로 지정해 저작·검증에 사용할 수 있다. 그 사본을 다시 다운로드할
필요는 없지만 비공개 로컬 경로 자체가 학생 Colab의 다운로드 주소가 되는 것은 아니다.

W02A의 합성 대안은 패키지에 포함된 `synthetic_movielens(seed=2026)`가 생성하는 80명·18편·
960평점의 독자적인 CC0 데이터다. 실제 MovieLens에서 추출한 표본이 아니며 성능 숫자는 원본과
다르다. W01B의 작은 파일형 대안과 기존 실행 방식은 과거 태그에서 계속 유지한다.

공식 서버의 인증서 만료에 대비해 [Dgraph 벤치마크의 고정된 MovieLens ZIP](https://github.com/dgraph-io/dgraph-benchmarks/blob/0399f1c120208d3e78431eb7bf4ecafdcee15d8d/movielens/conv100k/ml-100k.zip)을 대체 다운로드 경로로 사용한다.
2026-09-08 실제 다운로드에서 공식 MD5 `0e33842e24a9c977be4e0107933c0723`,
SHA-256 `50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229`를 확인했다.
출처는 [GroupLens MovieLens 100K](https://grouplens.org/datasets/movielens/100k/)이며 이용 조건은 원본 README를 따른다.
이 저장소에는 원본 데이터를 추가하지 않으며 TLS 인증서 검증을 끄지 않는다.

## 운영 원칙

- CI와 기본 notebook은 대용량 데이터 없이 실행되어야 합니다.
- 데이터 분할과 전처리는 재현 가능하게 version과 random seed를 기록합니다.
- 데이터의 사용자·평점·리뷰에는 개인정보나 민감정보가 없는지 확인합니다.
