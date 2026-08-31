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

## 원칙

- CI와 기본 notebook은 대용량 데이터 없이 실행되어야 합니다.
- 데이터 분할과 전처리는 재현 가능하게 version과 random seed를 기록합니다.
- 데이터의 사용자·평점·리뷰에는 개인정보나 민감정보가 없는지 확인합니다.
