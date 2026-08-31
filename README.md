# 딥러닝응용I(추천시스템)

동덕여자대학교 데이터사이언스전공 · 2026학년도 2학기  
담당교수: 유원상  
수업: 화·목 10:30-11:45

이 저장소는 추천시스템 이론, Python/Google Colab 실습, 미니 웹 애플리케이션,
과제와 참고자료를 제공합니다.

## 학습 범위

인기 기반 추천과 내용 기반 추천부터 협업 필터링, Matrix Factorization,
Factorization Machines, 딥러닝 추천, AutoEncoder, Transformer, LLM,
하이브리드 추천과 실제 구축 이슈까지 단계적으로 학습합니다.

## 저장소 구성

```text
course/notion/       Notion 강의 페이지 원본 Markdown과 이미지
notebooks/student/   학생용 Google Colab notebook
assignments/student/ 공개 과제 안내
src/luna_recsys/     수업 전체에서 재사용하는 Python 패키지
data/                데이터 사용 안내, registry, 소규모 공개 샘플
```

## Colab에서 패키지 설치

수업에서 지정한 검증 tag를 사용합니다.

```python
COURSE_VERSION = "2026-fall-w01"  # 수업 공지의 버전으로 변경
!pip -q install "git+https://github.com/lunalab-ai/recommender.git@{COURSE_VERSION}"
```

개발 중인 최신 버전을 시험할 때만 `main`을 사용합니다.

## 강의자료

- Notion 공개 페이지: [W01A · 오리엔테이션과 추천시스템의 세계](https://app.notion.com/p/w01a-orientation-and-recommenders-3cd7fd00109f818db55fc0a452d428f3?source=copy_link)
- 강의계획서: `course/course.yml`의 주차별 개요 참조
- 학생용 notebook: `notebooks/student/`
- 과제: `assignments/student/`

## 데이터

대용량 원본 데이터나 재배포 권한이 명확하지 않은 출판사 자료는 저장소에 포함하지 않습니다.
필요한 파일명, 출처, 라이선스, 준비 방법은 `data/registry.yml`과 `data/README.md`에 기록합니다.

## 평가

중간고사 30%, 기말고사 40%, 과제 10%, 출석 20%입니다.
실습 과제는 원칙적으로 통과 10점 또는 탈락 0점으로 평가합니다.

## 주의

학생용 notebook의 코드를 순서대로 실행하는 것만으로 학습을 완료한 것으로 보지 않습니다.
각 notebook의 빈칸 채우기, 디버깅, 해석, 퀴즈와 확장 활동을 직접 수행하십시오.
