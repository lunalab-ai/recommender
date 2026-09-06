# {{TITLE}}

{{IDENTITY}}

![사용자와 아이템의 행동 데이터에서 추천 모델, 추천 결과, 평가와 개선으로 이어지는 수업의 흐름](course/assets/course-overview.svg)

**사용자에게 무엇을 추천할지, 왜 그렇게 추천했는지, 좋은 추천인지 함께 알아봅니다.**

인기 기반 추천에서 시작해 협업 필터링, 행렬분해, 딥러닝 추천으로 발전시키는 수업입니다. Python과 Google Colab으로 작은 데이터를 직접 다루고, 추천 결과를 비교·평가하며, 한 학기 동안 작은 추천 웹 애플리케이션을 단계적으로 발전시킵니다.

## 수업 시작하기

1. 아래 표에서 오늘 수업의 **MD 또는 PDF 강의노트**를 엽니다. Notion 링크로도 읽을 수 있습니다.
2. **Colab**을 열어 Drive에 사본을 저장하고, 설명을 읽으며 위에서 아래로 실습합니다.
3. 점검 퀴즈를 먼저 풀고 **퀴즈 답안**으로 확인합니다. 코드 실행 뒤에는 추천 결과가 나온 이유를 설명해 봅니다.

[전체 PDF·답안](course/handouts/README.md) · [Python 복습 길잡이](course/python-basics.md) · [모든 참고자료](course/references.md) · [과제](assignments/student/README.md) · [데이터 안내](data/README.md)

## 차시별 강의자료

자료가 추가되면 이 표도 함께 갱신됩니다. `—`는 아직 게시되지 않았거나 별도 자료가 없는 항목입니다. 일정 변경과 과제 제출·시험 안내는 스마트클래스 공지를 확인하세요.

{{SESSIONS}}

강의노트와 답안은 최신 자료입니다. 실습의 ipynb·Colab 링크는 같은 버전을 열며, `수업본`은 공지된 검증 tag, `최신본`은 main을 사용합니다. PDF는 MD와 같은 내용이며 화면 배치와 페이지 나눔은 다를 수 있습니다.

## 이 수업에서 배우는 것

{{OUTCOMES}}

## 16주 커리큘럼

[강의계획](course/course.yml)의 주차별 내용을 따릅니다. 차시별 실제 날짜와 자료는 위 표에서 확인하세요.

{{CURRICULUM}}

## Python 기초가 약하다면

한 번에 모든 문법을 익히려고 하기보다 **셀 실행 → 변수와 list → 조건문과 반복문 → 함수 → DataFrame** 순서로 복습하세요. [Python 복습 길잡이](course/python-basics.md)에 각 단계의 예제와 확인할 질문을 모았습니다.

공식 Python 자습서는 기본적인 프로그래밍 경험을 가정합니다. 프로그래밍 자체가 처음이면 길잡이의 작은 예제와 Helsinki 입문 자료부터 시작하고, 공식 문서는 문법을 찾아보는 용도로 활용하세요.

{{PYTHON_RESOURCES}}

## 교재와 참고자료

{{TEXTBOOK}}

아래 자료는 수업을 위한 보충 읽기입니다. 각 링크의 설명은 학습을 돕기 위해 추가했으며, 외부 예제의 데이터와 실행 환경은 수업 notebook과 다를 수 있습니다.

{{OTHER_RESOURCES}}

**[차시별 출처와 본문 참고 링크 전체 보기](course/references.md)** — 매 수업의 참고자료를 한곳에 모아 확인할 수 있습니다.

## 과제·데이터·수업 코드

- [학생용 과제 안내](assignments/student/README.md): 공개 과제 자료와 실습 제출 안내
- [학생용 notebook 모음](notebooks/student/): 실습 파일을 내려받아 보관할 때
- [수업 Python 패키지](src/luna_recsys/): 여러 차시에서 재사용하는 추천·데이터 처리 코드
- [데이터 사용 안내](data/README.md) · [데이터 목록과 출처](data/registry.yml) · [소규모 샘플](data/sample/)
- [Notion 강의자료 목차](course/notion/index.md) · [PDF·퀴즈 답안 모음](course/handouts/README.md)

MovieLens 100K 원본은 저장소에 포함하지 않습니다. 수업 notebook의 데이터 준비 절차에 따라 공식 배포처에서 내려받거나 제공된 synthetic sample을 사용하세요.

## 수업 운영과 학습 방법

{{ASSESSMENT}}

수업 전에는 핵심 질문을 읽고, 수업 중에는 조건을 바꾸어 결과를 비교하고, 수업 후에는 퀴즈와 Take-home message로 설명할 수 있는지 점검하세요. 질문할 때는 어느 차시·어느 셀인지와 오류 메시지를 함께 준비하면 좋습니다. 제출·상담 등 운영 세부 사항은 스마트클래스와 수업 시간 안내를 따릅니다.
