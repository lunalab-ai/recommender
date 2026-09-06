# W01B · Colab 환경 구축과 첫 추천 앱 · 퀴즈 답안

> 딥러닝응용I(추천시스템) · 동덕여자대학교 · 유원상 교수 · 2026-2

먼저 강의자료의 점검 퀴즈를 풀고 확인하세요. 서술형 문항은 같은 의미의 다른 표현도 가능합니다.

이 답안은 해당 강의의 설명을 바탕으로 정리한 학습용 해설입니다.

## 1. GitHub repository와 Colab runtime은 각각 무엇을 보관하는가?

GitHub는 공개 코드·문서와 버전 기록, Colab runtime은 실행 중 변수·package·임시 파일을 보관한다.

## 2. runtime을 다시 시작한 뒤 `ratings`가 사라지는 이유는 무엇인가?

`ratings`는 runtime memory에 있으므로 runtime 초기화와 함께 사라진다.

## 3. MovieLens 100K 원본을 수업 GitHub에 넣지 않는 이유는 무엇인가?

MovieLens 100K의 구버전 사용 조건에 별도 허가 없는 재배포 제한이 있기 때문이다.

## 4. `u.user`, `u.item`, `u.data`는 각각 무엇을 나타내는가?

`u.user`는 사용자, `u.item`은 영화·장르, `u.data`는 사용자-영화 평점 행동이다.

## 5. `u.data`의 `sep="\t"`에서 `\t`는 무엇인가?

tab 문자다.

## 6. 평균 평점만 정렬할 때 평점 수가 매우 적은 영화가 문제가 될 수 있는 이유는 무엇인가?

한두 개의 높은 평점도 5.0 평균을 만들 수 있어 관찰량이 많은 영화와 같은 근거로 비교하기 어렵다.

## 7. `run_button.click`에서 callback, inputs, outputs는 어떤 순서로 연결되는가?

버튼 event가 inputs의 값을 순서대로 callback에 전달하고 callback 반환값을 outputs에 표시한다.

## 8. 오늘 앱이 개인화 추천이 아닌 이유는 무엇인가?

사용자 ID나 사용자별 과거 행동을 사용하지 않고 모두에게 같은 집계 순위를 적용하기 때문이다.
