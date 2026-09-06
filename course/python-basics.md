# 추천시스템 수업을 위한 Python 복습 길잡이

이 문서는 수업을 돕기 위해 추가한 복습 안내입니다. 별도 제출 과제가 아닙니다. 아래 예제를 Colab의 새 코드 셀에 직접 입력하고, 숫자나 조건을 바꿔 결과를 확인하세요.

## 1. 먼저 셀을 실행해 보기

[Google Colab 시작 notebook](https://colab.research.google.com/notebooks/intro.ipynb)을 열어 설명 셀과 코드 셀을 구분합니다. 수업 notebook은 README의 Colab 링크에서 열고 Drive에 사본을 저장하세요.

```python
print("추천시스템 실습 시작!")
```

`Shift+Enter`로 실행합니다. Notebook은 코드를 담은 문서이고, runtime은 코드가 실행되는 공간입니다. Runtime을 다시 시작하면 앞서 만든 변수를 다시 계산해야 합니다.

## 2. 변수와 list: 작은 평점 목록 만들기

```python
ratings = [5, 3, 4]
mean_rating = sum(ratings) / len(ratings)
print(mean_rating)
```

`ratings`는 값 여러 개를 담는 list입니다. `sum`은 합계, `len`은 원소 수를 구합니다. 평점 하나를 바꾸면 평균이 어떻게 달라지는지 먼저 예측해 보세요. 빈 목록의 평균을 계산할 수 없는 이유도 생각해 봅니다.

읽을 자료: [Python 공식 자습서 3장](https://docs.python.org/ko/3/tutorial/introduction.html).

## 3. dictionary·조건문·반복문: 조건에 맞는 영화 찾기

```python
movies = [
    {"title": "Movie A", "rating": 4.5},
    {"title": "Movie B", "rating": 3.0},
]
for movie in movies:
    if movie["rating"] >= 4.0:
        print(movie["title"])
```

Dictionary는 이름(key)과 값(value)을 연결합니다. `for`는 영화를 하나씩 꺼내고, `if`는 조건을 만족할 때만 출력합니다. 들여쓰기도 코드의 일부입니다. 기준을 `3.0`으로 바꾸면 무엇이 출력될까요?

읽을 자료: [제어 흐름](https://docs.python.org/ko/3/tutorial/controlflow.html), [자료 구조](https://docs.python.org/ko/3/tutorial/datastructures.html).

## 4. 함수: 입력과 반환값 구분하기

```python
def is_candidate(rating_count, minimum_count):
    return rating_count >= minimum_count

print(is_candidate(20, 10))
```

`def`로 함수를 정의하고, 괄호 안의 인자로 값을 전달합니다. `return`은 계산 결과를 호출한 곳으로 돌려줍니다. `print`는 화면에 표시하는 역할입니다. 평점 수가 적은 영화를 후보에서 제외하는 조건으로 연결해 보세요.

읽을 자료: [Python 공식 자습서 4장](https://docs.python.org/ko/3/tutorial/controlflow.html)의 함수 정의.

## 5. DataFrame: 표 읽고 선택하기

```python
import pandas as pd

ratings_table = pd.DataFrame({
    "movie_id": [1, 1, 2],
    "rating": [5, 3, 4],
})
print(ratings_table.head())
print(ratings_table.groupby("movie_id")["rating"].mean())
```

DataFrame은 행과 열이 있는 표입니다. `groupby`는 같은 영화의 행을 묶고, `mean`은 묶음별 평균을 계산합니다. 먼저 손으로 영화별 평균을 구한 뒤 코드와 비교하세요.

읽을 자료: [pandas 입문](https://pandas.pydata.org/docs/getting_started/intro_tutorials/index.html). 배열의 `shape`나 인덱싱이 낯설면 [NumPy 입문](https://numpy.org/doc/stable/user/absolute_beginners.html)을 이어서 읽습니다.

## 막혔을 때 확인할 것

| 증상 | 먼저 확인하기 |
|---|---|
| `NameError` | 변수를 만든 앞 셀을 실행했는가? 이름이 같은가? |
| `IndentationError` | `for`, `if`, `def` 안의 들여쓰기가 맞는가? |
| `KeyError` | dictionary의 key 또는 DataFrame의 열 이름이 맞는가? |
| `FileNotFoundError` | 데이터 준비 셀을 실행했고 파일 경로가 맞는가? |
| 실행은 되지만 결과가 다름 | 입력 데이터와 조건, 셀 실행 순서가 같은가? |

오류 메시지의 마지막 줄부터 읽고, 해당 셀의 입력과 변수 이름을 확인하세요. [Python 에러와 예외](https://docs.python.org/ko/3/tutorial/errors.html)에 기본 설명이 있습니다.

프로그래밍 자체가 처음이라 더 많은 연습이 필요하면 [Helsinki Python Programming MOOC 2026](https://programming-26.mooc.fi/)의 Introduction Part 1부터 시작하세요(영어). 공식 Python 자습서는 프로그래밍 경험이 있는 독자를 가정하므로 문법 복습과 검색용으로 함께 활용하면 좋습니다.

[강의 허브로 돌아가기](../README.md) · [참고자료 전체](references.md)
