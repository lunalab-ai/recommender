# W10A · 재사용 API 안내

이 모듈들은 수업의 계산·전처리·학습·앱에서 함께 사용합니다. `import`만으로 TensorFlow 학습이나 데이터 다운로드를 시작하지 않습니다. 실제 정의 위치를 검토할 수 있도록 고정 수업 ref와 줄 번호를 연결합니다.

| 함수 | 입력과 기본값 | 반환·행 순서 | 상태와 부작용 |
|---|---|---|---|
| `prepare_neural_data` | PreparedMovieLens, seed20261103, max_rows=None | NeuralData, train/validation/test 프레임·훈련 통계 | 키 결합과 메모리 복사, 네트워크 없음 |
| `fixed_forward` | user A/B, item P/Q, activation relu, occupation None | p/q,z,kernel,pre,hidden,중심화/평점 사전 | 새 합성 배열, 학습/파일쓰기 없음 |
| `build_mf` | 표 행수M/N, k8, reg.0002, seed20261103 | Keras Model, 중심화 출력(B,1) | TF import·전역seed·모수초기화 |
| `build_mlp` | M/N,k8,hidden16/8,직업표행수None,age=False | Keras Model, 중심화 출력(B,1) | 직업Embedding3,선택나이Dense3,전역seed |
| `fit_neural` | NeuralData,kind mf,최대8epoch,batch128,학습률.05 | model/history/best_epoch/metrics/predictions/counts/setting | 실제SGD학습,검증선택,시험최종평가 |
| `predict_ratings` | model,data,관측프레임,kind mf | 원래 행순서의예측(B,),warm mask(B,) | 평균복원,미학습ID평균대체,학습없음 |
| `rating_metrics` | 유한한동일길이1차원y/prediction | mse,rmse,n 사전 | 잘못된shape는오류,방송허용안함 |

표 행 수는 예약 ID0을 포함합니다. `data.dimensions`로 세 표의 행 수를 얻습니다. 실제 데이터 준비 함수의 예약0과 작은 고정 표에서의0기반 행 번호를 구별합니다. `fit_neural`의 history loss에는 정규화 벌점이 포함될 수 있으며 `rmse`는 순수 오차 지표입니다.

```python
prepared = prepare_movielens(mode='synthetic')
data = prepare_neural_data(prepared)
print({name: len(frame) for name, frame in data.frames.items()})
model = build_mf(*data.dimensions[:2], k=8)
print(model.count_params())
```

기본 합성 자료는672/144/144행으로 분할됩니다. 표 행 수가81/19라면 MF 모수는900개입니다. 실제 결과는 화면의 데이터 모드·크기와 함께 읽습니다. 별도 노트북에 알고리즘을 복사하지 않고 이 함수들을 import합니다.

## 실제 정의 위치

- [fixed_forward](https://github.com/lunalab-ai/recommender/blob/2026-fall-w10a/src/luna_recsys/neural_recommenders.py#L16)
- [prepare_neural_data](https://github.com/lunalab-ai/recommender/blob/2026-fall-w10a/src/luna_recsys/neural_recommenders.py#L97)
- [build_mf](https://github.com/lunalab-ai/recommender/blob/2026-fall-w10a/src/luna_recsys/neural_recommenders.py#L147)
- [build_mlp](https://github.com/lunalab-ai/recommender/blob/2026-fall-w10a/src/luna_recsys/neural_recommenders.py#L171)
- [fit_neural](https://github.com/lunalab-ai/recommender/blob/2026-fall-w10a/src/luna_recsys/neural_recommenders.py#L242)
- [predict_ratings](https://github.com/lunalab-ai/recommender/blob/2026-fall-w10a/src/luna_recsys/neural_recommenders.py#L225)
- [rating_metrics](https://github.com/lunalab-ai/recommender/blob/2026-fall-w10a/src/luna_recsys/neural_recommenders.py#L214)


## 앱과 데이터 준비

`build_neural_explainer()`는 기존FM과새MF/MLP탭의Blocks를 반환합니다. `forward_view(user,item,model,activation,occupation)`는 요약·입력표·층별표·그림을 반환합니다. `export_forward`는 다운로드용 합성CSV를 임시폴더에 쓰고경로를 반환합니다. Gradio는그림캐시를쓸수있고외부공유는launch호출로시작합니다. 실제데이터를학습한성능과이앱의고정합성계산을구별합니다.

`prepare_movielens(cache_dir='data/local',mode='auto',timeout=20)`의mode='real'은공식다운로드실패를오류로전달하고mode='synthetic'은독자가상자료를네트워크없이준비합니다. auto의대체모드도반환description에표시합니다. 정확한데이터실험은모드를명시하세요.
