# LSTM 발전량 예측 실습 — 일경험 사전실무교육

기상 데이터(API)와 RP2040 센서(Modbus)에서 수집한 데이터를 MySQL에 적재하고,
**과거 35시간의 흐름으로 다음 1시간의 태양광 발전량(kW)을 예측하는 LSTM 모델**을
직접 구축해 보는 실습 프로젝트입니다.

## 1. 무엇을 배우고 만들었나

```
기상 API (Open-Meteo)        →  weather_hourly   (1시간 1행)
RP2040 센서 (Modbus)         →  power_realtime   →  power_hourly (뷰, 1시간 평균)
                                      ↓ join (같은 시각끼리 결합)
train_baseline.py  →  베이스라인 모델 — 시각 t 의 8개 정보 → t+1 시각의 발전량 예측
lstm_train.py      →  LSTM 모델     — 과거 35시간 × 8개 정보 → 다음 발전량 예측
```

- 1시간 단위 데이터만 사용 (시간별 평균)
- 입력 8 feature: 기온·습도·풍속·일사·강수(기상) + 발전량·패널온도·패널습도(RP2040)
- 베이스라인(HistGradientBoosting)과 LSTM 두 모델을 만들어 성능을 비교

## 2. 실습 파이프라인 (직접 작성한 코드)

| 파일 | 역할 |
|------|------|
| `db.py` | MySQL 연결, 기상 데이터 저장 공통 함수 |
| `weather_openmeteo.py` | Open-Meteo Historical Weather API 호출 |
| `collect_weather.py` | API 원본 JSON 1일치 확인용 저장 |
| `collect_weather_backfill.py` | 30일(약 720시간) 기상 데이터 backfill |
| `collect_rp2040_modbus.py` | RP2040 Modbus(RTU/TCP)에서 실시간 발전량 수집·저장 |
| `ml_shared.py` | DB join, 정규화·시퀀스 생성 등 학습 공통 유틸 |
| `train_baseline.py` | 베이스라인 모델(HistGradientBoosting) 학습·평가 |
| `lstm_train.py` | LSTM 모델 학습·평가, 베이스라인과 MAE 비교(`[COMPARE]`) |

## 3. 학습 결과

| 모델 | 입력 | test MAE (kW) |
|------|------|---------------|
| 베이스라인 (HistGradientBoosting) | 시각 t 의 8 feature → t+1 발전량 | **0.1264** |
| LSTM | 과거 35시간 × 8 feature → 다음 발전량 | **0.2292** |

> 데이터 기간이 약 29일(696시간)로 비교적 짧아 베이스라인이 더 낮은 오차를 보였습니다.
> 데이터가 더 길게 쌓이면 LSTM이 시계열 패턴을 더 잘 학습해 역전되는 경향이 있습니다.

## 4. 신호전송 시뮬레이션 — LSTM 예측기 동작 흐름

LSTM 예측기가 **데이터(신호)를 어떤 순서로 주고받으며 최종 예측까지 이어지는지**를
블럭도와 애니메이션으로 시각화했습니다.

| 파일 | 내용 |
|------|------|
| [`lstm_pipeline_blockdiagram.md`](./lstm_pipeline_blockdiagram.md) | 블럭도 + 신호별 설명 표 |
| `lstm_pipeline_blockdiagram.png` | 블럭도 이미지 |
| `lstm_pipeline_signal_animation.gif` | 신호(점)가 블럭을 따라 이동하는 애니메이션 |
| `lstm_pipeline_simulation.py` | 실제 DB 데이터로 신호 흐름(①~⑦)을 단계별로 출력하는 실행 스크립트 |
| `make_blockdiagram.py` / `make_signal_animation.py` | 위 이미지·GIF를 생성하는 스크립트 |

**신호 흐름 요약**

```
① 기상 API → weather_hourly
② RP2040 센서 → power_realtime
③ power_realtime → power_hourly (1시간 평균)
④ weather_hourly + power_hourly → JOIN (8 feature 결합)
⑤ JOIN 결과 → 정규화 + 과거 35시간 시퀀스 생성
⑥ 시퀀스 → LSTM 모델 추론 (정규화된 예측값)
⑦ 예측값 → 역변환 → 최종 kW 단위 예측 결과
```

## 5. 실행 방법 (요약)

```powershell
# 1) 패키지 설치
uv sync

# 2) .env 작성 (DB·기상 소스·Modbus 설정) — 저장소에는 포함되지 않음 (.gitignore)

# 3) 데이터 수집
uv run python collect_weather_backfill.py 30
uv run python collect_rp2040_modbus.py

# 4) 모델 학습·비교
uv run python train_baseline.py
uv run python lstm_train.py

# 5) 신호전송 시뮬레이션
uv run python lstm_pipeline_simulation.py
uv run python make_blockdiagram.py
uv run python make_signal_animation.py
```

전체 진행 절차는 [`실습매뉴얼_공공데이터_RP2040_MySQL.md`](./실습매뉴얼_공공데이터_RP2040_MySQL.md) 를 참고하세요.
