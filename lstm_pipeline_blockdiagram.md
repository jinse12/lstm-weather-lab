# LSTM 발전량 예측기 — 블럭도 & 신호전송 시뮬레이션

## 1. 블럭도 (Block Diagram)

```
 ┌───────────────────┐        ┌────────────────────┐
 │   기상 API         │        │   RP2040 (Modbus)   │
 │ (Open-Meteo /     │        │  power_kw, temp,    │
 │  공공데이터 ASOS) │        │  humidity 측정      │
 └─────────┬─────────┘        └──────────┬──────────┘
           │ 신호①: 시간별 기상 관측값      │ 신호②: 1초 주기 실시간 측정값
           ▼                              ▼
 ┌───────────────────┐        ┌────────────────────┐
 │  weather_hourly    │        │   power_realtime    │
 │  (MySQL, 1시간1행) │        │ (MySQL, 1초1행)     │
 └─────────┬─────────┘        └──────────┬──────────┘
           │                              │ 신호③: 시간 단위 평균(VIEW)
           │                              ▼
           │                    ┌────────────────────┐
           │                    │   power_hourly      │
           │                    │ (VIEW, 1시간1행)    │
           │                    └──────────┬──────────┘
           │ 신호④: obs_time = hour_time   │
           └──────────────┬───────────────┘
                          ▼
              ┌────────────────────────┐
              │   JOIN (ml_shared.py)   │
              │  8 feature 시계열 생성  │
              │ temperature, humidity,  │
              │ wind_speed,             │
              │ solar_radiation,        │
              │ precipitation, power_kw,│
              │ panel_temp,             │
              │ panel_humidity          │
              └───────────┬────────────┘
                          │ 신호⑤: 8feature × N시간 행렬
                          ▼
              ┌────────────────────────┐
              │  전처리 (MinMaxScaler)  │
              │  과거 35시간 윈도우     │
              │  시퀀스 생성            │
              │  shape=(N, 35, 8)       │
              └───────────┬────────────┘
                          │ 신호⑥: 정규화된 시퀀스 텐서
                          ▼
              ┌────────────────────────┐
              │      LSTM 모델          │
              │  LSTM(64) → Dense(1)    │
              │  과거 35h×8f            │
              │   → 다음 시각 power_kw  │
              └───────────┬────────────┘
                          │ 신호⑦: 정규화된 예측값 (0~1)
                          ▼
              ┌────────────────────────┐
              │   역변환(inverse scale) │
              │   최종 예측 결과 (kW)   │
              └────────────────────────┘
```

## 2. 신호별 설명

| 신호 | 구간 | 내용 | 코드 위치 |
|------|------|------|-----------|
| ① | 기상 API → `weather_hourly` | 1시간 단위 기온·습도·풍속·일사·강수 | `weather_openmeteo.py`, `collect_weather_backfill.py` |
| ② | RP2040 → `power_realtime` | 1초 주기 발전량(kW)·패널 온습도 | `collect_rp2040_modbus.py` |
| ③ | `power_realtime` → `power_hourly` | 초단위 데이터를 1시간 단위로 평균 (VIEW, `AVG`) | MySQL `CREATE VIEW power_hourly` |
| ④ | `weather_hourly` + `power_hourly` → JOIN | 같은 시각(`obs_time` = `hour_time`)끼리 결합, 8 feature 행 생성 | `ml_shared.load_joined` |
| ⑤ | JOIN 결과 → 시퀀스 생성 | 8 feature × N시간 행렬을 0~1로 정규화 후 과거 35시간 슬라이딩 윈도우 적용 | `ml_shared.make_sequences` |
| ⑥ | 시퀀스 → LSTM 입력 | `shape=(샘플 수, 35, 8)` 텐서를 LSTM(64) 레이어에 입력 | `lstm_train.py` |
| ⑦ | LSTM 출력 → 최종 예측 | 정규화된 출력(0~1)을 `inverse_power_kw`로 kW 단위로 환원 | `ml_shared.inverse_power_kw` |

## 3. 신호전송 시뮬레이션 실행

위 블럭도의 신호 흐름(①~⑦)을 실제 DB 데이터로 단계별로 재현하는 스크립트는
[`lstm_pipeline_simulation.py`](./lstm_pipeline_simulation.py) 이다.

```powershell
uv run python lstm_pipeline_simulation.py
```

실행하면 각 블럭을 통과할 때마다 신호(데이터)의 모양과 값이 다음과 같이 단계별로 출력된다.

```
===== [BLOCK] 1. 데이터 수집 (기상 API + RP2040 Modbus) -> DB join =====
[신호] join 결과 행 수= ...
[신호] feature 8개 = [...]

===== [BLOCK] 2. 전처리 (MinMax 정규화 -> 과거 35시간 시퀀스 생성) =====
[신호] 정규화 후 시퀀스 텐서 shape = (N, 35, 8)

===== [BLOCK] 3. LSTM 모델 추론 (과거 35시간x8feature -> 다음 시각 power_kw 예측) =====
[신호] 모델 출력 (정규화된 power_kw) = 0.xxxx

===== [BLOCK] 4. 역변환 (정규화 해제) -> 최종 예측 결과 (kW) =====
[신호] 예측 power_kw = x.xxxx kW
[신호] 실제 power_kw = x.xxxx kW
```

즉, **물리 신호(기상·발전량 측정값) → DB 적재 → 시계열 결합 → 정규화/시퀀스화 → LSTM 추론 → 역변환된 예측치**까지
한 사이클의 신호전송을 코드로 시뮬레이션한다.
