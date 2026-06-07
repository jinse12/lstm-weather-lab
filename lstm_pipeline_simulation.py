"""LSTM 발전량 예측기 블럭도 신호전송 시뮬레이션.

각 블럭을 순서대로 거치며 실제 DB 데이터를 사용해 신호(데이터)가
어떻게 변환·전달되는지 단계별로 출력한다.

[기상 API]------> [weather_hourly] --+
                                      +--> [join] --> [LSTM 입력 시퀀스] --> [LSTM 모델] --> [예측 power_kw]
[RP2040 Modbus]-> [power_realtime] -> [power_hourly] --+
"""
from __future__ import annotations

import numpy as np

from ml_shared import (
    FEATURES,
    SEQ_LEN,
    TARGET,
    inverse_power_kw,
    load_joined,
    make_sequences,
    require_rows,
)


def block(title: str) -> None:
    print()
    print(f"===== [BLOCK] {title} =====")


def main():
    # 신호 1: 기상 API + RP2040 Modbus -> weather_hourly / power_realtime(power_hourly)
    # -> join 으로 합쳐진 8-feature 시계열 (DB에서 적재된 실제 신호)
    block("1. 데이터 수집 (기상 API + RP2040 Modbus) -> DB join")
    df = load_joined()
    require_rows(df, min_rows=SEQ_LEN + 50)
    print(f"[신호] join 결과 행 수= {len(df)}")
    print(f"[신호] 기간 = {df['obs_time'].min()} ~ {df['obs_time'].max()}")
    print(f"[신호] feature {len(FEATURES)}개 = {FEATURES}")
    print("[신호] 최신 1행 샘플:")
    print(df[["obs_time", *FEATURES]].tail(1).to_string(index=False))

    # 신호 2: 8-feature 시계열 -> 정규화(스케일링) -> 35시간 윈도우 시퀀스
    block("2. 전처리 (MinMax 정규화 -> 과거 35시간 시퀀스 생성)")
    X, y_scaled, scaler = make_sequences(df, SEQ_LEN)
    print(f"[신호] 정규화 후 시퀀스 텐서 shape = {X.shape}  (표본 수, {SEQ_LEN}시간, {len(FEATURES)}feature)")
    print(f"[신호] 정답(y, 정규화) shape = {y_scaled.shape}")
    sample_seq = X[-1]
    print(f"[신호] 마지막 시퀀스의 첫 시간 (정규화값) = {np.round(sample_seq[0], 4)}")
    print(f"[신호] 마지막 시퀀스의 끝 시간 (정규화값) = {np.round(sample_seq[-1], 4)}")

    # 신호 3: 시퀀스 -> 학습된 LSTM 모델 -> 정규화된 예측값
    block("3. LSTM 모델 추론 (과거 35시간x8feature -> 다음 시각 power_kw 예측)")
    try:
        from tensorflow.keras.layers import LSTM, Dense
        from tensorflow.keras.models import Sequential

        model = Sequential([LSTM(64, input_shape=(SEQ_LEN, len(FEATURES))), Dense(1)])
        model.compile(optimizer="adam", loss="mse")
        # 데모 목적: 빠른 추론 확인을 위해 가벼운 학습만 수행 (정식 학습은 lstm_train.py)
        model.fit(X[:-1], y_scaled[:-1], epochs=1, batch_size=32, verbose=0)
        pred_scaled = model.predict(X[-1:], verbose=0).flatten()
        print(f"[신호] 모델 출력 (정규화된 power_kw) = {pred_scaled[0]:.4f}")
    except ImportError:
        print("[안내] tensorflow 미설치 - 모델 추론 단계는 건너뛰고 정답값으로 대체")
        pred_scaled = y_scaled[-1:]

    # 신호 4: 정규화된 예측값 -> 역변환 -> 최종 kW 단위 예측 결과
    block("4. 역변환 (정규화 해제) -> 최종 예측 결과 (kW)")
    pred_kw = inverse_power_kw(scaler, pred_scaled)
    actual_kw = inverse_power_kw(scaler, y_scaled[-1:])
    print(f"[신호] 예측 power_kw = {pred_kw[0]:.4f} kW")
    print(f"[신호] 실제 power_kw = {actual_kw[0]:.4f} kW")
    print(f"[신호] 오차(|예측-실제|) = {abs(pred_kw[0] - actual_kw[0]):.4f} kW")

    block("시뮬레이션 종료 - 신호 흐름: API/Modbus -> DB -> join -> 정규화/시퀀스 -> LSTM -> 역변환 -> 예측")


if __name__ == "__main__":
    main()
