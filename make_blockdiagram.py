"""LSTM 발전량 예측기 블럭도를 이미지(PNG)로 생성한다."""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_PATH = "lstm_pipeline_blockdiagram.png"

# Windows 한글 폰트(맑은 고딕) 적용 - 미설치 환경이면 기본 폰트로 대체
for font_name in ("Malgun Gothic", "맑은 고딕", "NanumGothic"):
    if any(font_name == f.name for f in font_manager.fontManager.ttflist):
        rcParams["font.family"] = font_name
        break
rcParams["axes.unicode_minus"] = False

# (x, y, width, height, label, facecolor)
BOXES = {
    "weather_api": (
        0.5, 9.0, 3.2, 1.1,
        "① 기상 API\nOpen-Meteo / 공공데이터 ASOS\n→ 시간별 기온·습도·풍속·일사·강수",
        "#cfe8ff",
    ),
    "rp2040": (
        6.0, 9.0, 3.2, 1.1,
        "② RP2040 (Modbus 센서)\n1초마다 발전량(kW)·\n패널 온도·습도 측정",
        "#ffe6cc",
    ),
    "weather_hourly": (
        0.5, 7.2, 3.2, 1.0,
        "weather_hourly 테이블\n(MySQL · 1시간에 1행 저장)",
        "#cfe8ff",
    ),
    "power_realtime": (
        6.0, 7.2, 3.2, 1.0,
        "power_realtime 테이블\n(MySQL · 1초마다 1행 저장)",
        "#ffe6cc",
    ),
    "power_hourly": (
        6.0, 5.5, 3.2, 1.0,
        "power_hourly 뷰(VIEW)\n1시간 단위로 평균(AVG) 계산",
        "#ffe6cc",
    ),
    "join": (
        2.9, 3.7, 3.7, 1.2,
        "JOIN 결합 (ml_shared.load_joined)\n같은 시각끼리 묶어 8개 항목으로 정리\n(기온·습도·풍속·일사·강수·발전량·\n패널온도·패널습도)",
        "#d9f2d9",
    ),
    "preprocess": (
        2.9, 2.0, 3.7, 1.1,
        "전처리 (정규화 + 시퀀스 만들기)\n값을 0~1로 맞추고, 과거 35시간씩\n묶어서 입력 데이터 준비",
        "#d9f2d9",
    ),
    "lstm": (
        2.9, 0.3, 3.7, 1.1,
        "LSTM 예측 모델\n과거 35시간의 흐름을 학습해\n다음 1시간 뒤 발전량을 예측",
        "#f6d9ff",
    ),
    "inverse": (
        2.9, -1.4, 3.7, 1.1,
        "결과 변환 (역정규화)\n0~1 사이 예측값을 실제 단위인\nkW(킬로와트)로 환산해 출력",
        "#fff3b0",
    ),
}

ARROWS = [
    ("weather_api", "weather_hourly", "관측값 저장"),
    ("rp2040", "power_realtime", "측정값 저장"),
    ("power_realtime", "power_hourly", "1시간 평균으로 집계"),
    ("weather_hourly", "join", "시간 맞춰 결합"),
    ("power_hourly", "join", "시간 맞춰 결합"),
    ("join", "preprocess", "결합된 표 데이터 전달"),
    ("preprocess", "lstm", "정규화된 입력 시퀀스 전달"),
    ("lstm", "inverse", "예측값(0~1) 전달"),
]


def center_bottom(box):
    x, y, w, h, *_ = box
    return (x + w / 2, y)


def center_top(box):
    x, y, w, h, *_ = box
    return (x + w / 2, y + h)


def center(box):
    x, y, w, h, *_ = box
    return (x + w / 2, y + h / 2)


def main():
    fig, ax = plt.subplots(figsize=(11, 14))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-2.6, 10.6)
    ax.axis("off")
    ax.set_title(
        "LSTM 발전량 예측기 — 신호(데이터) 흐름 블럭도\n"
        "(센서·API에서 수집한 데이터가 어떻게 예측 결과로 변환되는가)",
        fontsize=14, fontweight="bold", pad=18,
    )

    for key, (x, y, w, h, label, color) in BOXES.items():
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.08,rounding_size=0.12",
            linewidth=1.4, edgecolor="#333333", facecolor=color,
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9.5, linespacing=1.4)

    for src, dst, sig_label in ARROWS:
        b_src, b_dst = BOXES[src], BOXES[dst]
        sx, sy = center(b_src)
        dx, dy = center(b_dst)
        if abs(sx - dx) < 0.3:
            start = center_bottom(b_src)
            end = center_top(b_dst)
        else:
            start = (sx, sy - BOXES[src][3] / 2)
            end = (dx, dy + BOXES[dst][3] / 2)
        arrow = FancyArrowPatch(
            start, end,
            arrowstyle="-|>", mutation_scale=16,
            linewidth=1.6, color="#1565c0", shrinkA=2, shrinkB=2,
        )
        ax.add_patch(arrow)
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        ax.text(mx + 0.15, my, f"▶ {sig_label}", fontsize=8.5, color="#1565c0", va="center")

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"[OK] 블럭도 이미지 저장 -> {OUT_PATH}")


if __name__ == "__main__":
    main()
