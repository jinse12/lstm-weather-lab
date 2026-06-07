"""LSTM 발전량 예측기 블럭도 위에서 신호(데이터)가 이동하는 모습을 GIF 애니메이션으로 만든다.

블럭 레이아웃은 make_blockdiagram.py 의 BOXES 를 그대로 재사용하고,
화살표 경로를 따라 빛나는 점(신호)이 순서대로 이동하며
각 구간에서 무슨 일이 일어나는지 자막으로 설명한다.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, FancyBboxPatch

from make_blockdiagram import BOXES, center, center_bottom, center_top

OUT_PATH = "lstm_pipeline_signal_animation.gif"

for font_name in ("Malgun Gothic", "맑은 고딕", "NanumGothic"):
    if any(font_name == f.name for f in font_manager.fontManager.ttflist):
        rcParams["font.family"] = font_name
        break
rcParams["axes.unicode_minus"] = False

FRAMES_PER_SEGMENT = 24
PAUSE_FRAMES = 12
TRAIL_LEN = 8

# (출발 블럭, 도착 블럭, 신호 색, 친절한 설명 자막)
SEGMENTS = [
    ("weather_api", "weather_hourly", "#1565c0",
     "① 기상 API에서 받아온 시간별 기온·습도 등의 값을\nweather_hourly 표에 차곡차곡 저장합니다"),
    ("rp2040", "power_realtime", "#ef6c00",
     "② RP2040 센서가 1초마다 보내주는 발전량과\n패널 온습도 값을 power_realtime 표에 저장합니다"),
    ("power_realtime", "power_hourly", "#ef6c00",
     "③ 1초 단위로 쌓인 기록을 1시간 단위 평균으로 묶어서\npower_hourly 표(뷰)를 만듭니다"),
    ("weather_hourly", "join", "#1565c0",
     "④ 같은 시각의 '기상 정보'를 가져옵니다"),
    ("power_hourly", "join", "#ef6c00",
     "④ 같은 시각의 '발전량 정보'도 함께 가져와\n총 8가지 항목으로 표를 정리합니다"),
    ("join", "preprocess", "#2e7d32",
     "⑤ 정리된 데이터를 컴퓨터가 이해하기 쉽도록\n0~1 사이 값으로 맞추고, 과거 35시간씩 묶습니다"),
    ("preprocess", "lstm", "#2e7d32",
     "⑥ 35시간 동안의 흐름을 LSTM 모델에 보여주면,\n모델이 다음 1시간 뒤의 발전량을 추론합니다"),
    ("lstm", "inverse", "#8e24aa",
     "⑦ 모델이 내놓은 0~1 사이의 결과값을\n실제 단위인 kW(킬로와트)로 환산해 최종 예측치를 보여줍니다"),
]


def edge_points(src_key, dst_key):
    b_src, b_dst = BOXES[src_key], BOXES[dst_key]
    sx, sy = center(b_src)
    dx, dy = center(b_dst)
    if abs(sx - dx) < 0.3:
        return center_bottom(b_src), center_top(b_dst)
    return (sx, sy - b_src[3] / 2), (dx, dy + b_dst[3] / 2)


def ease_in_out(t: float) -> float:
    """부드럽게 출발하고 부드럽게 도착하도록 하는 보간 함수."""
    return t * t * (3 - 2 * t)


def main():
    fig, ax = plt.subplots(figsize=(11, 14))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-2.7, 10.6)
    ax.axis("off")
    ax.set_title(
        "LSTM 발전량 예측기 — 신호(데이터) 이동 시뮬레이션\n"
        "빛나는 점이 화살표를 따라 이동하며, 각 단계에서 무슨 일이 일어나는지 알려줍니다",
        fontsize=12.5, fontweight="bold", pad=16,
    )

    box_patches = {}
    for key, (x, y, w, h, label, color) in BOXES.items():
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.08,rounding_size=0.12",
            linewidth=1.4, edgecolor="#333333", facecolor=color,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9, linespacing=1.4)
        box_patches[key] = (patch, color)

    # 신호 잔상(트레일) - 점이 지나간 자리에 옅게 남는 효과
    trail_circles = []
    for i in range(TRAIL_LEN):
        c = Circle((0, 0), 0.15 * (1 - i / TRAIL_LEN), color="#1565c0", alpha=0.0, zorder=4)
        ax.add_patch(c)
        trail_circles.append(c)

    glow = Circle((0, 0), 0.32, color="#1565c0", alpha=0.22, zorder=4)
    ax.add_patch(glow)
    signal_dot = Circle((0, 0), 0.16, color="#1565c0", zorder=5)
    ax.add_patch(signal_dot)

    caption = ax.text(
        5.0, -2.35, "", ha="center", va="center", fontsize=11,
        color="#222222", linespacing=1.6,
        bbox=dict(boxstyle="round,pad=0.55", facecolor="#f7f7f7", edgecolor="#bbbbbb"),
    )
    step_label = ax.text(
        9.7, 10.3, "", ha="right", va="top", fontsize=10, color="#555555",
    )

    seg_total = FRAMES_PER_SEGMENT + PAUSE_FRAMES
    total_frames = len(SEGMENTS) * seg_total
    history: list[tuple[float, float, str]] = []
    state = {"prev_seg": -1}

    def reset_box_styles():
        for patch, original_face in box_patches.values():
            patch.set_facecolor(original_face)
            patch.set_edgecolor("#333333")
            patch.set_linewidth(1.4)
            patch.set_alpha(1.0)

    def update(frame):
        seg_idx = min(frame // seg_total, len(SEGMENTS) - 1)
        local_frame = frame % seg_total
        src_key, dst_key, color, text = SEGMENTS[seg_idx]
        start, end = edge_points(src_key, dst_key)

        if seg_idx != state["prev_seg"]:
            history.clear()
            state["prev_seg"] = seg_idx

        reset_box_styles()
        src_patch, _ = box_patches[src_key]
        dst_patch, _ = box_patches[dst_key]
        src_patch.set_alpha(0.5)
        dst_patch.set_edgecolor(color)
        dst_patch.set_linewidth(2.8)

        t_raw = min(local_frame / (FRAMES_PER_SEGMENT - 1), 1.0)
        t = ease_in_out(t_raw)
        x = start[0] + (end[0] - start[0]) * t
        y = start[1] + (end[1] - start[1]) * t

        history.append((x, y, color))
        if len(history) > TRAIL_LEN + 1:
            history.pop(0)
        past = history[:-1][::-1]
        for i, circle in enumerate(trail_circles):
            if i < len(past):
                hx, hy, hcolor = past[i]
                circle.center = (hx, hy)
                circle.set_color(hcolor)
                circle.set_alpha(0.30 * (1 - i / TRAIL_LEN))
            else:
                circle.set_alpha(0.0)

        signal_dot.center = (x, y)
        signal_dot.set_color(color)
        glow.center = (x, y)
        glow.set_color(color)

        caption.set_text(text)
        step_label.set_text(f"단계 {seg_idx + 1} / {len(SEGMENTS)}")
        return (signal_dot, glow, caption, step_label, *trail_circles)

    anim = FuncAnimation(fig, update, frames=total_frames, interval=70, blit=False)
    anim.save(OUT_PATH, writer=PillowWriter(fps=15))
    print(f"[OK] 신호 이동 애니메이션 저장 -> {OUT_PATH}")


if __name__ == "__main__":
    main()
