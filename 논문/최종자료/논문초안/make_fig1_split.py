# 논문 그림 1: 결과 확정 시점 기준 분할 구조 (outputs_final/split_table.csv의 날짜로 그림)
# 행 = 단계(무엇을 하는가), 색 = 데이터의 역할(무엇으로 쓰이는가)
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

HERE = os.path.dirname(os.path.abspath(__file__))
st = pd.read_csv(os.path.join(HERE, "..", "outputs_final", "split_table.csv")).set_index("구간")
TRAIN_START = pd.Timestamp("2013-10-01")
C = {"학습": "#c8d6e3", "선택 폴드": "#1d3a5f", "보정 구간": "#277f8e", "테스트": "#b8312a"}
ROWS = [("모델 선택 ①", "선택 폴드 1", "선택 폴드"), ("모델 선택 ②", "선택 폴드 2", "선택 폴드"),
        ("모델 선택 ③", "선택 폴드 3", "선택 폴드"), ("임곗값 설정", "보정 구간", "보정 구간"),
        ("최종 평가", "테스트", "테스트")]

plt.rcParams["font.family"] = "Malgun Gothic"
fig, ax = plt.subplots(figsize=(6.5, 2.6), dpi=300)
for i, (label, key, role) in enumerate(ROWS):
    y = len(ROWS) - 1 - i
    tr_end = pd.Timestamp(st.loc[key, "학습 입소 끝"])
    ev0, ev1 = [pd.Timestamp(s.strip()) for s in st.loc[key, "평가 입소"].split("~")]
    ax.barh(y, tr_end - TRAIN_START, left=TRAIN_START, height=0.55, color=C["학습"])
    ax.barh(y, ev1 - ev0, left=ev0, height=0.55, color=C[role])
ax.set_yticks(range(len(ROWS)))
ax.set_yticklabels([r[0] for r in ROWS][::-1], fontsize=11)
ax.xaxis.set_major_locator(mdates.YearLocator(2)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.tick_params(axis="x", labelsize=10); ax.tick_params(axis="y", length=0)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
handles = [plt.Rectangle((0, 0), 1, 1, color=C[k]) for k in C]
ax.legend(handles, ["학습 데이터", "선택 폴드(검증)", "보정 구간", "테스트(2024)"], ncol=4, frameon=False,
          loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=10, handlelength=1.2, columnspacing=1.2)
plt.tight_layout()
fig.savefig(os.path.join(HERE, "fig1_split.png"), dpi=300, facecolor="white")
print(st[["평가 입소", "학습 입소 끝"]].head(5).to_string())
