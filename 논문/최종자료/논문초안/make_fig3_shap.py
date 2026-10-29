# 논문 그림 3: 최종 모델의 원래 변수 단위 SHAP (outputs_final/shap_by_variable.csv에서 읽음, 숫자 손입력 없음)
import os
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sv = pd.read_csv(os.path.join(HERE, "..", "outputs_final", "shap_by_variable.csv"))
KO = {"Intake Condition": "건강상태", "age_days": "나이", "is_mixed_breed": "잡종 여부", "animal_type": "동물 종류",
      "breed_group": "품종군", "Breed": "품종", "is_sterilized": "중성화 여부", "Intake Type": "입소 유형",
      "life_stage": "생애단계", "intake_month": "입소 월", "Color": "색상", "is_multicolor": "다색 여부"}
col = "mean|SHAP| (합산)"
top, rest = sv.head(6), sv.iloc[6:]

plt.rcParams["font.family"] = "Malgun Gothic"
fig, ax = plt.subplots(figsize=(5.6, 2.4), dpi=300)
d = top.iloc[::-1]
colors = ["#1d3a5f" if v in ("Intake Condition", "age_days") else "#2a7ad5" for v in d["변수"]]
bars = ax.barh(d["변수"].map(KO), d[col], color=colors, height=0.68)
ax.bar_label(bars, fmt="%.3f", fontsize=12, padding=3)
ax.set_xlim(0, d[col].max() * 1.17)
ax.tick_params(axis="y", labelsize=13, length=0)
ax.tick_params(axis="x", labelsize=11)
ax.set_xlabel("평균 |SHAP| (로그 오즈)", fontsize=12)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="x", alpha=.3)
ax.set_axisbelow(True)
ax.text(0.99, 0.04, f"나머지 {len(rest)}개 변수는 각 {rest[col].max():.3f} 이하", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=11, color="#555555")
plt.tight_layout()
fig.savefig(os.path.join(HERE, "fig3_shap.png"), dpi=300, facecolor="white")
print(top[["변수", col]].round(3).to_string(index=False), "| 나머지 최대", round(rest[col].max(), 3))
