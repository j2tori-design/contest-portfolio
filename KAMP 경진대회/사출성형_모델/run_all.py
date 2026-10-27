"""전처리 → 진단 → 모델 비교(CV) → 최종모델 학습·보정·저장 → 테스트 추론 → 오류분석·현장활용 분석 → 부가 실험까지 한 번에 실행.

    python run_all.py               # 정식 실행 → outputs/
    python run_all.py --quick       # 축소 실행 (CV 반복 5→1회) → outputs_quick/  (점검용, 최종 성능으로 쓰지 않음)
    python run_all.py --no-extra    # 9단계 부가 실험 생략
    python run_all.py --clean       # 출력 폴더를 비우고 처음부터 생성
"""
import os
import sys

if "--quick" in sys.argv:                      # src 를 불러오기 전에 설정해야 반복 수·출력 폴더가 바뀐다
    os.environ["KAMP_QUICK"] = "1"
if "--no-extra" in sys.argv:
    os.environ["KAMP_NO_EXTRA"] = "1"

import hashlib
import importlib.metadata
import json
import platform
import shutil
import time
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold

from src import experiments as ex
from src import insights as ins
from src.config import (DATA_DIR, INDEX_COL, N_REPEATS, N_SPLITS, OUT_DIR, QUICK, ROOT, RUN_EXTRA, SEED, TARGET,
                        TEST_FRAC)
from src.data import (data_quality_summary, diagnose, feature_columns, feature_matrix, load_all, load_labeled,
                      process_cols, scaling_check, structure_report)
from src.evaluate import best_threshold, grouped_cv, inner_oof, metrics
from src.models import get_models

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

PACKAGES = ["numpy", "pandas", "scikit-learn", "scipy", "lightgbm", "xgboost", "catboost", "imbalanced-learn", "shap",
            "matplotlib", "seaborn", "joblib"]


def holdout_split(df):
    """사출 그룹·불량 층화로 TEST_FRAC 만큼 테스트셋 분리 (모델·임계값·보정 선택에 사용하지 않음).

    연속 행 기준 그룹(seq_group)으로 분할한 뒤, 교차 기록된 짝이 양쪽에 걸치면 짝 전체를 먼저 나온 행 쪽으로 옮긴다.
    """
    cv = StratifiedGroupKFold(n_splits=round(1 / TEST_FRAC), shuffle=True, random_state=SEED)
    strat = df[TARGET].astype(str) + "_" + df["product"]
    _, test_idx = next(cv.split(df, strat, df["seq_group"]))
    side = pd.Series(False, index=df.index)
    side.iloc[test_idx] = True
    first_row = df.groupby("group")[INDEX_COL].transform("idxmin")      # 그룹에서 생산 순번이 가장 이른 행
    first_side = pd.Series(side.to_numpy()[first_row.to_numpy()], index=df.index)
    moved = int((side != first_side).groupby(df["group"]).any().sum())
    dev, test = df[~first_side].reset_index(drop=True), df[first_side].reset_index(drop=True)
    dev.attrs["moved_groups"] = moved
    return dev, test


def save(df, name, **kw):
    df.to_csv(OUT_DIR / name, index=kw.pop("index", False), encoding="utf-8-sig", **kw)


def file_hashes():
    files = sorted(DATA_DIR.glob("*.csv")) + sorted((ROOT / "src").glob("*.py")) + [ROOT / "run_all.py"]
    return {str(f.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(f.read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:16]
            for f in files}


def versions():
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for p in PACKAGES:
        try:
            out[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            out[p] = "미설치"
    return out


def write_manifest(t0, status, steps):
    man = {"status": status, "started": datetime.fromtimestamp(t0).isoformat(timespec="seconds"),
           "finished": datetime.now().isoformat(timespec="seconds"), "runtime_sec": round(time.time() - t0),
           "mode": "quick(축소 실행)" if QUICK else "full", "run_extra": RUN_EXTRA,
           "config": {"SEED": SEED, "N_SPLITS": N_SPLITS, "N_REPEATS": N_REPEATS, "TEST_FRAC": TEST_FRAC},
           "environment": versions(), "input_hashes": file_hashes(), "steps": steps}
    (OUT_DIR / "run_manifest.json").write_text(json.dumps(man, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    t0 = time.time()
    if "--clean" in sys.argv and OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    (OUT_DIR / "model").mkdir(parents=True, exist_ok=True)
    steps = []
    write_manifest(t0, "running", steps)
    if QUICK:
        print("[quick] 축소 실행: 반복 1회 → outputs_quick/ (정식 결과와 수치가 다르며 최종 성능으로 쓰지 않음)\n")

    # 1. 데이터 적재 + 진단
    df = load_all()
    diag = diagnose(df)
    save(diag, "data_diagnosis.csv")
    st_sum, st_single = structure_report(df)
    save(st_sum, "structure_summary.csv")
    save(st_single, "single_groups.csv")
    save(data_quality_summary(df), "data_quality.csv")
    save(scaling_check(), "scaling_check.csv")
    feats = [c for c in process_cols(load_labeled("cn7")) if df[c].nunique() > 1]
    save(ins.outlier_table(df, feats), "outliers.csv")
    save(ins.variable_stage_table(df), "variable_stages.csv")
    print("[1] 데이터 진단\n", diag.T, "\n", st_sum.T, "\n")
    steps.append("1 진단")

    # 2. 개발/테스트 분리 + 분할 점검
    dev, test = holdout_split(df)
    fcols = feature_columns(dev)                                   # 입력 열은 개발셋 기준으로 고정
    audit = ins.split_audit(df, dev, test)
    save(audit, "split_audit.csv")
    assert (audit["결과"] == "통과").all(), audit
    counts = []
    for name, d in [("dev", dev), ("test", test)]:
        for scope in ["all", "cn7", "rg3"]:
            s = d if scope == "all" else d[d["product"] == scope]
            counts.append({"set": name, "scope": scope, "행 수": len(s), "추정 사출 그룹 수": s["group"].nunique(),
                           "불량 수": int(s[TARGET].sum()), "불량률": s[TARGET].mean()})
    save(pd.DataFrame(counts), "sample_counts.csv")
    print(f"[2] dev {len(dev)}행 (불량 {dev[TARGET].sum()}) / test {len(test)}행 (불량 {test[TARGET].sum()}),"
          f" 양쪽에 걸쳐 옮긴 짝 {dev.attrs['moved_groups']}개\n", audit.to_string(index=False), "\n")
    steps.append("2 분할·누수 점검")

    # 3. 모델 × 특성셋 비교 (dev 에서 반복 그룹 CV)
    y = dev[TARGET].to_numpy()
    groups = dev["group"].to_numpy()
    products = dev["product"].to_numpy()
    records, oofs, fold_preds = [], {}, {}
    for use_cavity in (False, True):
        X = feature_matrix(dev, use_cavity, fcols)
        fs = "process+cavity" if use_cavity else "process_only"
        for name, model in get_models().items():
            recs, oof, fp = grouped_cv(model, X, y, groups, products, keep_preds=True)
            for r in recs:
                r.update(model=name, features=fs)
            records += recs
            oofs[(name, fs)], fold_preds[(name, fs)] = oof, fp
            print(f"  done {name:13s} {fs}  ({time.time() - t0:.0f}s)")

    cv = pd.DataFrame(records)
    save(cv, "cv_folds.csv")
    metric_cols = ["PR_AUC", "ROC_AUC", "F1", "Precision", "Recall", "Recall@5%", "Recall@10%", "Recall@20%",
                   "Precision@20%", "F1@20%"]
    save(cv.groupby(["scope", "features", "model"])[metric_cols].agg(["mean", "std"]).round(4), "cv_summary.csv", index=True)
    print("\n[3] CV 요약 (평균)\n", cv.groupby(["scope", "features", "model"])[metric_cols].mean().round(3), "\n")
    steps.append("3 모델 비교 CV")

    # 4. 최종모델 선정: 개발셋 CV 전체 PR-AUC 평균 최대 (사전에 정한 규칙)
    allscope = cv[cv.scope == "all"].groupby(["model", "features"])["PR_AUC"].mean()
    best_model, best_fs = allscope.idxmax()
    print(f"[4] 최종모델: {best_model} / {best_fs} (CV PR-AUC {allscope.max():.3f})")
    sig = ex.significance_tests(cv, best_model, best_fs)
    save(sig, "significance.csv")
    print("\n[4-1] 최종모델 대비 유의성 (PR-AUC)\n",
          sig[sig.metric == "PR_AUC"][["compared_model", "features", "mean_diff", "wins", "losses",
                                        "corrected_t_p", "corrected_t_p_holm"]].round(3).to_string(index=False), "\n")

    use_cavity = best_fs == "process+cavity"
    X_dev, X_test = feature_matrix(dev, use_cavity, fcols), feature_matrix(test, use_cavity, fcols)
    assert list(X_dev.columns) == list(X_test.columns)
    model = get_models()[best_model]
    thr = best_threshold(y, inner_oof(model, X_dev, y, groups, SEED))   # 개발셋 내부 그룹 CV 로 정한 공통 임계값
    final = clone(model).fit(X_dev, y)
    steps.append("4 최종모델 선정·학습")

    # 5. 확률보정: 보정 없음 / sigmoid / isotonic 중 Brier 최소 방식을 표시용 확률로 사용
    cal, cal_oof = ins.calibration_comparison(dev, model, use_cavity)
    save(cal, "calibration_cv.csv")
    brier = cal.groupby("method")["Brier"].mean()
    cal_method = brier.idxmin()
    save(cal_oof, "calibration_oof.csv")
    print(f"\n[5] 확률보정 Brier {brier.round(5).to_dict()} → 표시용: {cal_method} ({time.time() - t0:.0f}s)")
    final_cal = final if cal_method == "raw" else ins.calibrated(model, groups, y, cal_method).fit(X_dev, y)
    cal_prob_dev = cal_oof[f"prob_{cal_method}"].to_numpy()
    grade_cut = {"위험": float(np.quantile(cal_prob_dev, 0.90)), "주의": float(np.quantile(cal_prob_dev, 0.70))}
    steps.append("5 확률보정")

    # 모델 저장 (predict.py 에서 사용)
    joblib.dump(final, OUT_DIR / "model" / "final_model.joblib")
    joblib.dump(final_cal, OUT_DIR / "model" / "calibrated_model.joblib")
    meta = {"model": best_model, "features": best_fs, "use_cavity": bool(use_cavity), "feature_columns": list(X_dev.columns),
            "process_columns": fcols, "threshold_common": thr, "calibration": cal_method, "grade_cut": grade_cut,
            "operating_point": "제품별 위험점수 상위 20% 검사", "trained_on": f"dev {len(dev)}행",
            "environment": versions()}
    (OUT_DIR / "model" / "model_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # 6. 테스트 추론
    p_test = final.predict_proba(X_test)[:, 1]
    y_test = test[TARGET].to_numpy()
    test_m = {"model": best_model, "features": best_fs, "threshold": thr, "calibration": cal_method,
              "grade_cut": grade_cut, "all": metrics(y_test, p_test, thr)}
    for prod in ["cn7", "rg3"]:
        m = test["product"].to_numpy() == prod
        test_m[prod] = metrics(y_test[m], p_test[m], thr)
    (OUT_DIR / "test_metrics.json").write_text(json.dumps(test_m, indent=2, ensure_ascii=False), encoding="utf-8")

    pred = test[[INDEX_COL, "product", "shot_id", "cavity", "pair_type", TARGET]].copy()
    pred["risk_score"] = p_test                                     # 순위용 위험점수 (보정 전)
    pred["risk_prob_cal"] = final_cal.predict_proba(X_test)[:, 1]   # 표시용 보정 확률
    pred["risk_grade"] = np.select([pred.risk_prob_cal >= grade_cut["위험"], pred.risk_prob_cal >= grade_cut["주의"]],
                                   ["위험", "주의"], "정상")
    pred["pred_common_thr"] = (p_test >= thr).astype(int)
    pred["inspect_rank_all"] = pred["risk_score"].rank(ascending=False, method="first").astype(int)
    pred["inspect_rank_product"] = pred.groupby("product")["risk_score"].rank(ascending=False, method="first").astype(int)
    n_prod = pred.groupby("product")["risk_score"].transform("size")
    pred["inspect_top20_product"] = (pred["inspect_rank_product"] <= np.ceil(n_prod * 0.2)).astype(int)
    pred = pred.join(ex.top_factors(final, X_test))                 # 예측 기여 변수 top3
    save(pred.sort_values("inspect_rank_all"), "test_predictions.csv")
    save(ins.bootstrap_test_ci(pred, prob_col="risk_score", thr=thr), "test_bootstrap_ci.csv")
    print("\n[6] 테스트 성능\n",
          pd.DataFrame({k: v for k, v in test_m.items() if isinstance(v, dict) and k != "grade_cut"}).round(3))
    steps.append("6 테스트 추론")

    # 7. 개발셋 OOF 와 판정 관점별 혼동행렬
    fp = fold_preds[(best_model, best_fs)]
    save(fp.assign(**{INDEX_COL: dev[INDEX_COL].to_numpy()[fp["row"]], "product": products[fp["row"]],
                      "group": groups[fp["row"]], TARGET: y[fp["row"]]}), "oof_fold_preds.csv")
    oof = dev.copy()
    oof["oof_prob"] = oofs[(best_model, best_fs)]               # 반복 평균 OOF 위험점수
    oof["pred_common_thr"] = (oof["oof_prob"] >= thr).astype(int)
    oof["error_common_thr"] = np.select([(oof[TARGET] == 1) & (oof.pred_common_thr == 0),
                                         (oof[TARGET] == 0) & (oof.pred_common_thr == 1)], ["FN", "FP"], "OK")
    oof = ins.add_operating_point(oof)
    save(oof, "oof_dev.csv")
    views = ins.oof_threshold_views(dev, fp, oof["oof_prob"].to_numpy(), thr)
    save(views, "oof_threshold_views.csv")
    print("\n[7] OOF 판정 관점별 혼동행렬\n", views.round(3).to_string(index=False))
    steps.append("7 OOF 판정 관점")

    # 8. 과거→미래 대리 검증 · 오류 집중 조건 · 규칙 · 검사 용량 · 비용 · 사전경보
    fwd = [ins.forward_validation(dev, model, use_cavity).assign(model=best_model, features=best_fs)]
    if use_cavity:
        fwd.append(ins.forward_validation(dev, model, False).assign(model=best_model, features="process_only"))
    fwd = pd.concat(fwd, ignore_index=True)
    save(fwd, "forward_cv.csv")
    rc, assign = ex.regime_clusters()
    save(rc, "regime_clusters.csv")
    save(assign, "regime_cluster_assign.csv")
    top_vars = [c for c in ["Clamp_Close_Time", "Max_Injection_Speed", "Plasticizing_Position", "Mold_Temperature_4",
                            "Max_Back_Pressure"] if c in oof.columns]
    save(ins.concentration_tables(oof, top_vars, assign), "op_concentration.csv")
    rule_feats = [c for c in X_dev.columns if c != "is_rg3"]
    for kind in ["FN", "FP"]:
        text, leaf = ins.error_rules(oof, rule_feats, kind)
        (OUT_DIR / f"op_rules_{kind}.txt").write_text(text, encoding="utf-8")
        save(leaf, f"op_rules_{kind}_leaves.csv", index=True)
    save(ins.rule_baselines(oof), "rule_baselines.csv")
    save(ins.capacity_table(oof), "capacity_table.csv")
    curves, best = ins.cost_curves(oof)
    save(curves, "cost_curves.csv")
    save(best, "cost_best.csv")
    save(ins.cost_breakeven(cal_oof, f"prob_{cal_method}"), "cost_breakeven.csv")
    s1, m1 = ins.early_warning(dev, ins.process_instability(dev, fcols), signal="공정 흔들림")
    s2, m2 = ins.early_warning(dev, ins.time_block_oof(dev, model, use_cavity), signal="모델 점수")
    save(pd.concat([s1, s2], ignore_index=True), "early_warning_series.csv")
    ew = pd.concat([m1, m2], ignore_index=True)
    save(ew, "early_warning_summary.csv")
    print(f"\n[8] 과거→미래 대리 검증\n{fwd.round(3).to_string(index=False)}\n운영점 혼동행렬\n"
          f"{pd.crosstab(oof['product'], oof['op_type'])}\n사전경보\n{ew.round(3).to_string(index=False)} ({time.time() - t0:.0f}s)")
    steps.append("8 현장 분석")
    write_manifest(t0, "core_done", steps)

    # 9. 부가 실험
    if RUN_EXTRA:
        run_extra(dev, t0, model, use_cavity)
        steps.append("9 부가 실험")
    write_manifest(t0, "done", steps)
    print(f"\n완료 ({time.time() - t0:.0f}s) → {OUT_DIR.name}/")


def run_extra(dev, t0, best_model, use_cavity):
    imb = ex.imbalance_comparison(dev, best_model, use_cavity)
    save(imb, "imbalance_cv.csv")
    print("\n[9-1] 불균형 처리 방식 비교 (최종모델 기준)\n",
          imb.groupby("method")[["PR_AUC_all", "PR_AUC_cn7", "PR_AUC_rg3", "Recall@20%"]].mean().round(3),
          f"({time.time() - t0:.0f}s)")

    pm = ex.product_model_comparison(dev, best_model, use_cavity)
    save(pm, "product_model_cv.csv")
    print("\n[9-2] 통합 모델 vs 제품별 모델\n", pm.groupby(["product", "approach"])[["PR_AUC", "Recall@20%"]].mean().round(3),
          f"({time.time() - t0:.0f}s)")

    lk = ex.leakage_comparison(dev)
    save(lk, "leakage_cv.csv")
    print("\n[9-3] 검증 방식별 PR-AUC (행 단위 vs 사출 그룹 단위)\n",
          lk.groupby(["model", "scheme"])["PR_AUC"].agg(["mean", "std"]).round(3), f"({time.time() - t0:.0f}s)")

    dm = ex.domain_discrimination()
    save(dm, "unlabeled_domain_auc.csv")
    print("\n[9-4] 라벨/비라벨 판별 AUC\n", dm.round(3).to_string(index=False), f"({time.time() - t0:.0f}s)")

    st = ex.self_training(dev)
    save(st, "self_training_cv.csv")
    print("\n[9-5] 자기학습 실험 (평균)\n",
          st.groupby("method")[["PR_AUC_all", "PR_AUC_cn7", "PR_AUC_rg3", "Recall@20%"]].mean().round(3),
          f"({time.time() - t0:.0f}s)")

    dl = ex.dl_stacking_comparison(dev, best_model, use_cavity)
    save(dl, "dl_stacking_cv.csv")
    print("\n[9-6] 스태킹 앙상블 · 딥러닝(MLP) 비교\n",
          dl.groupby("model", sort=False)[["PR_AUC_all", "ROC_AUC", "Recall@20%", "PR_AUC_cn7", "PR_AUC_rg3"]].mean().round(3),
          f"({time.time() - t0:.0f}s)")

    tb = ex.time_block_validation(dev)
    save(tb, "time_block_cv.csv")
    print("\n[9-7] 생산순번 블록 CV\n",
          tb[tb.model != "-"].groupby(["n_blocks", "features", "model"])[["PR_AUC_all", "ROC_AUC", "Recall@20%"]].mean().round(3),
          "\n평가 불가 fold:", int((tb.model == "-").sum()), f"({time.time() - t0:.0f}s)")

    ae = ex.autoencoder_cv(dev)
    save(ae, "autoencoder_cv.csv")
    print("\n[9-8] 오토인코더(양품 학습, 복원오차) CV 평균\n",
          ae[["PR_AUC_all", "PR_AUC_cn7", "PR_AUC_rg3", "Recall@20%"]].mean().round(3).to_dict(),
          f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
