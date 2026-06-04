import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

MZ_THRESHOLD = 3.5
MZ_CONST = 0.6745



def load_data(data_dir):
    all_files = sorted(glob.glob(os.path.join(data_dir, "**", "*.parquet"), recursive=True))
    output_path = os.path.join(data_dir, "output")
    files = [f for f in all_files if not f.startswith(output_path)]
    if not files:
        print(f"[ERROR] Parquet-файлы не найдены в {data_dir}")
        print(f"        Убедись что папки month=... лежат рядом со скриптом")
        sys.exit(1)
    print(f"      Найдено файлов: {len(files)}")
    for f in files:
        print(f"        {os.path.relpath(f, data_dir)}")
    parts = [pd.read_parquet(f) for f in files]
    df = pd.concat(parts, ignore_index=True)
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")
    df["researchdate"] = pd.to_datetime(df["researchdate"]).dt.date
    if "CategoryNameDelivery" not in df.columns and "CategoryDelivery" in df.columns:
        df = df.rename(columns={"CategoryDelivery": "CategoryNameDelivery"})
    return df


def compute_daily_ots(df):
    dv = df[
        (df["BrandinDelivery"] == 1) &
        df["CategoryNameDelivery"].notna() &
        (df["CategoryNameDelivery"].astype(str).str.strip() != "")
    ].copy()

    counts = (
        dv.groupby(["SubjectID", "BrandID", "CategoryNameDelivery", "researchdate"])
        .size()
        .reset_index(name="count_rows")
    )
    weights = (
        dv.groupby(["SubjectID", "researchdate"])["Weight"]
        .first()
        .reset_index()
    )
    brand_meta = dv[["BrandID", "Brand"]].drop_duplicates("BrandID")

    daily = counts.merge(weights, on=["SubjectID", "researchdate"])
    daily["daily_ots"] = daily["Weight"] * daily["count_rows"]
    daily = daily.merge(brand_meta, on="BrandID", how="left")
    return daily


def compute_modified_zscore(daily):
    daily = daily.copy()
    daily["log_ots"] = np.log1p(daily["daily_ots"])

    cat_params = (
        daily.groupby("CategoryNameDelivery")["log_ots"]
        .agg(
            cat_median=lambda x: x.median(),
            cat_mad=lambda x: (x - x.median()).abs().median()
        )
        .reset_index()
    )

    daily = daily.merge(cat_params, on="CategoryNameDelivery")

    daily["mz_score"] = (
        MZ_CONST * (daily["log_ots"] - daily["cat_median"]) / daily["cat_mad"]
    )

    daily["threshold_log"] = daily["cat_median"] + MZ_THRESHOLD * daily["cat_mad"] / MZ_CONST
    daily["threshold_ots"] = np.expm1(daily["threshold_log"])

    return daily


def detect_anomalies(daily):
    flagged = daily[daily["mz_score"] > MZ_THRESHOLD].copy()

    def make_reason(row):
        return (
            f"Modified Z-score = {row['mz_score']:.3f} > {MZ_THRESHOLD} "
            f"(log(daily_ots) = {row['log_ots']:.3f}, "
            f"cat_median = {row['cat_median']:.3f}, "
            f"cat_MAD = {row['cat_mad']:.3f}); "
            f"daily_ots = {row['daily_ots']:,.0f} > порог {row['threshold_ots']:,.0f}"
        )

    flagged["reason"] = flagged.apply(make_reason, axis=1)
    return flagged


def build_output_tables(flagged):
    anomalies = (
        flagged[["SubjectID", "researchdate"]]
        .drop_duplicates()
        .sort_values(["researchdate", "SubjectID"])
        .reset_index(drop=True)
    )

    anomaly_reasons = (
        flagged[[
            "SubjectID", "researchdate", "BrandID", "Brand",
            "CategoryNameDelivery", "daily_ots", "mz_score",
            "threshold_ots", "reason"
        ]]
        .rename(columns={"mz_score": "score", "threshold_ots": "threshold"})
        .sort_values(["researchdate", "SubjectID", "BrandID"])
        .reset_index(drop=True)
    )

    return anomalies, anomaly_reasons


def _bad_pairs_set(anomalies):
    return set(
        zip(anomalies["SubjectID"].astype(str), anomalies["researchdate"].astype(str))
    )


def _ots_rows(df):
    dv = df[(df["BrandinDelivery"] == 1) & df["CategoryNameDelivery"].notna()].copy()
    dv["Weight"] = pd.to_numeric(dv["Weight"], errors="coerce")
    w = dv.groupby(["SubjectID", "researchdate"])["Weight"].first().reset_index()
    cr = (
        dv.groupby(["SubjectID", "BrandID", "CategoryNameDelivery", "researchdate"])
        .size()
        .reset_index(name="count_rows")
    )
    cr = cr.merge(w, on=["SubjectID", "researchdate"])
    cr["daily_ots"] = cr["Weight"] * cr["count_rows"]
    return cr


def compute_ots_by_day(df, anomalies):
    cr = _ots_rows(df)
    before = cr.groupby("researchdate")["daily_ots"].sum().rename("ots_before")
    bad = _bad_pairs_set(anomalies)
    mask = cr.apply(
        lambda r: (str(r["SubjectID"]), str(r["researchdate"])) not in bad, axis=1
    )
    after = cr[mask].groupby("researchdate")["daily_ots"].sum().rename("ots_after")
    result = pd.DataFrame({"ots_before": before, "ots_after": after}).fillna(0)
    result.index = pd.to_datetime(result.index)
    return result.sort_index().reset_index().rename(columns={"index": "researchdate"})


def compute_category_ots_change(df, anomalies):
    cr = _ots_rows(df)
    before = cr.groupby("CategoryNameDelivery")["daily_ots"].sum()
    bad = _bad_pairs_set(anomalies)
    mask = cr.apply(
        lambda r: (str(r["SubjectID"]), str(r["researchdate"])) not in bad, axis=1
    )
    after = cr[mask].groupby("CategoryNameDelivery")["daily_ots"].sum()
    result = pd.DataFrame({"before": before, "after": after}).dropna()
    result["pct_change"] = (result["after"] - result["before"]) / result["before"] * 100
    return result.sort_values("pct_change")


def plot_total_ots(ots_day, plots_dir):
    fig, ax = plt.subplots(figsize=(14, 5))
    dates = ots_day["researchdate"]
    avg_b = ots_day["ots_before"].mean()
    avg_a = ots_day["ots_after"].mean()
    pct = 100 * avg_a / avg_b if avg_b > 0 else 0
    ax.plot(dates, ots_day["ots_before"] / 1e3, color="red",
            label="OTS до фильтрации", linewidth=1.5)
    ax.plot(dates, ots_day["ots_after"] / 1e3, color="green",
            label="OTS после фильтрации", linewidth=1.5)
    ax.set_title(
        f"Изменение ежедневного OTS (до и после удаления аномалий)\n"
        f"avg_before = {avg_b/1e3:.1f}k, avg_after = {avg_a/1e3:.1f}k, "
        f"сохранено = {pct:.2f}%",
        fontsize=11
    )
    ax.set_xlabel("Дата")
    ax.set_ylabel("OTS (в тыс.)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "total_ots_before_after.png"), dpi=150)
    plt.close()


def plot_category_ots_change(cat_df, plots_dir):
    fig, ax = plt.subplots(figsize=(14, 8))
    cats = cat_df.index.tolist()
    vals = cat_df["pct_change"].tolist()
    bars = ax.barh(cats, vals, color="steelblue")
    for bar, v in zip(bars, vals):
        ax.text(
            v - 0.05 if v < 0 else v + 0.05,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.1f}",
            va="center",
            ha="right" if v < 0 else "left",
            fontsize=7.5
        )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Изменение OTS, %")
    ax.set_title("Изменение суммарного OTS по категориям после удаления аномалий (%)")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "category_ots_change.png"), dpi=150)
    plt.close()


def plot_daily_anomaly_count(anomalies, plots_dir):
    daily_cnt = (
        anomalies.groupby("researchdate")["SubjectID"]
        .nunique()
        .reset_index(name="n_anomalous")
    )
    daily_cnt["researchdate"] = pd.to_datetime(daily_cnt["researchdate"])
    daily_cnt = daily_cnt.sort_values("researchdate")
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(daily_cnt["researchdate"], daily_cnt["n_anomalous"],
           color="steelblue", width=0.8)
    ax.set_xlabel("Дата")
    ax.set_ylabel("Количество аномальных респондентов")
    ax.set_title("Количество аномальных респондентов по дням")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "daily_anomaly_count.png"), dpi=150)
    plt.close()


def analytics_respondent_profile(df, anomalies, groupby_col):
    dv = df[(df["BrandinDelivery"] == 1)].copy()
    dv["Weight"] = pd.to_numeric(dv["Weight"], errors="coerce")
    bad = _bad_pairs_set(anomalies)
    dv["is_anomalous"] = dv.apply(
        lambda r: (str(r["SubjectID"]), str(r["researchdate"])) in bad, axis=1
    )
    before = dv.groupby(groupby_col)["Weight"].sum().rename("before")
    after = dv[~dv["is_anomalous"]].groupby(groupby_col)["Weight"].sum().rename("after")
    result = pd.DataFrame({"before": before, "after": after}).fillna(0)
    result["pct_change"] = (result["after"] - result["before"]) / result["before"] * 100
    return result


def analytics_resource_profile(df, anomalies, groupby_col):
    return analytics_respondent_profile(df, anomalies, groupby_col)


def analytics_brand_ots_by_day(df, anomalies, brand_id):
    dv = df[(df["BrandinDelivery"] == 1) & (df["BrandID"] == brand_id)].copy()
    dv["Weight"] = pd.to_numeric(dv["Weight"], errors="coerce")
    w = dv.groupby(["SubjectID", "researchdate"])["Weight"].first().reset_index()
    cr = dv.groupby(["SubjectID", "researchdate"]).size().reset_index(name="count_rows")
    cr = cr.merge(w, on=["SubjectID", "researchdate"])
    cr["daily_ots"] = cr["Weight"] * cr["count_rows"]
    before = cr.groupby("researchdate")["daily_ots"].sum().rename("before")
    bad = _bad_pairs_set(anomalies)
    mask = cr.apply(
        lambda r: (str(r["SubjectID"]), str(r["researchdate"])) not in bad, axis=1
    )
    after = cr[mask].groupby("researchdate")["daily_ots"].sum().rename("after")
    result = pd.DataFrame({"before": before, "after": after}).fillna(0)
    result.index = pd.to_datetime(result.index)
    return result.sort_index()


def analytics_query_text(df, anomalies, subject_id, research_date):
    date_val = pd.to_datetime(research_date).date()
    mask = (df["SubjectID"] == subject_id) & (df["researchdate"] == date_val)
    cols = ["researchdate", "QueryText", "Brand", "CategoryNameDelivery", "ResourceName"]
    return df[mask][cols].reset_index(drop=True)


def print_summary(anomalies, anomaly_reasons, daily):
    total_resp = daily["SubjectID"].nunique()
    removed_resp = anomalies["SubjectID"].nunique()
    total_ots = daily["daily_ots"].sum()
    flagged_ots = anomaly_reasons["daily_ots"].sum()
    print("=" * 60)
    print("ИТОГ ПОИСКА АНОМАЛИЙ")
    print("=" * 60)
    print(f"  Всего респондентов:               {total_resp:,}")
    print(f"  Аномальных респондентов:          {removed_resp:,} ({100*removed_resp/total_resp:.2f}%)")
    print(f"  Пар (SubjectID, date) к удалению: {len(anomalies):,}")
    print(f"  Триггеров аномалий:               {len(anomaly_reasons):,}")
    print(f"  OTS триггеров / суммарный OTS:    {100*flagged_ots/total_ots:.2f}%")
    print(f"  Порог Modified Z-score:           {MZ_THRESHOLD}")
    print("=" * 60)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("[1/7] Загрузка данных...")
    df = load_data(DATA_DIR)
    print(f"      {len(df):,} строк, {df['SubjectID'].nunique():,} респондентов")

    print("[2/7] Вычисление daily_ots...")
    daily = compute_daily_ots(df)
    print(f"      {len(daily):,} записей (SubjectID × BrandID × CategoryNameDelivery × date)")

    print("[3/7] Вычисление Modified Z-score...")
    daily = compute_modified_zscore(daily)

    print("[4/7] Поиск аномалий...")
    flagged = detect_anomalies(daily)
    anomalies, anomaly_reasons = build_output_tables(flagged)
    print_summary(anomalies, anomaly_reasons, daily)

    print("[5/7] Сохранение результатов...")
    anomalies.to_csv(os.path.join(OUTPUT_DIR, "anomalies.csv"), index=False)
    anomaly_reasons.to_csv(os.path.join(OUTPUT_DIR, "anomaly_reasons.csv"), index=False)

    print("[6/7] Построение графиков...")
    ots_day = compute_ots_by_day(df, anomalies)
    cat_df = compute_category_ots_change(df, anomalies)
    plot_total_ots(ots_day, PLOTS_DIR)
    plot_category_ots_change(cat_df, PLOTS_DIR)
    plot_daily_anomaly_count(anomalies, PLOTS_DIR)
    print("      Графики сохранены в output/plots/")

    print("[7/7] Готово.")
    print()
    print("Аналитические функции доступны для импорта из solution.py:")
    print("  analytics_respondent_profile(df, anomalies, 'Пол')")
    print("  analytics_respondent_profile(df, anomalies, 'Возраст')")
    print("  analytics_respondent_profile(df, anomalies, 'Регион')")
    print("  analytics_respondent_profile(df, anomalies, 'Федеральный_округ')")
    print("  analytics_resource_profile(df, anomalies, 'ResourceName')")
    print("  analytics_resource_profile(df, anomalies, 'ResourceType')")
    print("  analytics_resource_profile(df, anomalies, 'Platform')")
    print("  analytics_resource_profile(df, anomalies, 'UseType')")
    print("  analytics_brand_ots_by_day(df, anomalies, brand_id)")
    print("  analytics_query_text(df, anomalies, subject_id, research_date)")


if __name__ == "__main__":
    main()
