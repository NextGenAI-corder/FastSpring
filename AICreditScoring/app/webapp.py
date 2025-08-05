from flask import Flask, render_template_string
import pandas as pd
import pickle
import numpy as np

# モデルとカテゴリマップの読み込み
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/cat_maps.pkl", "rb") as f:
    cat_maps = pickle.load(f)

# CSVファイルのパス
CSV_FILE = "data/input.csv"

# Flaskアプリケーションの初期化
app = Flask(__name__)

# HTMLテンプレート
HTML_PAGE = """
<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8">
  <title>与信スコア 一覧表示</title>
  <style>
    table, th, td {
      border: 1px solid black;
      border-collapse: collapse;
      padding: 5px;
      text-align: center;
    }
    .risk-high {
      background-color: #ffcccc;
      font-weight: bold;
    }
  </style>
</head>
<body>
  <h1>与信スコア 一覧表示</h1>
  {{ table|safe }}
</body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    try:
        # CSVファイルの読み込み
        df = pd.read_csv(CSV_FILE, encoding="utf-8-sig")

        # カテゴリマッピング（正方向）
        for col, mapping in cat_maps.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(0)

        # ターゲット列の退避
        delinquency_info_col = None
        if "DelinquencyInfo" in df.columns:
            delinquency_info_col = df["DelinquencyInfo"].copy()
            df = df.drop(columns=["DelinquencyInfo"])

        # 特徴量（全19列）を抽出
        feature_cols = [
            "Sex",
            "Marital",
            "Age",
            "Income",
            "CreditAppAmount",
            "OtherDebts",
            "DebtRestruct",
            "BorrowingRatio",
            "JobType",
            "Occupation",
            "Industry",
            "Education",
            "Dependents",
            "OwnHouse",
            "Foreigner",
            "Phone",
            "EmploymentYears",
            "Guarantor",
            "Collateral",
        ]
        # 不要な BorrowingRatio を除外（モデル未学習のため）
        # feature_cols.remove("BorrowingRatio")
        # if "BorrowingRatio" in feature_cols:
        #    feature_cols.remove("BorrowingRatio")

        X = df[feature_cols].copy()
        # 特徴量の順序を学習時と完全に一致させる
        # X = X[X_train.columns]
        # X = df.copy()
        # X = X[[col for col in feature_cols if col in df.columns]]

        # スコアリング（延滞確率を出力）
        proba_list = model.predict_proba(X)[:, 1]
        df["Score"] = np.round(proba_list, 4)

        # ターゲット列の復元（あれば）
        if delinquency_info_col is not None:
            df["DelinquencyInfo"] = delinquency_info_col

        # リバースマッピング
        cat_maps_rev = {
            col: {v: k for k, v in mapping.items()} for col, mapping in cat_maps.items()
        }
        for col, mapping_rev in cat_maps_rev.items():
            if col in df.columns:
                df[col] = df[col].map(mapping_rev).fillna("")

        # 表示対象列（任意に調整可）
        display_cols = [
            "Name",
            "Sex",
            "Age",
            "Income",
            "BorrowingRatio",
            "Industry",
            "CreditAppAmount",
            "OtherDebts",
            "DelinquencyInfo",
            "DebtRestruct",
            "Score",
        ]

        # スコアが0.7以上なら赤く表示
        def highlight_risk(val):
            try:
                return 'class="risk-high"' if float(val) >= 0.7 else ""
            except:
                return ""

        # テーブルをHTML形式に変換
        table_html = df[display_cols].to_html(
            index=False,
            escape=False,
            border=1,
            formatters={"Score": lambda x: f"<span {highlight_risk(x)}>{x}</span>"},
        )

        return render_template_string(HTML_PAGE, table=table_html)

    except Exception as e:
        return render_template_string(
            HTML_PAGE, table=f"<p style='color:red;'>エラー: {e}</p>"
        )
