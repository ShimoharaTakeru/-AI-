import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# --- 設定：ページのデザイン ---
st.set_page_config(page_title="ラグビー勝敗予想AI", layout="wide")

# --- 1. データの読み込みと学習 ---
# 毎回読み込むと遅いので、データをキャッシュ（一時保存）する仕組みを使います
@st.cache_data
def load_and_train_model():
    # スプレッドシートをCSVとして直接読み込む（公開設定になっている必要があります）
    sheet_id = '1Bi6tNgxPxbGVf3hZOhfC3bAsE91-qeBQL_BNxJx_R7g'
    url = f'https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv'
    
    try:
        # ヘッダーなしで読み込んでから処理
        df_raw = pd.read_csv(url, header=None)
    except Exception as e:
        return None, None, None, f"データ読み込みエラー: スプレッドシートの「リンクを知っている全員」への共有設定を確認してください。"

    # 行と列を入れ替え（転置）
    df = df_raw.transpose()
    df.columns = df.iloc[0]
    df = df.drop(df.index[0]).reset_index(drop=True)

    # 数値化処理
    def clean_num(x):
        if isinstance(x, str):
            x = x.replace(',', '').replace('%', '')
            if x == '-' or x == '': return 0
            return pd.to_numeric(x, errors='coerce')
        return x

    for col in df.columns:
        if col != '列 1':
            df[col] = df[col].apply(clean_num)

    # 学習データ作成
    target = '勝率'
    exclude_cols = ['勝率', '勝利数', '引き分け数', '敗北数', '順位', '勝ち点', '得失点差', '試合数', '得点', '失点', '列 1']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols].fillna(0)
    y = df[target].fillna(0)

    # モデル学習
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, feature_cols, df, None

# --- アプリのメイン処理 ---
st.title("🏉 ラグビー勝敗予想AI")
st.markdown("現在のスタッツを入力して、AIに勝敗を判定させます。")

# 読み込み実行
model, feature_cols, df_source, error_msg = load_and_train_model()

if error_msg:
    st.error(error_msg)
    st.stop()
else:
    st.success("✅ データの読み込みとAIの学習が完了しました！")

# --- 入力フォーム ---
st.markdown("---")
st.header("スタッツ入力")

# 画面を左右に分割
col1, col2 = st.columns(2)

input_me = []
input_enemy = []

with col1:
    st.subheader("🔵 自チーム")
    for col in feature_cols:
        # スプレッドシートの平均値を初期値の参考にするなどの工夫も可能ですが、ここではシンプルに0
        val = st.number_input(f"{col} (自)", value=0.0, step=1.0, key=f"me_{col}")
        input_me.append(val)

with col2:
    st.subheader("🔴 敵チーム")
    for col in feature_cols:
        val = st.number_input(f"{col} (敵)", value=0.0, step=1.0, key=f"enemy_{col}")
        input_enemy.append(val)

# --- 判定ボタン ---
st.markdown("---")
if st.button("🆚 勝敗を判定する", type="primary", use_container_width=True):
    # データフレーム作成
    df_me = pd.DataFrame([input_me], columns=feature_cols)
    df_enemy = pd.DataFrame([input_enemy], columns=feature_cols)
    
    # 予測
    score_me = model.predict(df_me)[0]
    score_enemy = model.predict(df_enemy)[0]
    
    diff = score_me - score_enemy
    
    # 結果表示
    st.markdown("## 📊 分析結果")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("自チーム評価スコア", f"{score_me:.1f}")
    c2.metric("敵チーム評価スコア", f"{score_enemy:.1f}")
    c3.metric("スコア差", f"{diff:.1f}")
    
    if diff > 10:
        st.success(f"🏆 **【自チームが圧倒的優勢】** です！このままいけば勝利確実です。")
    elif diff > 0:
        st.info(f"⚔️ **【自チームがやや優勢】** です。接戦ですが勝利に近いです。")
    elif diff > -10:
        st.warning(f"⚠️ **【敵チームがやや優勢】** です。相手のペースです。")
    else:
        st.error(f"🚨 **【敵チームが圧倒的優勢】** です。かなり厳しい状況です。")
