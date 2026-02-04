import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import os
import io
import base64

# --- 1. 日本語フォント設定 (ローカル & Cloud 両対応) ---
def setup_font():
    """fontsフォルダからフォントを読み込み、日本語表示を有効化"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    font_path = os.path.join(current_dir, "fonts", "ipaexg.ttf")
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        prop = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = prop.get_name()
        return prop.get_name()
    else:
        # フォールバック: システムフォントを試行
        plt.rcParams['font.family'] = ['Meiryo', 'MS Gothic', 'Hiragino Sans', 'sans-serif']
        return 'sans-serif'

font_name = setup_font()
plt.rcParams['axes.unicode_minus'] = False  # マイナス記号の文字化け対策
sns.set_theme(style="whitegrid", rc={"font.family": font_name})

st.set_page_config(page_title="イオン 四半期別セグメント業績分析", layout="wide")

# --- 2. ユーティリティ関数 ---
def get_html_report(df, title, fig=None):
    """HTMLダウンロード用データの生成（テーブル＋チャート）"""
    # チャートをbase64エンコード
    chart_html = ""
    if fig is not None:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        chart_html = f'<div style="text-align:center; margin: 20px 0;"><img src="data:image/png;base64,{img_base64}" style="max-width:100%;"/></div>'
    
    return f"""
    <html><head><meta charset='utf-8'>
    <style>
        body {{ font-family: 'Hiragino Sans', 'Meiryo', sans-serif; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; background: white; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: right; }}
        th {{ background: linear-gradient(135deg, #1f77b4, #ff7f0e); color: white; text-align: center; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        tr:hover {{ background-color: #f0f0f0; }}
        h2 {{ color: #2C3E50; border-left: 5px solid #1f77b4; padding-left: 15px; margin-top: 0; }}
        .timestamp {{ color: #888; font-size: 12px; text-align: right; margin-top: 20px; }}
    </style></head>
    <body>
    <div class="container">
        <h2>📊 {title}</h2>
        {chart_html}
        <h3>📋 詳細データ</h3>
        {df.to_html(classes='data-table')}
        <p class="timestamp">生成日時: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    </body></html>
    """

def sort_quarter_key(q):
    """四半期データのソート用キー（FY2023-3Q → (2023, 3)）"""
    try:
        if '-' in q:
            parts = q.replace('FY', '').split('-')
            year = int(parts[0])
            quarter = int(parts[1].replace('Q', ''))
            return (year, quarter)
        else:
            return (int(q.replace('FY', '')), 0)
    except:
        return (0, 0)

# --- 3. データの読み込み ---
def convert_to_numeric(series):
    """カンマ区切り文字列を数値に変換"""
    if series.dtype == 'object':
        return pd.to_numeric(
            series.astype(str).str.replace(',', '').str.strip(),
            errors='coerce'
        ).fillna(0)
    return series

@st.cache_data
def load_segment_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(current_dir, "data", "segment_data.csv")
    if os.path.exists(path):
        # エンコーディングを自動判定して読み込み（Windows対応）
        encodings = ['utf-8', 'cp932', 'shift_jis', 'utf-8-sig']
        df = None
        for encoding in encodings:
            try:
                df = pd.read_csv(path, encoding=encoding)
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        if df is None:
            return None
        
        # 四半期データのみを抽出（Q1, Q2, Q3, Q4）
        df = df[df['決算種別'].isin(['Q1', 'Q2', 'Q3', 'Q4'])].reset_index(drop=True)
        
        # 数値カラムの変換（カンマ区切り・スペース対応）
        numeric_cols = ['営業収益', '営業利益', '設備投資']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = convert_to_numeric(df[col])
        
        # 営業利益率を丸める
        df['営業利益率'] = np.round(df['営業利益率'], 1)
        
        # ソート用の数値列を追加
        df['ソートキー'] = df['決算年度'].apply(lambda x: sort_quarter_key(x))
        df = df.sort_values(['セグメント', 'ソートキー']).reset_index(drop=True)
        
        return df
    return None

# --- 4. メイン UI ---
st.title("📊 イオン 四半期別セグメント業績分析ダッシュボード")

df_raw = load_segment_data()

if df_raw is not None:
    # --- サイドバー ---
    st.sidebar.header("🔧 分析条件")
    
    # 四半期リスト取得（ソート済み）
    raw_quarters = sorted(df_raw['決算年度'].unique(), key=sort_quarter_key)
    
    # 分析期間選択
    st.sidebar.subheader("📅 分析期間")
    col_start, col_end = st.sidebar.columns(2)
    with col_start:
        start_q = st.selectbox("開始四半期", raw_quarters, index=0)
    with col_end:
        end_q = st.selectbox("終了四半期", raw_quarters, index=len(raw_quarters)-1)
    
    # 期間でフィルタリング
    start_idx = raw_quarters.index(start_q)
    end_idx = raw_quarters.index(end_q)
    if start_idx > end_idx:
        st.sidebar.error("開始四半期は終了四半期より前を選択してください")
        st.stop()
    
    selected_quarters = raw_quarters[start_idx:end_idx+1]
    df_filtered = df_raw[df_raw['決算年度'].isin(selected_quarters)].copy()
    
    # セグメントリスト取得
    segment_list = df_filtered['セグメント'].unique().tolist()
    
    # セグメント詳細分析用の選択
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 セグメント詳細分析")
    selected_segment = st.sidebar.selectbox("セグメントを選択", segment_list)
    
    # 分析期間の表示
    st.sidebar.markdown("---")
    st.sidebar.info(f"📊 分析期間: {start_q} 〜 {end_q}\n\n📈 四半期数: {len(selected_quarters)}")

    # --- タブ構成 ---
    tab_overview, tab_composition, tab_margin, tab_growth, tab_detail = st.tabs([
        "📊 全体概要", "📈 構成比推移", "💹 利益率推移", "🚀 成長率分析", "🔍 セグメント詳細"
    ])

    # --- 色パレット定義 ---
    segment_colors = {
        'GMS事業': '#1f77b4',
        'SM事業': '#ff7f0e',
        'H&W事業': '#2ca02c',
        '総合金融事業': '#d62728',
        'ディベロッパー事業': '#9467bd',
        'サービス・専門店事業': '#8c564b',
        '国際事業': '#e377c2',
        'DS事業': '#7f7f7f',
        'その他': '#bcbd22'
    }

    # ==========================================================
    # タブ1: 全体概要
    # ==========================================================
    with tab_overview:
        st.subheader("セグメント別収益・利益の推移（四半期）")
        
        # 営業収益の積み上げ棒グラフ
        pivot_revenue = df_filtered.pivot_table(
            index='決算年度', columns='セグメント', values='営業収益', aggfunc='sum'
        ).reindex(selected_quarters)
        
        fig1, ax1 = plt.subplots(figsize=(14, 6))
        pivot_revenue.plot(kind='bar', stacked=True, ax=ax1, 
                          color=[segment_colors.get(s, '#333') for s in pivot_revenue.columns])
        ax1.set_title('セグメント別営業収益の推移（積み上げ）', fontsize=14, fontweight='bold')
        ax1.set_xlabel('決算四半期')
        ax1.set_ylabel('営業収益（百万円）')
        ax1.legend(title='セグメント', bbox_to_anchor=(1.02, 1), loc='upper left')
        ax1.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig1)
        
        # 営業利益の積み上げ棒グラフ
        pivot_profit = df_filtered.pivot_table(
            index='決算年度', columns='セグメント', values='営業利益', aggfunc='sum'
        ).reindex(selected_quarters)
        
        fig2, ax2 = plt.subplots(figsize=(14, 6))
        pivot_profit.plot(kind='bar', stacked=True, ax=ax2, 
                         color=[segment_colors.get(s, '#333') for s in pivot_profit.columns])
        ax2.set_title('セグメント別営業利益の推移（積み上げ）', fontsize=14, fontweight='bold')
        ax2.set_xlabel('決算四半期')
        ax2.set_ylabel('営業利益（百万円）')
        ax2.axhline(y=0, color='black', linewidth=0.5)
        ax2.legend(title='セグメント', bbox_to_anchor=(1.02, 1), loc='upper left')
        ax2.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        st.pyplot(fig2)
        
        # 営業収益テーブル
        st.markdown("#### 営業収益一覧（百万円）")
        revenue_table = pivot_revenue.T.copy()
        st.dataframe(revenue_table.style.format("{:,.0f}"), use_container_width=True)
        html_content = get_html_report(revenue_table, f"セグメント別営業収益（{start_q}〜{end_q}）", fig1)
        st.download_button("📥 HTMLでダウンロード（チャート＋テーブル）", html_content, "四半期_営業収益レポート.html", "text/html", key="revenue_html")
        
        # 営業利益テーブル
        st.markdown("#### 営業利益一覧（百万円）")
        profit_table = pivot_profit.T.copy()
        st.dataframe(profit_table.style.format("{:,.0f}"), use_container_width=True)
        html_content2 = get_html_report(profit_table, f"セグメント別営業利益（{start_q}〜{end_q}）", fig2)
        st.download_button("📥 HTMLでダウンロード（チャート＋テーブル）", html_content2, "四半期_営業利益レポート.html", "text/html", key="profit_html")

    # ==========================================================
    # タブ2: 構成比推移
    # ==========================================================
    with tab_composition:
        st.subheader("セグメント別構成比の推移（四半期）")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 営業収益構成比
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            for segment in segment_list:
                seg_data = df_filtered[df_filtered['セグメント'] == segment].sort_values('ソートキー')
                ax3.plot(seg_data['決算年度'], seg_data['営業収益構成比'], 
                        marker='o', label=segment, color=segment_colors.get(segment, '#333'))
            ax3.set_title('営業収益構成比の推移', fontsize=14, fontweight='bold')
            ax3.set_xlabel('決算四半期')
            ax3.set_ylabel('構成比（%）')
            ax3.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
            ax3.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            st.pyplot(fig3)
        
        with col2:
            # 営業利益構成比
            fig4, ax4 = plt.subplots(figsize=(10, 6))
            for segment in segment_list:
                seg_data = df_filtered[df_filtered['セグメント'] == segment].sort_values('ソートキー')
                ax4.plot(seg_data['決算年度'], seg_data['営業利益構成比'], 
                        marker='o', label=segment, color=segment_colors.get(segment, '#333'))
            ax4.set_title('営業利益構成比の推移', fontsize=14, fontweight='bold')
            ax4.set_xlabel('決算四半期')
            ax4.set_ylabel('構成比（%）')
            ax4.axhline(y=0, color='black', linewidth=0.5)
            ax4.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
            ax4.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            st.pyplot(fig4)
        
        # 構成比テーブル（クロス集計）
        st.markdown("#### 営業収益構成比一覧（%）")
        crosstab_rev = pd.crosstab(
            df_filtered['セグメント'], df_filtered['決算年度'], 
            values=df_filtered['営業収益構成比'], aggfunc='sum'
        ).reindex(columns=selected_quarters)
        crosstab_rev = crosstab_rev.sort_values(selected_quarters[-1], ascending=False)
        st.dataframe(crosstab_rev.style.format("{:.1f}"), use_container_width=True)
        html_comp1 = get_html_report(crosstab_rev, f"営業収益構成比（{start_q}〜{end_q}）", fig3)
        st.download_button("📥 HTMLでダウンロード（チャート＋テーブル）", html_comp1, "四半期_営業収益構成比レポート.html", "text/html", key="comp_rev_html")
        
        st.markdown("#### 営業利益構成比一覧（%）")
        crosstab_profit = pd.crosstab(
            df_filtered['セグメント'], df_filtered['決算年度'], 
            values=df_filtered['営業利益構成比'], aggfunc='sum'
        ).reindex(columns=selected_quarters)
        crosstab_profit = crosstab_profit.sort_values(selected_quarters[-1], ascending=False)
        st.dataframe(crosstab_profit.style.format("{:.1f}"), use_container_width=True)
        html_comp2 = get_html_report(crosstab_profit, f"営業利益構成比（{start_q}〜{end_q}）", fig4)
        st.download_button("📥 HTMLでダウンロード（チャート＋テーブル）", html_comp2, "四半期_営業利益構成比レポート.html", "text/html", key="comp_profit_html")

    # ==========================================================
    # タブ3: 利益率推移
    # ==========================================================
    with tab_margin:
        st.subheader("セグメント別営業利益率の推移（四半期）")
        
        fig6, ax6 = plt.subplots(figsize=(14, 7))
        for segment in segment_list:
            seg_data = df_filtered[df_filtered['セグメント'] == segment].sort_values('ソートキー')
            ax6.plot(seg_data['決算年度'], seg_data['営業利益率'], 
                    marker='o', label=segment, color=segment_colors.get(segment, '#333'), linewidth=2)
        ax6.set_title('セグメント別営業利益率の推移', fontsize=14, fontweight='bold')
        ax6.set_xlabel('決算四半期')
        ax6.set_ylabel('営業利益率（%）')
        ax6.axhline(y=0, color='black', linewidth=0.5)
        ax6.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        ax6.tick_params(axis='x', rotation=45)
        ax6.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig6)
        
        # 営業利益率テーブル
        st.markdown("#### 営業利益率一覧（%）")
        crosstab_margin = pd.crosstab(
            df_filtered['セグメント'], df_filtered['決算年度'], 
            values=df_filtered['営業利益率'], aggfunc='sum'
        ).reindex(columns=selected_quarters)
        crosstab_margin = crosstab_margin.sort_values(selected_quarters[-1], ascending=False)
        st.dataframe(crosstab_margin.style.format("{:.1f}"), use_container_width=True)
        
        html_content3 = get_html_report(crosstab_margin, f"セグメント別営業利益率（{start_q}〜{end_q}）", fig6)
        st.download_button("📥 HTMLでダウンロード（チャート＋テーブル）", html_content3, "四半期_営業利益率レポート.html", "text/html", key="margin_html")

    # ==========================================================
    # タブ4: 成長率分析
    # ==========================================================
    with tab_growth:
        st.subheader(f"セグメント別営業収益成長率（{start_q}基準）")
        
        # 成長率計算（start_q基準、「その他」を除外）
        growth_segments = [s for s in segment_list if s != 'その他']
        
        growth_df = pd.DataFrame()
        for segment in growth_segments:
            seg_data = df_filtered[df_filtered['セグメント'] == segment].sort_values('ソートキー').copy()
            if not seg_data.empty:
                base_value = seg_data.iloc[0]['営業収益']
                if base_value > 0:
                    seg_data[f'営業収益成長率(対{start_q})'] = np.round(seg_data['営業収益'] / base_value, 2)
                    growth_df = pd.concat([growth_df, seg_data], axis=0)
        
        growth_df = growth_df.reset_index(drop=True)
        
        if not growth_df.empty:
            fig7, ax7 = plt.subplots(figsize=(14, 7))
            for segment in growth_segments:
                seg_data = growth_df[growth_df['セグメント'] == segment].sort_values('ソートキー')
                if not seg_data.empty:
                    ax7.plot(seg_data['決算年度'], seg_data[f'営業収益成長率(対{start_q})'], 
                            marker='o', label=segment, color=segment_colors.get(segment, '#333'), linewidth=2)
            ax7.set_title(f'セグメント別営業収益成長率（{start_q}=1.00）', fontsize=14, fontweight='bold')
            ax7.set_xlabel('決算四半期')
            ax7.set_ylabel('成長率（倍）')
            ax7.axhline(y=1.0, color='black', linewidth=0.5, linestyle='--')
            ax7.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
            ax7.tick_params(axis='x', rotation=45)
            ax7.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig7)
            
            # 成長率テーブル
            st.markdown(f"#### 営業収益成長率一覧（{start_q}=1.00）")
            crosstab_growth = pd.crosstab(
                growth_df['セグメント'], growth_df['決算年度'], 
                values=growth_df[f'営業収益成長率(対{start_q})'], aggfunc='sum'
            ).reindex(columns=selected_quarters)
            crosstab_growth = crosstab_growth.sort_values(selected_quarters[-1], ascending=False)
            st.dataframe(crosstab_growth.style.format("{:.2f}"), use_container_width=True)
            
            html_growth = get_html_report(crosstab_growth, f"セグメント別営業収益成長率（{start_q}基準）", fig7)
            st.download_button("📥 HTMLでダウンロード（チャート＋テーブル）", html_growth, "四半期_成長率レポート.html", "text/html", key="growth_html")
        else:
            st.warning("成長率を計算できるデータがありません。")

    # ==========================================================
    # タブ5: セグメント詳細
    # ==========================================================
    with tab_detail:
        st.subheader(f"🔍 {selected_segment} - 詳細分析（四半期）")
        
        # セグメントデータ抽出
        seg_detail = df_filtered[df_filtered['セグメント'] == selected_segment].sort_values('ソートキー').copy()
        
        if not seg_detail.empty:
            # 成長率計算
            base_revenue = seg_detail.iloc[0]['営業収益']
            if base_revenue > 0:
                seg_detail['営業収益成長率'] = np.round(seg_detail['営業収益'] / base_revenue, 2)
            else:
                seg_detail['営業収益成長率'] = 0
            
            # 前期比成長率計算
            seg_detail['営業収益対前期成長率'] = np.round(
                (seg_detail['営業収益'] / seg_detail['営業収益'].shift(1) - 1) * 100, 1
            )
            seg_detail.loc[seg_detail.index[0], '営業収益対前期成長率'] = np.nan
            
            quarters_display = seg_detail['決算年度'].tolist()
            
            # 2x2サブプロット
            fig8, axs = plt.subplots(2, 2, figsize=(14, 10))
            
            # 営業収益
            axs[0, 0].bar(quarters_display, seg_detail['営業収益'], color='skyblue')
            axs[0, 0].set_title('営業収益', fontsize=12, fontweight='bold')
            axs[0, 0].set_ylabel('金額（百万円）')
            axs[0, 0].tick_params(axis='x', rotation=90)
            
            # 営業利益
            colors = ['orange' if v >= 0 else 'red' for v in seg_detail['営業利益']]
            axs[0, 1].bar(quarters_display, seg_detail['営業利益'], color=colors)
            axs[0, 1].set_title('営業利益', fontsize=12, fontweight='bold')
            axs[0, 1].set_ylabel('金額（百万円）')
            axs[0, 1].axhline(y=0, color='black', linewidth=0.5)
            axs[0, 1].tick_params(axis='x', rotation=90)
            
            # 営業収益成長率
            axs[1, 0].plot(quarters_display, seg_detail['営業収益成長率'], marker='o', color='green', linewidth=2)
            axs[1, 0].set_title(f'営業収益成長率（{start_q}=1.00）', fontsize=12, fontweight='bold')
            axs[1, 0].set_ylabel('成長率（倍）')
            axs[1, 0].axhline(y=1.0, color='black', linewidth=0.5, linestyle='--')
            axs[1, 0].tick_params(axis='x', rotation=90)
            axs[1, 0].grid(True, alpha=0.3)
            
            # 営業利益率
            axs[1, 1].plot(quarters_display, seg_detail['営業利益率'], marker='o', color='purple', linewidth=2)
            axs[1, 1].set_title('営業利益率', fontsize=12, fontweight='bold')
            axs[1, 1].set_ylabel('利益率（%）')
            axs[1, 1].axhline(y=0, color='black', linewidth=0.5)
            axs[1, 1].tick_params(axis='x', rotation=90)
            axs[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig8)
            
            # 詳細テーブル
            st.markdown("#### 業績推移テーブル")
            display_cols = ['決算年度', '営業収益', '営業利益', '営業収益成長率', '営業収益対前期成長率', '営業利益率']
            display_df = seg_detail[display_cols].copy()
            display_df = display_df.set_index('決算年度')
            
            format_dict = {
                '営業収益': '{:,.0f}',
                '営業利益': '{:,.0f}',
                '営業収益成長率': '{:.2f}',
                '営業収益対前期成長率': '{:.1f}',
                '営業利益率': '{:.1f}'
            }
            st.dataframe(display_df.style.format(format_dict), use_container_width=True)
            
            # 構成比テーブル（横持ち・バーチャート風スタイル）
            st.markdown("#### 構成比推移")
            comp_df = seg_detail[['決算年度', '営業収益構成比', '営業利益構成比']].copy()
            comp_df = comp_df.set_index('決算年度').T
            
            st.dataframe(
                comp_df.style.format("{:.1f}%").bar(subset=comp_df.columns, color='skyblue', vmin=0),
                use_container_width=True
            )
            
            html_content4 = get_html_report(display_df, f"{selected_segment} - 四半期業績推移（{start_q}〜{end_q}）", fig8)
            st.download_button(f"📥 HTMLでダウンロード（チャート＋テーブル）", html_content4, f"{selected_segment}_四半期詳細レポート.html", "text/html", key="detail_html")
        
        else:
            st.warning("選択されたセグメントのデータが見つかりません。")

else:
    st.error("データファイルが見つかりません。リポジトリの data/ フォルダを確認してください。")

# --- フッター ---
st.divider()
st.markdown("""
<div style="text-align: center; color: #888; font-size: 12px;">
    📊 イオン 四半期別セグメント業績分析ダッシュボード | Powered by Streamlit
</div>
""", unsafe_allow_html=True)
