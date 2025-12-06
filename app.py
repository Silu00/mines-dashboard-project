import streamlit as st
import pandas as pd
import numpy as np
import gspread
from google.oauth2.service_account import Credentials
from scipy.stats import zscore, t
import plotly.graph_objects as go
from fpdf import FPDF
import os
from datetime import datetime

st.set_page_config(
    page_title="Weyland-Yutani Operations",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

def local_css():
    st.markdown("""
    <style>
        .reportview-container { background: #f0f2f6; }
        div.stButton > button {
            background-color: #27ae60; color: white; border-radius: 4px; border: none; font-weight: bold; transition: 0.3s;
        }
        div.stButton > button:hover { background-color: #1e8449; border: 1px solid white; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data(ttl=60)
def load_data():
    scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]

    if os.path.exists("credentials.json"):
        credentials = Credentials.from_service_account_file("credentials.json", scopes=scopes)
    elif "gcp_service_account" in st.secrets:
        creds_dict = dict(st.secrets["gcp_service_account"])
        credentials = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    else:
        st.error("JSON credentials file not found.")
        return None

    gc = gspread.authorize(credentials)

    target_sheet_name = "Mining Ops Simulator"

    try:
        sh = gc.open(target_sheet_name)
    except gspread.exceptions.SpreadsheetNotFound:
        st.error(f"CRITICAL ERROR: Sheet not found '{target_sheet_name}'.")
        return None

    try:
        worksheet = sh.worksheet("Output")
        raw_data = worksheet.get_all_values()
    except gspread.exceptions.WorksheetNotFound:
        st.error("Sheet cannot be found.")
        return None

    if not raw_data:
        return None

    headers = raw_data.pop(0)
    df = pd.DataFrame(raw_data, columns=headers)

    df = df.loc[:, df.columns != '']

    df = df.replace(r'^\s*$', np.nan, regex=True)
    df = df.dropna(how='all')

    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])

    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    return df

def detect_iqr(input_df, column, k=1.5):
    q1, q3 = input_df[column].quantile([0.25, 0.75])
    iqr_value = q3 - q1
    return ~input_df[column].between(q1 - k * iqr_value, q3 + k * iqr_value)

def detect_zscore(input_df, column, threshold=3.0):
    return np.abs(zscore(input_df[column], nan_policy='omit')) > threshold

def detect_moving_average(input_df, column, window=7, threshold_pct=0.2):
    ma = input_df[column].rolling(window=window).mean()
    ma = ma.replace(0, np.nan)
    return (np.abs((input_df[column] - ma) / ma) > threshold_pct).fillna(False)

def detect_grubbs(input_df, column, alpha=0.05):
    data = input_df[column].values
    n = len(data)
    if n < 3: return pd.Series([False] * n, index=input_df.index)
    mean, std = np.mean(data), np.std(data, ddof=1)
    if std == 0: return pd.Series([False] * n, index=input_df.index)
    g_calculated = np.abs(data - mean) / std
    tc = t.ppf(1 - alpha / (2 * n), n - 2)
    g_critical = ((n - 1) * np.sqrt(np.square(tc))) / (np.sqrt(n * (n - 2 + np.square(tc))))
    return pd.Series(g_calculated > g_critical, index=input_df.index)

class PDFReport(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, 'Weyland-Yutani Corp', 0, 0, 'R')
        self.ln(15)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128)
        self.cell(0, 10, f'Confidential - Page {self.page_no()}', 0, 0, 'C')

    def title_page(self, title, subtitle):
        self.add_page()
        self.set_fill_color(30, 42, 54)
        self.rect(0, 0, 210, 297, 'F')
        self.set_y(100)
        self.set_font('Helvetica', 'B', 24)
        self.set_text_color(255, 255, 255)
        self.cell(0, 15, title, 0, 1, 'C')
        self.set_font('Helvetica', '', 16)
        self.set_text_color(243, 156, 18)
        self.cell(0, 10, subtitle, 0, 1, 'C')
        self.set_y(250)
        self.set_font('Helvetica', '', 10)
        self.set_text_color(200, 200, 200)
        self.cell(0, 10, f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 0, 1, 'C')

    def chapter_title(self, title):
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(30, 42, 54)
        self.cell(0, 10, title, 0, 1, 'L')
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

    def chapter_body(self, body):
        self.set_font('Helvetica', '', 11)
        self.set_text_color(0)
        self.multi_cell(0, 7, body)
        self.ln()

    def add_table_header(self, headers):
        self.set_font('Helvetica', 'B', 10)
        self.set_fill_color(243, 156, 18)
        self.set_text_color(255)
        for h in headers:
            self.cell(63, 8, h, 1, 0, 'C', 1)
        self.ln()

def create_chart_figure(input_df, selected_col, anomalies_series, chart_kind, poly_degree, title_suffix=""):
    fig = go.Figure()

    date_col = input_df.columns[0]

    if chart_kind == "Line":
        fig.add_trace(go.Scatter(x=input_df[date_col], y=input_df[selected_col], mode='lines', name='Output', line=dict(color='#2c3e50')))
    elif chart_kind == "Bar":
        fig.add_trace(go.Bar(x=input_df[date_col], y=input_df[selected_col], name='Output', marker_color=dict(color='#2c3e50')))
    elif chart_kind == "Area (Stacked)":
        fig.add_trace(
            go.Scatter(x=input_df[date_col], y=input_df[selected_col], mode='lines', fill='tozeroy', name='Output', line=dict(color='#2c3e50')))

    anomaly_points = input_df[anomalies_series]
    if not anomaly_points.empty:
        fig.add_trace(
            go.Scatter(x=anomaly_points[date_col], y=anomaly_points[selected_col], mode='markers', name='Anomaly', marker=dict(color='#c0392b', size=10, symbol='x')))

    x_numeric = np.arange(len(input_df))
    y_numeric = input_df[selected_col].values
    if len(y_numeric) > 0:
        z = np.polyfit(x_numeric, y_numeric, poly_degree)
        p = np.poly1d(z)
        fig.add_trace(go.Scatter(x=input_df[date_col], y=p(x_numeric), mode='lines', name='Trend', line=dict(color='#f39c12', dash='dash', width=2)))

    fig.update_layout(
        title=f"{selected_col} | {title_suffix}",
        xaxis_title="Date", yaxis_title="Output",
        template="simple_white", height=450, margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

def main():
    local_css()

    with st.sidebar:
        st.markdown("🏭 Weyland-Yutani")
        st.caption("Mining Operations Division")
        st.divider()
        if st.button("🔄 Reload Data"):
            load_data.clear()
            st.rerun()

        st.subheader("⚙️ Analysis Parameters")
        all_params = {}
        with st.expander("IQR Rule", expanded=True):
            all_params['IQR Rule'] = {'k': st.slider("Multiplier (k)", 1.0, 4.0, 1.5)}
        with st.expander("Z-Score"):
            all_params['Z-Score'] = {'threshold': st.slider("Threshold", 1.0, 5.0, 3.0)}
        with st.expander("Moving Average"):
            all_params['Moving Average'] = {
                'window': st.slider("MA Window", 2, 30, 7),
                'threshold': st.slider("Deviation %", 0.05, 1.0, 0.2)
            }
        with st.expander("Grubbs Test"):
            all_params['Grubbs Test'] = {'alpha': st.slider("Alpha", 0.01, 0.10, 0.05)}

        st.subheader("📊 Chart Settings")
        chart_type = st.radio("Chart Type", ["Line", "Bar", "Area (Stacked)"])
        trend_degree = st.slider("Trendline Degree", 1, 4, 1)

    col_title, col_date = st.columns([3, 1])
    with col_title:
        st.title("Mining Operations Dashboard")
    with col_date:
        st.markdown(f"**Date:** {datetime.now().strftime('%Y-%m-%d')}")
        st.success("Mainframe Connected")

    df = load_data()

    if df is not None:
        date_col = df.columns[0]
        mine_columns = df.columns[1:].tolist()

        df['Total Output'] = df[mine_columns].sum(axis=1)
        options_list = ['Total Output'] + mine_columns

        st.info("Select Sector for Deep Dive Analysis:")
        selected_option = st.selectbox("Select Sector", options_list, label_visibility="collapsed")

        analysis_df = df[[date_col, selected_option]].copy()

        results = {}
        results['IQR Rule'] = detect_iqr(analysis_df, selected_option, k=all_params['IQR Rule']['k'])
        results['Z-Score'] = detect_zscore(analysis_df, selected_option, threshold=all_params['Z-Score']['threshold'])
        results['Moving Average'] = detect_moving_average(analysis_df, selected_option, window=all_params['Moving Average']['window'], threshold_pct=all_params['Moving Average']['threshold'])
        results['Grubbs Test'] = detect_grubbs(analysis_df, selected_option, alpha=all_params['Grubbs Test']['alpha'])

        st.divider()
        st.markdown("📈 Key Performance Indicators")
        ds = analysis_df[selected_option]

        if len(ds) > 0:
            mean_val = ds.mean()
            median_val = ds.median()
            std_val = ds.std()
            iqr_val = ds.quantile(0.75) - ds.quantile(0.25)
        else:
            mean_val = median_val = std_val = iqr_val = 0

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Mean Output", f"{mean_val:.2f}")
        k2.metric("Median", f"{median_val:.2f}")
        k3.metric("Std Dev", f"{std_val:.2f}")
        k4.metric("IQR Range", f"{iqr_val:.2f}")

        st.divider()
        st.markdown(f"📊 Test Results for: {selected_option}")
        tabs = st.tabs(list(results.keys()))

        for i, (method_name, anom_series) in enumerate(results.items()):
            with tabs[i]:
                fig = create_chart_figure(analysis_df, selected_option, anom_series, chart_type, trend_degree, title_suffix=method_name)
                st.plotly_chart(fig, width="stretch")

                count = anom_series.sum()
                if count > 0:
                    st.warning(f"⚠️ {method_name}: Detected {count} anomalies.")
                    with st.expander("Show Anomaly Log"):
                        temp_df = analysis_df.copy()
                        temp_df['Is_Anomaly'] = anom_series

                        st.dataframe(temp_df[temp_df['Is_Anomaly'] == True], width="stretch")
                else:
                    st.success(f"✅ {method_name}: No anomalies detected.")

        st.divider()
        st.subheader("📄 Executive Reporting")
        if st.button("Generate PDF Report"):
            with st.spinner("Generating PDF..."):
                try:
                    pdf = PDFReport()
                    pdf.title_page("Mining Operations Audit", f"Sector: {selected_option}")

                    pdf.add_page()
                    pdf.chapter_title("Global Statistical Summary")
                    pdf.set_font('Helvetica', '', 12)
                    pdf.cell(50, 10, f"Mean: {mean_val:.2f}", 1)
                    pdf.cell(50, 10, f"Median: {median_val:.2f}", 1)
                    pdf.ln(10)
                    pdf.cell(50, 10, f"Std Dev: {std_val:.2f}", 1)
                    pdf.cell(50, 10, f"IQR: {iqr_val:.2f}", 1)
                    pdf.ln(20)

                    temp_files = []

                    for method_name, anom_series in results.items():
                        pdf.add_page()
                        pdf.chapter_title(f"Test Method: {method_name}")

                        fig = create_chart_figure(analysis_df, selected_option, anom_series, chart_type, trend_degree,
                                                  title_suffix=method_name)
                        f_name = f"temp_{method_name.replace(' ', '_')}.png"

                        fig.write_image(f_name, width=1000, height=500, scale=2)
                        temp_files.append(f_name)

                        pdf.image(f_name, x=10, w=190)
                        pdf.ln(10)

                        count = anom_series.sum()
                        pdf.chapter_body(f"Anomalies detected: {count}")

                        if count > 0:
                            pdf.add_table_header(["Date", "Value", "Status"])
                            pdf.set_font('Helvetica', '', 10)
                            pdf.set_text_color(0)

                            anoms = analysis_df[anom_series]
                            for _, row in anoms.tail(20).iterrows():
                                date_str = row[date_col].strftime('%Y-%m-%d')
                                val = row[selected_option]
                                status = "SPIKE" if val > mean_val else "DROP"

                                pdf.cell(63, 8, date_str, 1)
                                pdf.cell(63, 8, f"{val:.2f}", 1)
                                pdf.cell(63, 8, status, 1)
                                pdf.ln()
                        else:
                            pdf.chapter_body("Operations nominal.")

                    pdf_file = f"Weyland_Report_{selected_option}.pdf"
                    pdf.output(pdf_file)

                    with open(pdf_file, "rb") as f:
                        st.success("Report Ready.")
                        st.download_button("⬇️ Download PDF", f, file_name=pdf_file, mime="application/pdf")

                    for f in temp_files:
                        if os.path.exists(f): os.remove(f)

                except Exception as e:
                    st.error(f"Error generating PDF: {e}")

    else:
        st.warning("Waiting for the connection...")

if __name__ == "__main__":
    main()
