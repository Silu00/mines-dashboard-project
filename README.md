# ⛏️ Weyland-Yutani Mining Operations Dashboard

A comprehensive data analysis solution consisting of a realistic data generator (built entirely within Google Sheets) and an interactive analytical dashboard built with Streamlit.

This project simulates and analyzes the daily resource extraction of the **Weyland-Yutani Corporation**, providing insights into production trends, seasonality, and anomaly detection.

## 🔗 Live Demo
**Dashboard:** [https://mines-dashboard-project.streamlit.app/](https://mines-dashboard-project.streamlit.app/)

## 📝 Project Overview

The project is divided into two distinct components:

### 1. The Data Generator (Google Spreadsheet)
A complex simulation engine built exclusively using **Spreadsheet Formulas** (no scripts). It generates realistic time-series data rather than simple white noise.

**Key capabilities:**
* **Customizable Parameters:** Users can adjust mine names, date ranges, and base values.
* **Dynamic Distributions:** Supports both **Uniform** and **Normal** distributions with dynamic parameter updates.
* **Smoothing & Correlation:** Implements smoothing algorithms to simulate realistic day-to-day correlations.
* **Seasonality:** Day-of-week factors (e.g., lower output on Sundays) to introduce periodic patterns.
* **Trend Injection:** Adjustable overall production trends (linear/polynomial).
* **Event Simulation:** A sophisticated anomaly generator that creates spikes or drops based on probability, duration, and magnitude (using bell-shaped curves).

👉 **[View the Data Source Spreadsheet Here](https://docs.google.com/spreadsheets/d/1kRW1bXc7XtgmX7-Vl8IY5DE49psouSYi0oCeuDiokns/edit?usp=drive_link)**

### 2. The Web Dashboard (Streamlit)
A Python-based application that consumes data from the spreadsheet to perform advanced statistical analysis and visualization.

**Features:**
* **Statistical Overview:** Calculates Mean, Standard Deviation, Median, and Interquartile Range (IQR) for individual mines and total output.
* **Advanced Anomaly Detection:**
    * **IQR Rule:** Detects outliers based on statistical dispersion.
    * **Z-Score:** Identifies data points far from the mean.
    * **Moving Average:** Flags deviations based on percentage distance from the trend.
    * **Grubbs' Test:** Rigorous statistical test for outliers in univariate datasets.
* **Interactive Visualization:**
    * Line, Bar, and Stacked charts.
    * Dynamic Trendlines (Polynomial degrees 1-4).
    * Visual highlighting of detected anomalies.

## 🛠️ Technology Stack

* **Python 3.10+**
* **Streamlit:** For the web interface.
* **Pandas & NumPy:** For data manipulation and statistical calculations.
* **Plotly / Altair:** For interactive charting.
* **Scipy:** For advanced statistical tests (e.g., Grubbs' test).
* **Google Sheets API:** For fetching live data from the generator.

## 📊 Anomaly Detection Logic

The dashboard allows users to fine-tune the sensitivity of anomaly detection algorithms:

* **IQR Multiplier:** Standard is 1.5, but can be adjusted to strict (1.0) or loose (3.0) detection.
* **Z-Score Threshold:** Defines how many standard deviations away from the mean constitute an anomaly (typically >3).
* **MA Window:** Adjusts the window size for the Moving Average comparison.
