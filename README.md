# 🛰️ Orbital Trend Detective: Tri-Sensor Edge-AI Earth System Engine

**Team:** Fluid Edge Intelligence  
**Event:** 2026 NASA International Space Apps Challenge  
**Challenge:** *Be An Earth System Trend Detective!*  

---

## 🌍 Project Overview
**Orbital Trend Detective** solves two fundamental bottlenecks in Earth observation: separating genuine multi-year climate shifts from repeating seasonal weather noise, and overcoming CubeSat-to-ground radio downlink constraints.

Analyzing **16 years (`2010–2026`, `189` monthly orbital cycles)** of multi-satellite telemetry from **NASA Giovanni** across three distinct spatial regimes—**Global Planetary (`-180° to +180°`)**, **Greater Hyderabad Urban-Industrial Airshed (`78°E–79°E, 17°N–18°N`)**, and the **Amazon Basin (`63°W–61°W, 4°S–2°S`)**—our hybrid system pairs **econometric cause-and-effect testing** with an **8-bit integer (`INT8`) quantized CubeSat flight-computer simulator (`0.28 KB`)** that reduces satellite downlink bandwidth by **96.3% to 98.4%**.

---

## ⚙️ System Architecture & Methodology

### 1. Tri-Sensor Seasonal & Monsoon Noise Subtraction (`STL + GPM`)
Raw satellite measurements bounce up and down every year due to predictable summer heatwaves, winter snow cover, and monsoon rain washout. Our preprocessing pipeline isolates true multi-year anomalies in two steps:
* **Seasonal-Trend Decomposition using LOESS (`STL`):** Subtracts repeating 12-month seasonal cycles from Aerosol Optical Depth (`AOD`), Land Surface Temperature (`LST`), and Precipitation.
* **Monsoon Rainfall Washout Adjustment:** Regresses deseasonalized aerosol loads against **NASA GPM IMERG** rainfall anomalies (`slope * precip_stl`) so heavy rain scrubbing is never mistaken for falling emissions.

### 2. Econometric Trend & Causality Engine
Rather than relying on visual guesswork, our statistical verification layer tests every region quantitatively:
* **Augmented Dickey-Fuller (`ADF`) Test:** Verifies time-series stationarity before modeling.
* **Mann-Kendall Rank Test (`kendalltau`):** Proves non-parametric multi-year trend significance even in the presence of sudden wildfire or dust-storm spikes.
* **Granger Causality F-Tests:** Evaluates directional lag relationships (`1–6 months`) between atmospheric aerosols and local surface temperature shifts.

### 3. On-Board `INT8` CubeSat Edge-AI Simulator
Small satellites (CubeSats) cannot beam every gigabyte of orbital imagery back to Earth. We compress a 3-input neural anomaly detector (`AOD`, `LST`, `Precipitation`) into **8-bit signed integers (`INT8`)**:
* **Memory Footprint:** **`0.28 KB`** (`75%` smaller than `FP32` floating-point weights).
* **Pass Inference Latency:** **`~0.0066 – 0.0106 ms`** per orbital pass.
* **Downlink Bandwidth Reduction:** Transmits only top-decile compound heat-and-aerosol anomaly passes, cutting routine downlink traffic by **`96.3%` to `98.4%`**.

---

## 📊 Key Live NASA Telemetry Findings (`2010–2026`)

| Live NASA Telemetry Node | Spatial Bounding Box | Downlink Saved | Extreme Passes Flagged | On-Board `INT8` Latency | Granger Causality Result |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Global Planetary Composite** | `-180°, -90°, 180°, 90` | **98.4%** | `3 / 189` passes | `0.0066 ms` | `p = 0.3833` (Lag 2m) |
| **Greater Hyderabad Airshed** | `78.0°, 17.0°, 79.0°, 18.0°` | **97.4%** | `5 / 189` passes | `0.0102 ms` | `p = 0.5242` (Lag 2m) |
| **Amazon Basin Canopy** | `-63.0°, -4.0°, -61.0°, -2.0°` | **96.3%** | `7 / 189` passes | `0.0106 ms` | `p = 0.7295` (Lag 1m) |

* **Global Warming Significance:** Live global surface temperature exhibits a highly significant upward trend (`Kendall tau = 0.554, p < 0.0001`), while aggregate global aerosols decline gently (`tau = -0.192, p < 0.0001`).
* **Regional Contradictions:** Planetary averages dilute localized compound heat-and-aerosol events (`5/189` extreme passes in Hyderabad and `7/189` in the Amazon Basin), demonstrating why regional edge-AI filtering is essential for local climate warnings.

---

## 📂 Repository Structure
```text
orbital-trend-detective/
├── app.py                  # Main Streamlit dashboard, STL/GPM filter, Granger tests & INT8 HUD
├── requirements.txt        # Python dependencies for local and Streamlit Cloud deployment
├── global_aod.csv          # NASA MODIS MOD08_M3 Global Aerosol Optical Depth (2010–2026)
├── global_lst.csv          # NASA MERRA-2 M2TMNXLND Global Land Surface Temperature (2010–2026)
├── global_gpm.csv          # NASA GPM_3IMERGM Global Monthly Precipitation (2010–2026)
├── hyd_aod.csv             # NASA MODIS MOD08_M3 Greater Hyderabad AOD (2010–2026)
├── hyd_lst.csv             # NASA MERRA-2 M2TMNXLND Greater Hyderabad Surface Temp (2010–2026)
├── hyd_gpm.csv             # NASA GPM_3IMERGM Greater Hyderabad Precipitation (2010–2026)
├── amazon_aod.csv          # NASA MODIS MOD08_M3 Amazon Basin AOD (2010–2026)
├── amazon_lst.csv          # NASA MERRA-2 M2TMNXLND Amazon Basin Surface Temp (2010–2026)
├── amazon_gpm.csv          # NASA GPM_3IMERGM Amazon Basin Precipitation (2010–2026)
└── README.md               # Project documentation, methodology & NASA data citations
```

---

## 🚀 How to Run Locally
1. **Clone the repository:**
   ```bash
   git clone https://github.com<your-username>/orbital-trend-detective.git
   cd orbital-trend-detective
   ```
2. **Install required Python libraries:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Launch the interactive dashboard:**
   ```bash
   streamlit run app.py
   ```
   *(All three live NASA regional datasets load automatically from the local CSVs upon startup, and custom NASA Giovanni CSVs can also be uploaded via the sidebar).*

---

## 🛰️ NASA Data & Tool Citations
* **NASA Giovanni Portal:** Acker, J. G., & Leptoukh, G. (2007). *Online Analysis Enhances Use of NASA Earth Science Data*. Eos, Transactions American Geophysical Union, 88(2), 14–17. `https://nasa.gov`
* **Aerosol Optical Depth (`MOD08_M3 v6.1`):** Platnick, S., et al. (2015). *MODIS Atmosphere L3 Monthly Product (0.55 Micron Combined Dark Target and Deep Blue AOD)*. NASA MODIS Adaptive Processing System, Goddard Space Flight Center.
* **Land Surface Temperature (`M2TMNXLND v5.12.4`):** Global Modeling and Assimilation Office (GMAO). *MERRA-2 tavgM_2d_lnd_Nx: 2d, Monthly Mean, Time-Averaged, Single-Level, Assimilation, Land Surface Diagnostics*. NASA Goddard Earth Sciences Data and Information Services Center (GES DISC).
* **Precipitation (`GPM_3IMERGM v07`):** Huffman, G. J., et al. (2023). *GPM IMERG Final Precipitation L3 1 Month 0.1 degree x 0.1 degree V07*. NASA GES DISC.

---

## 🤖 AI Assistance Disclosure & License
* **AI Tool Disclosure:** Generative AI assistance was used during development to structure code documentation, debug CSV parsing edge cases, and format technical descriptions. All NASA satellite data acquisition, statistical testing, and architectural decisions were executed by **Team Fluid Edge Intelligence**.
* **License:** Released under the **MIT License**.
* "Built by Team Fluid Edge Intelligence — Architecture, Tri-Sensor Analytics & Edge-AI Modeling by Parikshith; Repository & Cloud Deployment by Varun"
* 
* 
