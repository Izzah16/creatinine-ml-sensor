# Creatinine ML Sensor

A desktop application for real-time quantification of serum creatinine using electrochemical sensing (DPV, CV, EIS) combined with machine learning.  
Built with **Python 3.8**, **PyQt5**, and a **Random Forest regression model**, this system integrates data acquisition, feature extraction, prediction, and visualization in a standalone GUI.

---

## Features
- Direct device integration with **PalmSens potentiostats** (via PalmSens SDK).
- Real-time plotting of **DPV, CV, and EIS** scans.
- Automated **feature extraction** from DPV signals.
- Machine learning model (**Random Forest**) to predict creatinine concentration.
- **GUI interface** for calibration, prediction, visualization, and exporting results.
- Works fully **offline**, runs on any Windows laptop (no internet required).

---

## Installation

### Prerequisites
- Python **3.8** (recommended to use a virtual environment).
- PalmSens SDK and drivers installed (required for hardware connectivity).

### Steps
1. Clone this repository:
   ```bash
   git clone https://github.com/Izzah16/creatinine-ml-sensor.git
   cd creatinine-ml-sensor
   ```

2. Create and activate a virtual environment:
   ```bash
   python3.8 -m venv venv
   source venv/bin/activate       # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Quick workflow

1. **Run GUI**  
   ```bash
   python main.app     # or python main.py if your script is named main.py
   ```

2. **Connect device**  
   - Click **Connect Device** (PalmSens instrument must be attached).
   - If successful, status bar will show *Connected to device*.

3. **Acquire DPV**  
   - Choose **DPV** .  
   - Set parameters or keep defaults.  
   - Click **Start Measurement** (real acquisition) or **Generate Test Scan** (offline demo).

4. **Save / load data (optional)**  
   - Save: File → *Save Data* → `*.csv`  
   - Load: File → *Open Data* (expects CSV with `Voltage (V),Current (A)`).

5. **Predict**  
    
   - Click **Show Result** → ML prediction of creatinine concentration (lab samples).  
   - *Real Sample Results* applies scaling for clinical serum validation *5 for serum in our case.

6. **Export plot**  
   - File → *Export Plot* → save as PNG/PDF.

---

## Example input format

Saved data should be in CSV:

```csv
Voltage (V),Current (A)
-0.500170,18.927914
-0.489963,15.455929
-0.479755,13.495940
...
```

---
## Screenshots

**Main GUI Window**
![Main Window](data/gui_main.png)

**Prediction Result Popup**
![Prediction Result](data/prediction_example.png)

## Extending the pipeline to a new analyte

This workflow can be adapted for other metabolites by modifying only the MIP layer and retraining the ML model.

1. **Re-design MIP**  
   - Replace the creatinine template with the new analyte during polymerization.

2. **Data collection**  
   - Acquire DPV scans across the relevant concentration range.  
   - Use ≥3 independent MIP batches and multiple replicates per concentration.  
   - Include both buffer and biological matrices.

3. **Feature extraction**  
   - Use existing features: peak current, peak potential, area under curve, mean, std, skew.  
   - Add extra descriptors if needed (e.g., peak width, baseline slope).

4. **Model training**  
   - Stratified train–test split across batches.  
   - Cross-validation (≥5-fold).  
   - Evaluate with R², MAE, RMSE, LOD/LOQ.  
   - Save final model as `rf_model.pkl`.

5. **Clinical validation**  
   - Test on independent samples, compare with reference assays.  
   - Report agreement using regression + error analysis.

6. **Deployment**  
   - Update GUI analyte name and load the new `rf_model.pkl`.  
   - Document retraining process in README.

---

## Requirements

See `requirements.txt` for full dependencies (with pinned versions):

```txt
numpy==1.24.4
pandas==2.0.3
scipy==1.10.1
scikit-learn==1.3.0
matplotlib==3.7.2
joblib==1.3.2
pyqt5==5.15.9
pythonnet==3.0.1
```

---

## Citation

If you use this software, please cite:

**"An Automated, Machine Learning Integrated Platform for Real-Time Quantification of Serum Creatinine in Clinical Samples"**  
DOI: *(to be updated after acceptance)*

---

## License

Open-source for research use. Please acknowledge the authors in derivative works.
