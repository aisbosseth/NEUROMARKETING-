from pathlib import Path


!pip install fpdf chardet

# Paso 1: Importar librerías necesarias
from google.colab import files
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from fpdf import FPDF
import chardet

# Paso 2: Subir y decodificar archivo CSV de forma segura
uploaded = files.upload()
csv_file = list(uploaded.keys())[0]
with open(csv_file, 'rb') as f:
    encoding_detected = chardet.detect(f.read())['encoding']

print(f"Detected encoding: {encoding_detected}")
df = pd.read_csv(csv_file, encoding=encoding_detected)

# Validación de columnas esperadas
required_columns = {"VAL", "MOT", "EXC", "Timestamp", "theta_fz", "beta_fz", "beta_pz", "beta_total", "alpha_total"}
if not required_columns.issubset(df.columns):
    raise ValueError("Missing one or more required columns required for analysis")

# Paso 3: Cálculo de resultados y umbrales
thresholds = {"VAL": 0.15, "MOT": 0.25, "EXC": 0.40}
minimum_success_rate = 0.7
results = {}

# Cálculo de los índices base
for key, threshold in thresholds.items():
    above = df[key] > threshold
    count_above = above.sum()
    percentage = count_above / len(df)
    results[key] = {
        "average": df[key].mean(),
        "above_threshold_%": percentage * 100,
        "threshold": threshold,
        "count": count_above
    }

# Cálculo de métricas adicionales
df["CE"] = df["theta_fz"] / df["beta_pz"]
df["TBR"] = df["theta_fz"] / df["beta_fz"]
df["EI"] = df["beta_total"] / (df["alpha_total"] + df["theta_total"])

for key in ["CE", "TBR", "EI"]:
    results[key] = {
        "average": df[key].mean(),
        "std_dev": df[key].std(),
        "min": df[key].min(),
        "max": df[key].max()
    }

is_viable = all(results[k]["above_threshold_%"] >= minimum_success_rate * 100 for k in thresholds)

# Paso 4: Análisis por fases de tiempo
df["block"] = pd.qcut(df.index, q=3, labels=["Start", "Middle", "End"])
block_summary = df.groupby("block")[["VAL", "MOT", "EXC"]].mean()

# Paso 5: Visualizaciones
os.makedirs("report_figures", exist_ok=True)

# Línea de tiempo
plt.figure(figsize=(10, 6))
for metric, color in zip(["VAL", "MOT", "EXC"], ["skyblue", "orange", "green"]):
    sns.lineplot(data=df, x="Timestamp", y=metric, label=metric, color=color)
plt.xticks(rotation=45)
plt.title("Affective Index Trends Over Time")
plt.xlabel("Time")
plt.ylabel("Index Value")
plt.tight_layout()
plt.savefig("report_figures/lineplot.png")
plt.close()

# Boxplot
plt.figure(figsize=(10, 5))
sns.boxplot(data=df[["VAL", "MOT", "EXC"]])
plt.title("Distribution of Affective Indices")
plt.savefig("report_figures/boxplot.png")
plt.close()

# Histogramas
plt.figure(figsize=(15, 4))
for i, key in enumerate(["VAL", "MOT", "EXC"]):
    plt.subplot(1, 3, i+1)
    sns.histplot(df[key], kde=True, bins=20)
    plt.title(f"Histogram: {key}")
plt.tight_layout()
plt.savefig("report_figures/histograms.png")
plt.close()

# Heatmap
plt.figure(figsize=(5, 4))
sns.heatmap(df[["VAL", "MOT", "EXC"]].corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Heatmap")
plt.savefig("report_figures/heatmap.png")
plt.close()

# Nuevas métricas - histogramas
plt.figure(figsize=(15, 4))
for i, key in enumerate(["CE", "TBR", "EI"]):
    plt.subplot(1, 3, i+1)
    sns.histplot(df[key], kde=True, bins=20)
    plt.title(f"{key} Histogram")
plt.tight_layout()
plt.savefig("report_figures/new_engagement_metrics.png")
plt.close()

# Tabla resumen visual
fig, ax = plt.subplots(figsize=(6, 2.5))
columns = ["Index", "Threshold", "Average", "% Time Above", "Status"]
cell_data = []
for k, v in results.items():
    if "threshold" in v:
        status = "PASS" if v["above_threshold_%"] >= minimum_success_rate * 100 else "FAIL"
        cell_data.append([k, f">{v['threshold']}", f"{v['average']:.2f}", f"{v['above_threshold_%']:.1f}%", status])
table = plt.table(cellText=cell_data, colLabels=columns, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 2.2)
plt.axis('off')
plt.title("Summary of Engagement Metrics", pad=20)
plt.savefig("report_figures/summary_table.png")
plt.close()

# Paso 6: Generar PDF
class PDFReport(FPDF):
    def header(self):
        self.set_font("Arial", 'B', 14)
        self.cell(0, 10, "VR Neuromarketing EEG Engagement Report", ln=True, align="C")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", 'I', 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def section_title(self, title):
        self.set_font("Arial", 'B', 12)
        self.cell(0, 10, title, ln=True)
        self.ln(1)

    def section_text(self, text):
        self.set_font("Arial", '', 11)
        self.multi_cell(0, 8, text)
        self.ln(2)

pdf = PDFReport()
pdf.add_page()

pdf.section_title("1. Objective & Methodology")
pdf.section_text(
    "This report evaluates the emotional and cognitive engagement of users within a VR environment. Using EEG data, "
    "three primary affective indices are calculated: Valence, Motivation, and Excitation. Additionally, cognitive metrics "
    "such as CE, TBR, and EI are computed from spectral band data and interpreted based on neuroscientific literature."
)

pdf.section_title("2. How Indices Were Calculated")
pdf.section_text(
    "- Valence (VAL): Computed as alpha asymmetry between F4 and F3 electrodes.\n"
    "- Motivation (MOT): Beta asymmetry between F4 and F3.\n"
    "- Excitation (EXC): Beta / (Alpha + Theta).\n"
    "- Cognitive Engagement (CE): Theta Fz / Beta Pz.\n"
    "- Theta/Beta Ratio (TBR): Theta Fz / Beta Fz.\n"
    "- Engagement Index (EI): Beta Total / (Alpha Total + Theta Total)."
)

pdf.section_title("3. Affective Index Results")
for key in ["VAL", "MOT", "EXC"]:
    value = results[key]
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 10, f"{key}:", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 8, f"  - Threshold: > {value['threshold']}", ln=True)
    pdf.cell(0, 8, f"  - Average: {value['average']:.4f}", ln=True)
    pdf.cell(0, 8, f"  - Time Above Threshold: {value['above_threshold_%']:.2f}%", ln=True)
    pdf.ln(2)

pdf.section_title("4. Additional Engagement Metrics")
for key in ["CE", "TBR", "EI"]:
    val = results[key]
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 10, f"{key}:", ln=True)
    pdf.set_font("Arial", '', 11)
    pdf.cell(0, 8, f"  - Average: {val['average']:.4f}", ln=True)
    pdf.cell(0, 8, f"  - Std Dev: {val['std_dev']:.4f}", ln=True)
    pdf.cell(0, 8, f"  - Min/Max: {val['min']:.4f} / {val['max']:.4f}", ln=True)
    pdf.ln(2)

pdf.section_title("5. Session Phase Analysis")
for block in block_summary.index:
    avg_vals = block_summary.loc[block].to_dict()
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 10, f"{block} Phase:", ln=True)
    pdf.set_font("Arial", '', 11)
    for key, val in avg_vals.items():
        pdf.cell(0, 8, f"  - {key}: {val:.4f}", ln=True)
    pdf.ln(1)

pdf.section_title("6. Interpretation & Final Conclusion")
conclusion_text = (
    "This extended analysis with multiple cognitive-affective metrics provides a robust view of user engagement in VR. "
    "If the core indices (VAL, MOT, EXC) pass thresholds, the experience is considered viable. If not, it's advised to optimize design."
)
pdf.section_text(conclusion_text)

# Visualizations
pdf.add_page()
pdf.section_title("7. Visualizations")
for path in [
    "report_figures/summary_table.png",
    "report_figures/lineplot.png",
    "report_figures/boxplot.png",
    "report_figures/histograms.png",
    "report_figures/heatmap.png",
    "report_figures/new_engagement_metrics.png"
]:
    if os.path.exists(path):
        pdf.image(path, w=180)
        pdf.ln(5)

# Guardar PDF
pdf_path = "VR_Engagement_Robust_Explained_Report.pdf"
pdf.output(pdf_path)
files.download(pdf_path)

from nbformat import v4
import json

notebook = v4.new_notebook(cells=[v4.new_code_cell(notebook_code)])
file_path = "/mnt/data/VR_Engagement_Expanded_Report.ipynb"
with open(file_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f)

file_path
