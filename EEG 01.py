import os
import csv
import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
from pylsl import StreamInlet, resolve_streams
import numpy as np
import threading
import time
from datetime import datetime
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Variables globales
base_path = r"C:\Users\aisbo\OneDrive\Escritorio\NEUROMKT"  # Ruta personalizada
recording = False
stop_program = False
data_accumulated = []
last_recorded_time = 0
event_counter = 1
event_log = []
saved = False
session_folder = None
fft_folder = None
raw_folder = None
raw_file_path = None
inlet_fft = None
inlet_raw = None
plot_data = {"VAL": [], "MOT": [], "EXC": [], "timestamps": []}
def check_aura_connection():
    def check():
        status_label.config(text="Buscando EEG stream...")
        start_time = time.time()
        stream = None
        while time.time() - start_time < 5:
            stream = resolve_('name', 'AURA_Power')
            if stream:
                break
        if stream:
            root.after(0, lambda: status_label.config(text="✅ AURA conectado"))
            root.after(0, lambda: messagebox.showinfo("Conexión", "AURA está conectado correctamente."))
        else:
            root.after(0, lambda: status_label.config(text="❌ AURA no encontrado"))
            root.after(0, lambda: messagebox.showerror("Error", "No se encontró conexión con AURA después de 5 segundos."))
    threading.Thread(target=check, daemon=True).start()


def toggle_recording():
    global recording, session_folder, fft_folder, raw_folder, raw_file_path
    recording = not recording
    state = "🟢 Grabando" if recording else "🔴 Detenido"
    status_label.config(text=f"Estado: {state}")
    if recording:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        session_folder = os.path.join(base_path, f"Session_{timestamp}")
        fft_folder = os.path.join(session_folder, "FFT")
        raw_folder = os.path.join(session_folder, "RAW")
        os.makedirs(fft_folder, exist_ok=True)
        os.makedirs(raw_folder, exist_ok=True)
        raw_file_path = os.path.join(raw_folder, f"raw_{timestamp}.csv")


def start_data_collection():
    global stop_program, last_recorded_time, event_counter, inlet_fft, inlet_raw
    try:
        stream_fft = [s for s in resolve_streams() if s.name() == 'AURA_Power']
        inlet_fft = StreamInlet(stream_fft[0])
        print("Conectado a AURA_Power")

        stream_raw = [s for s in resolve_streams() if s.name() == 'AURA']
        inlet_raw = StreamInlet(stream_raw[0])
        print("Conectado a AURA (RAW)")

    except Exception as e:
        messagebox.showerror("Error", f"No se pudo conectar con Aura: {e}")
        return


    threading.Thread(target=collect_raw_data, daemon=True).start()

    while not stop_program:
        if recording:
            try:
                current_time = time.time()
                if current_time - last_recorded_time < 1:
                    continue
                last_recorded_time = current_time

                sample, timestamp = inlet_fft.pull_sample()
                if not sample or len(sample) < 33:
                    continue

                theta = sample[9:17]
                alpha = sample[17:25]
                beta = sample[25:33]

                ce_values = [b / (t + a) if (t + a) != 0 else 0 for b, t, a in zip(beta, theta, alpha)]
                tbr_values = [t / b if b != 0 else 0 for t, b in zip(theta, beta)]

                alpha_f3 = alpha[0]
                alpha_f4 = alpha[2]
                beta_f3 = beta[0]
                beta_f4 = beta[2]
                alpha_total = sum(alpha)
                beta_total = sum(beta)
                theta_total = sum(theta)

                valencia = (alpha_f4 - alpha_f3) / (alpha_f4 + alpha_f3) if (alpha_f3 + alpha_f4) != 0 else 0
                motivacion = (beta_f4 - beta_f3) / (beta_f4 + beta_f3) if (beta_f3 + beta_f4) != 0 else 0
                excitacion = beta_total / (alpha_total + theta_total) if (alpha_total + theta_total) != 0 else 0

                ce_avg = np.mean(ce_values)
                tbr_avg = np.mean(tbr_values)
                formatted_time = datetime.now().strftime('%H:%M:%S')

                data_accumulated.append((formatted_time, event_counter, ce_values, tbr_values, valencia, motivacion, excitacion))

                root.after(0, lambda: ce_label.config(text=f"🧩 CE: {ce_avg:.4f}"))
                root.after(0, lambda: tbr_label.config(text=f"📊 TBR: {tbr_avg:.4f}"))
                root.after(0, lambda: val_label.config(text=f"💙 VAL: {valencia:.4f}"))
                root.after(0, lambda: mot_label.config(text=f"🧠 MOT: {motivacion:.4f}"))
                root.after(0, lambda: exc_label.config(text=f"⚡ EXC: {excitacion:.4f}"))

            except Exception as e:
                print(f"Error durante la adquisición de datos: {e}")
def collect_raw_data():
    global raw_file_path
    while not recording or not raw_file_path:
        time.sleep(0.1)
    with open(raw_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'F3', 'Fz', 'F4', 'C3', 'C4', 'P3', 'Pz', 'P4'])
        while not stop_program:
            if recording:
                try:
                    sample, timestamp = inlet_raw.pull_sample(timeout=0.1)
                    if sample:
                        formatted_time = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                        writer.writerow([formatted_time] + sample[:9])
                except Exception as e:
                    print(f"Error leyendo señal RAW: {e}")
            time.sleep(0.01)


def save_data():
    global saved
    if not data_accumulated or saved or not fft_folder:
        return
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_path = os.path.join(fft_folder, f"indices_ce_tbr_valmotexc_{timestamp}.csv")

    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        headers = ['Timestamp', 'Evento'] + \
                  [f'CE_{i+1}' for i in range(8)] + \
                  [f'TBR_{i+1}' for i in range(8)] + \
                  ['VAL', 'MOT', 'EXC']
        writer.writerow(headers)

        for row in data_accumulated:
            timestamp_, evento, ce_vals, tbr_vals, val, mot, exc = row
            writer.writerow([timestamp_, evento] + ce_vals + tbr_vals + [val, mot, exc])

        writer.writerow([])
        writer.writerow(["Promedios"])
        ce_cols = list(zip(*[ce for _, _, ce, _, _, _, _ in data_accumulated]))
        tbr_cols = list(zip(*[tbr for _, _, _, tbr, _, _, _ in data_accumulated]))
        ce_avgs = [np.mean(col) if col else 0 for col in ce_cols]
        tbr_avgs = [np.mean(col) if col else 0 for col in tbr_cols]
        writer.writerow([""] + [""] + ce_avgs + tbr_avgs + ["", "", ""])

        writer.writerow([])
        writer.writerow(["Eventos"])
        writer.writerow(["Número de Evento", "Hora"])
        for evento, hora in event_log:
            writer.writerow([evento, hora])

    messagebox.showinfo("Guardado", "Datos guardados con éxito.")
    saved = True


def change_folder():
    global base_path
    folder = filedialog.askdirectory(initialdir=base_path)
    if folder:
        base_path = folder
        folder_label.config(text=f"Carpeta base: {base_path}")


def open_folder():
    if session_folder:
        os.startfile(session_folder) if os.name == 'nt' else subprocess.run(["xdg-open", session_folder])


def on_space_press(event):
    global event_counter, stop_program
    if event_counter < 12:
        event_counter += 1
        event_time = datetime.now().strftime('%H:%M:%S')
        event_log.append((event_counter, event_time))
        event_label.config(text=f"Eventos: {event_counter}")
        if event_counter == 12:
            stop_program = True
            status_label.config(text="🛑 Recolección detenida: 12 eventos alcanzados")
            if not saved and messagebox.askyesno("Guardar datos", "¿Deseas guardar los datos ahora?"):
                save_data()


def on_closing():
    if not saved and data_accumulated:
        if messagebox.askyesno("¿Salir sin guardar?", "¿Deseas guardar los datos antes de cerrar?"):
            save_data()
    root.destroy()


# ==== INTERFAZ ====
root = tk.Tk()
root.title("EEG Índices Cognitivos y Afectivos")
root.geometry("700x600")
root.configure(bg="#f0f4f8")

frame = tk.Frame(root, bg="#ffffff", bd=2, relief="groove")
frame.pack(padx=10, pady=10, fill="both")

status_label = tk.Label(frame, text="Estado: 🔴 Detenido", font=("Arial", 14, "bold"), fg="#333", bg="#ffffff")
status_label.pack(pady=10)

ce_label = tk.Label(frame, text="🧩 CE: --", font=("Arial", 13), bg="#ffffff")
ce_label.pack()
tbr_label = tk.Label(frame, text="📊 TBR: --", font=("Arial", 13), bg="#ffffff")
tbr_label.pack()

val_label = tk.Label(frame, text="💙 VAL: --", font=("Arial", 13), bg="#ffffff")
val_label.pack()
mot_label = tk.Label(frame, text="🧠 MOT: --", font=("Arial", 13), bg="#ffffff")
mot_label.pack()
exc_label = tk.Label(frame, text="⚡ EXC: --", font=("Arial", 13), bg="#ffffff")
exc_label.pack()

btn_toggle = tk.Button(frame, text="▶ Iniciar / Detener Grabación", command=toggle_recording,
                       bg="#4caf50", fg="white", font=("Arial", 12, "bold"))
btn_toggle.pack(pady=10)

btn_save = tk.Button(frame, text="💾 Guardar Datos", command=save_data,
                     bg="#2196f3", fg="white", font=("Arial", 12))
btn_save.pack(pady=5)

btn_check_connection = tk.Button(frame, text="🔌 Revisar conexión AURA", command=check_aura_connection,
                                 bg="#ff9800", fg="white", font=("Arial", 12))
btn_check_connection.pack(pady=5)

folder_label = tk.Label(frame, text=f"Carpeta base: {base_path}", bg="#ffffff", font=("Arial", 9))
folder_label.pack(pady=5)

btn_folder = tk.Button(frame, text="📂 Cambiar Carpeta", command=change_folder)
btn_folder.pack(pady=3)

btn_open_folder = tk.Button(frame, text="📁 Abrir Carpeta de Sesión", command=open_folder)
btn_open_folder.pack(pady=3)

event_label = tk.Label(frame, text="Eventos: 1", font=("Arial", 12), bg="#ffffff")
event_label.pack(pady=8)

root.bind("<space>", on_space_press)
root.protocol("WM_DELETE_WINDOW", on_closing)
data_thread = threading.Thread(target=start_data_collection, daemon=True)
data_thread.start()
root.mainloop()
