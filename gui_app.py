import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import webbrowser
import tempfile
import os
from sensor_loader import SensorDataLoader
from step_analyzer import FootSensorAnalyzer
from step_plotter import StepPlotter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo

class FootPressureGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Analiza Nacisku Stopy")
        self.root.geometry("1600x1000")
        self.root.configure(bg='#f0f0f0')
        
        # Zmienne do przechowywania danych
        self.loader = None
        self.analyzer = None
        self.plotter = None
        self.current_sheet = 0
        self.sheets_data = {}
        
        # Canvas i figure dla matplotlib
        self.fig = None
        self.canvas = None
        self.toolbar = None
        
        self.create_widgets()
        
    def create_widgets(self):
        # Główny frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Konfiguracja grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=3)  # Większy udział dla wykresów
        main_frame.rowconfigure(0, weight=1)
        
        # Panel kontrolny (lewy)
        control_frame = ttk.LabelFrame(main_frame, text="Panel Kontrolny", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        control_frame.configure(width=350)
        
        # Przycisk wczytywania pliku
        ttk.Button(control_frame, text="Wczytaj plik Excel", 
                  command=self.load_file).grid(row=0, column=0, pady=5, sticky=(tk.W, tk.E))
        
        # Wybór arkusza
        ttk.Label(control_frame, text="Arkusz:").grid(row=1, column=0, pady=5, sticky=tk.W)
        self.sheet_var = tk.StringVar()
        self.sheet_combo = ttk.Combobox(control_frame, textvariable=self.sheet_var, 
                                       state="readonly", width=20)
        self.sheet_combo.grid(row=2, column=0, pady=5, sticky=(tk.W, tk.E))
        self.sheet_combo.bind('<<ComboboxSelected>>', self.on_sheet_changed)
        
        # Separator
        ttk.Separator(control_frame, orient='horizontal').grid(row=3, column=0, 
                                                               sticky=(tk.W, tk.E), pady=10)
        
        # Informacje o danych
        self.info_text = tk.Text(control_frame, height=12, width=35, wrap=tk.WORD,
                                font=('Consolas', 9))
        self.info_text.grid(row=4, column=0, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbar dla info_text
        info_scroll = ttk.Scrollbar(control_frame, orient="vertical", command=self.info_text.yview)
        info_scroll.grid(row=4, column=1, sticky=(tk.N, tk.S))
        self.info_text.configure(yscrollcommand=info_scroll.set)
        
        # Separator
        ttk.Separator(control_frame, orient='horizontal').grid(row=5, column=0, 
                                                               sticky=(tk.W, tk.E), pady=10)
        
        # Wybór typu wykresu
        ttk.Label(control_frame, text="Typ wykresu:", 
                 font=('Arial', 10, 'bold')).grid(row=6, column=0, pady=5, sticky=tk.W)
        
        self.plot_type_var = tk.StringVar(value="signals")
        
        # Radio buttons dla typów wykresów
        plot_options = [
            ("📊 Sygnały czujników", "signals"),
            ("👣 Wykrywanie kroków", "step_detection"),
            ("⏱️ Rytm kroków", "step_rhythm"),
            ("🔥 Mapa ciepła", "heatmap"),
            ("⚖️ Analiza symetrii", "symmetry"),
            ("📈 Korelacja kroków", "correlation")
        ]
        
        for i, (text, value) in enumerate(plot_options):
            ttk.Radiobutton(control_frame, text=text, variable=self.plot_type_var,
                           value=value, command=self.update_plot).grid(
                               row=7+i, column=0, pady=2, sticky=tk.W)
        
        # Separator
        ttk.Separator(control_frame, orient='horizontal').grid(row=13, column=0, 
                                                               sticky=(tk.W, tk.E), pady=10)
        
        # Przycisk eksportu do przeglądarki
        ttk.Button(control_frame, text="🌐 Otwórz w przeglądarce", 
                  command=self.export_to_browser).grid(row=14, column=0, pady=5, sticky=(tk.W, tk.E))
        
        # Panel wykresów (prawy)
        self.plot_frame = ttk.LabelFrame(main_frame, text="Wykresy", padding="10")
        self.plot_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Konfiguracja grid dla control_frame
        control_frame.columnconfigure(0, weight=1)
        control_frame.rowconfigure(4, weight=1)
        
        # Konfiguracja grid dla plot_frame
        self.plot_frame.columnconfigure(0, weight=1)
        self.plot_frame.rowconfigure(0, weight=1)
        
        # Inicjalizuj matplotlib figure
        self.setup_matplotlib()
        
        # Label powitalny
        self.welcome_label = ttk.Label(self.plot_frame, 
                                      text="Witaj w aplikacji analizy nacisku stopy!\n\n"
                                           "1. Wczytaj plik Excel z danymi\n"
                                           "2. Wybierz arkusz do analizy\n"
                                           "3. Wykresy pojawią się automatycznie\n"
                                           "4. Użyj opcji po lewej do zmiany typu wykresu",
                                      font=('Arial', 12),
                                      justify=tk.CENTER)
        self.welcome_label.grid(row=0, column=0, pady=50)
        
    def setup_matplotlib(self):
        """Inicjalizuje matplotlib w tkinter"""
        self.fig = Figure(figsize=(12, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Toolbar nawigacyjny – ręczne użycie .grid()
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        self.toolbar.update()
        self.toolbar.grid(row=1, column=0, sticky=(tk.W, tk.E))

        # Ukryj początkowo
        self.canvas.get_tk_widget().grid_remove()
        self.toolbar.grid_remove()

        
    def show_plot_area(self):
        """Pokazuje obszar wykresów"""
        self.welcome_label.grid_remove()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.toolbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
    def hide_plot_area(self):
        """Ukrywa obszar wykresów"""
        self.canvas.get_tk_widget().grid_remove()
        self.toolbar.grid_remove()
        self.welcome_label.grid(row=0, column=0, pady=50)
        
    def load_file(self):
        """Wczytuje plik Excel"""
        file_path = filedialog.askopenfilename(
            title="Wybierz plik Excel",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.loader = SensorDataLoader(file_path)
                # Wczytaj pierwszy arkusz żeby sprawdzić strukturę
                time, left, right, sheet_name = self.loader.load_sheet(0)
                
                # Populate combo box with sheet names
                self.sheet_combo['values'] = self.loader.sheet_names
                self.sheet_combo.set(self.loader.sheet_names[0])
                
                # Analyze first sheet
                self.analyze_current_sheet()
                
                # Pokaż wykresy
                self.show_plot_area()
                self.update_plot()
                
                messagebox.showinfo("Sukces", f"Wczytano plik: {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("Błąd", f"Nie można wczytać pliku:\n{str(e)}")
    
    def on_sheet_changed(self, event=None):
        """Obsługuje zmianę arkusza"""
        if self.loader and self.sheet_var.get():
            sheet_name = self.sheet_var.get()
            self.current_sheet = self.loader.sheet_names.index(sheet_name)
            self.analyze_current_sheet()
            self.update_plot()
    
    def analyze_current_sheet(self):
        """Analizuje aktualnie wybrany arkusz"""
        if not self.loader:
            return
            
        try:
            time, left, right, sheet_name = self.loader.load_sheet(self.current_sheet)
            self.analyzer = FootSensorAnalyzer(time, left, right, sheet_name)
            self.plotter = StepPlotter(time, self.analyzer)
            
            # Aktualizuj informacje
            self.update_info_display()
            
        except Exception as e:
            messagebox.showerror("Błąd", f"Nie można przeanalizować arkusza:\n{str(e)}")
    
    def update_info_display(self):
        """Aktualizuje wyświetlane informacje o danych"""
        if not self.analyzer:
            return
            
        self.info_text.delete(1.0, tk.END)
        
        # Podstawowe informacje
        info = f"=== {self.analyzer.sheet_name} ===\n"
        info += f"Czas pomiaru: {self.analyzer.time[-1] - self.analyzer.time[0]:.2f} s\n"
        info += f"Częstotliwość: {self.analyzer.sampling_rate:.1f} Hz\n"
        info += f"Próbek: {len(self.analyzer.time)}\n\n"
        
        # Analiza kroków
        left_steps, _ = self.analyzer.detect_steps('left')
        right_steps, _ = self.analyzer.detect_steps('right')
        
        info += "=== KROKI ===\n"
        info += f"Lewa stopa: {len(left_steps)}\n"
        info += f"Prawa stopa: {len(right_steps)}\n"
        
        if len(left_steps) > 0:
            left_phases = self.analyzer.analyze_step_phases('left', left_steps)
            avg_step_duration = np.mean([phase['step_duration'] for phase in left_phases])
            info += f"Śr. czas kroku (L): {avg_step_duration:.3f} s\n"
        
        if len(right_steps) > 0:
            right_phases = self.analyzer.analyze_step_phases('right', right_steps)
            avg_step_duration = np.mean([phase['step_duration'] for phase in right_phases])
            info += f"Śr. czas kroku (P): {avg_step_duration:.3f} s\n"
        
        # Rozkład nacisku
        info += "\n=== NACISK ===\n"
        left_dist = self.analyzer.calculate_pressure_distribution('left')
        right_dist = self.analyzer.calculate_pressure_distribution('right')
        
        info += "TOP 3 czujników (L):\n"
        sorted_left = sorted(left_dist.items(), key=lambda x: x[1]['mean'], reverse=True)
        for i, (sensor, stats) in enumerate(sorted_left[:3]):
            info += f"  {i+1}. {sensor}: {stats['mean']:.1f}\n"
            
        info += "TOP 3 czujników (P):\n"
        sorted_right = sorted(right_dist.items(), key=lambda x: x[1]['mean'], reverse=True)
        for i, (sensor, stats) in enumerate(sorted_right[:3]):
            info += f"  {i+1}. {sensor}: {stats['mean']:.1f}\n"
        
        self.info_text.insert(1.0, info)
    
    def update_plot(self):
        """Aktualizuje wykres na podstawie wybranego typu"""
        if not self.analyzer:
            return
            
        plot_type = self.plot_type_var.get()
        
        try:
            self.fig.clear()
            
            if plot_type == "signals":
                self.plot_signals_matplotlib()
            elif plot_type == "step_detection":
                self.plot_step_detection_matplotlib()
            elif plot_type == "step_rhythm":
                self.plot_step_rhythm_matplotlib()
            elif plot_type == "heatmap":
                self.plot_heatmap_matplotlib()
            elif plot_type == "symmetry":
                self.plot_symmetry_matplotlib()
            elif plot_type == "correlation":
                self.plot_correlation_matplotlib()
                
            self.fig.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Błąd", f"Nie można wygenerować wykresu:\n{str(e)}")
    
    def plot_signals_matplotlib(self):
        """Rysuje sygnały z czujników w matplotlib"""
        # Subplot dla lewej stopy
        ax1 = self.fig.add_subplot(2, 1, 1)
        for i in range(8):
            ax1.plot(self.analyzer.time, self.analyzer.left[:, i], 
                    label=f'L{i+1}', alpha=0.7)
        ax1.set_title('Lewa stopa - Sygnały z czujników')
        ax1.set_ylabel('Siła [N]')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Subplot dla prawej stopy
        ax2 = self.fig.add_subplot(2, 1, 2)
        for i in range(8):
            ax2.plot(self.analyzer.time, self.analyzer.right[:, i], 
                    label=f'P{i+1}', alpha=0.7)
        ax2.set_title('Prawa stopa - Sygnały z czujników')
        ax2.set_xlabel('Czas [s]')
        ax2.set_ylabel('Siła [N]')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
    def plot_step_detection_matplotlib(self):
        """Rysuje wykrywanie kroków w matplotlib"""
        ax1 = self.fig.add_subplot(2, 1, 1)
        
        # Lewa stopa - suma sił
        left_total = np.sum(self.analyzer.left, axis=1)
        ax1.plot(self.analyzer.time, left_total, 'b-', label='Suma sił', linewidth=2)
        
        # Wykryte kroki
        left_steps, _ = self.analyzer.detect_steps('left')
        if len(left_steps) > 0:
            ax1.scatter(self.analyzer.time[left_steps], left_total[left_steps], 
                       color='red', s=50, zorder=5, label='Wykryte kroki')
        
        ax1.set_title('Lewa stopa - Wykrywanie kroków')
        ax1.set_ylabel('Siła [N]')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Prawa stopa
        ax2 = self.fig.add_subplot(2, 1, 2)
        right_total = np.sum(self.analyzer.right, axis=1)
        ax2.plot(self.analyzer.time, right_total, 'r-', label='Suma sił', linewidth=2)
        
        right_steps, _ = self.analyzer.detect_steps('right')
        if len(right_steps) > 0:
            ax2.scatter(self.analyzer.time[right_steps], right_total[right_steps], 
                       color='red', s=50, zorder=5, label='Wykryte kroki')
        
        ax2.set_title('Prawa stopa - Wykrywanie kroków')
        ax2.set_xlabel('Czas [s]')
        ax2.set_ylabel('Siła [N]')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
    def plot_step_rhythm_matplotlib(self):
        """Rysuje rytm kroków w matplotlib"""
        times_left, periods_left = self.analyzer.get_step_periods("left")
        times_right, periods_right = self.analyzer.get_step_periods("right")
        
        ax1 = self.fig.add_subplot(2, 2, 1)
        if len(times_left) > 0:
            ax1.plot(times_left, periods_left, 'bo-', label='Lewa stopa')
        if len(times_right) > 0:
            ax1.plot(times_right, periods_right, 'ro-', label='Prawa stopa')
        ax1.set_title('Okresy kroków w czasie')
        ax1.set_xlabel('Czas [s]')
        ax1.set_ylabel('Okres [s]')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Histogram okresów
        ax2 = self.fig.add_subplot(2, 2, 2)
        if len(periods_left) > 0:
            ax2.hist(periods_left, bins=10, alpha=0.5, label='Lewa stopa', color='blue')
        if len(periods_right) > 0:
            ax2.hist(periods_right, bins=10, alpha=0.5, label='Prawa stopa', color='red')
        ax2.set_title('Rozkład okresów kroków')
        ax2.set_xlabel('Okres [s]')
        ax2.set_ylabel('Liczba kroków')
        ax2.legend()
        
        # Statystyki
        ax3 = self.fig.add_subplot(2, 2, 3)
        stats_data = []
        labels = []
        
        if len(periods_left) > 0:
            stats_data.extend([np.mean(periods_left), np.std(periods_left)])
            labels.extend(['Średnia L', 'Odch. std L'])
        if len(periods_right) > 0:
            stats_data.extend([np.mean(periods_right), np.std(periods_right)])
            labels.extend(['Średnia P', 'Odch. std P'])
            
        if stats_data:
            bars = ax3.bar(labels, stats_data, color=['blue', 'lightblue', 'red', 'lightcoral'])
            ax3.set_title('Statystyki okresów kroków')
            ax3.set_ylabel('Czas [s]')
            
            # Dodaj wartości na słupkach
            for bar, value in zip(bars, stats_data):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                        f'{value:.3f}', ha='center', va='bottom')
        
        # Wykres różnic
        ax4 = self.fig.add_subplot(2, 2, 4)
        if len(periods_left) > 1:
            diff_left = np.diff(periods_left)
            ax4.plot(times_left[1:], diff_left, 'b-', label='Różnice L', alpha=0.7)
        if len(periods_right) > 1:
            diff_right = np.diff(periods_right)
            ax4.plot(times_right[1:], diff_right, 'r-', label='Różnice P', alpha=0.7)
        ax4.set_title('Zmienność rytmu kroków')
        ax4.set_xlabel('Czas [s]')
        ax4.set_ylabel('Różnica okresów [s]')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
    def plot_heatmap_matplotlib(self):
        """Rysuje mapy ciepła w matplotlib"""
        # Oblicz średnie naciski dla każdego czujnika
        left_means = [np.mean(self.analyzer.left[:, i]) for i in range(8)]
        right_means = [np.mean(self.analyzer.right[:, i]) for i in range(8)]
        
        # Symulacja układu czujników na stopie (2x4)
        left_heatmap = np.array(left_means[:4]).reshape(2, 2)
        left_heatmap = np.vstack([left_heatmap, np.array(left_means[4:]).reshape(2, 2)])
        
        right_heatmap = np.array(right_means[:4]).reshape(2, 2)
        right_heatmap = np.vstack([right_heatmap, np.array(right_means[4:]).reshape(2, 2)])
        
        # Lewa stopa
        ax1 = self.fig.add_subplot(1, 2, 1)
        im1 = ax1.imshow(left_heatmap, cmap='YlOrRd', aspect='auto')
        ax1.set_title('Lewa stopa - Rozkład nacisku')
        ax1.set_xlabel('Pozycja X')
        ax1.set_ylabel('Pozycja Y')
        
        # Dodaj wartości na mapie
        for i in range(left_heatmap.shape[0]):
            for j in range(left_heatmap.shape[1]):
                ax1.text(j, i, f'{left_heatmap[i, j]:.1f}', 
                        ha='center', va='center', color='black', fontweight='bold')
        
        # Prawa stopa
        ax2 = self.fig.add_subplot(1, 2, 2)
        im2 = ax2.imshow(right_heatmap, cmap='YlOrRd', aspect='auto')
        ax2.set_title('Prawa stopa - Rozkład nacisku')
        ax2.set_xlabel('Pozycja X')
        ax2.set_ylabel('Pozycja Y')
        
        # Dodaj wartości na mapie
        for i in range(right_heatmap.shape[0]):
            for j in range(right_heatmap.shape[1]):
                ax2.text(j, i, f'{right_heatmap[i, j]:.1f}', 
                        ha='center', va='center', color='black', fontweight='bold')
        
        # Colorbar
        self.fig.colorbar(im2, ax=[ax1, ax2], label='Siła [N]')
        
    def plot_symmetry_matplotlib(self):
        """Rysuje analizę symetrii w matplotlib"""
        symmetry = self.analyzer.analyze_multi_sensor_symmetry()
        
        # Wskaźniki symetrii
        ax1 = self.fig.add_subplot(2, 2, 1)
        sensors = list(symmetry['sensor_symmetry_indices'].keys())
        sym_values = list(symmetry['sensor_symmetry_indices'].values())
        
        bars1 = ax1.bar(sensors, sym_values, color='lightblue')
        ax1.set_title('Wskaźniki symetrii czujników')
        ax1.set_ylabel('Wskaźnik symetrii [%]')
        ax1.tick_params(axis='x', rotation=45)
        
        # Dodaj wartości na słupkach
        for bar, value in zip(bars1, sym_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        # Korelacje
        ax2 = self.fig.add_subplot(2, 2, 2)
        corr_values = list(symmetry['sensor_correlations'].values())
        bars2 = ax2.bar(sensors, corr_values, color='lightgreen')
        ax2.set_title('Korelacje L-P')
        ax2.set_ylabel('Korelacja')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars2, corr_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.2f}', ha='center', va='bottom')
        
        # PCA
        ax3 = self.fig.add_subplot(2, 2, 3)
        pca_labels = [f'PC{i+1}' for i in range(len(symmetry['pca_explained_variance']))]
        bars3 = ax3.bar(pca_labels, symmetry['pca_explained_variance'], color='lightcoral')
        ax3.set_title('PCA - Wyjaśniona wariancja')
        ax3.set_ylabel('Wariancja')
        
        for bar, value in zip(bars3, symmetry['pca_explained_variance']):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.2f}', ha='center', va='bottom')
        
        # Porównanie średnich sił
        ax4 = self.fig.add_subplot(2, 2, 4)
        left_means = [np.mean(self.analyzer.left[:, i]) for i in range(8)]
        right_means = [np.mean(self.analyzer.right[:, i]) for i in range(8)]
        
        x = np.arange(len(sensors))
        width = 0.35
        
        ax4.bar(x - width/2, left_means, width, label='Lewa stopa', color='blue', alpha=0.7)
        ax4.bar(x + width/2, right_means, width, label='Prawa stopa', color='red', alpha=0.7)
        
        ax4.set_title('Porównanie średnich sił')
        ax4.set_xlabel('Czujnik')
        ax4.set_ylabel('Średnia siła [N]')
        ax4.set_xticks(x)
        ax4.set_xticklabels(sensors)
        ax4.legend()
        
    def plot_correlation_matplotlib(self):
        """Rysuje korelację kroków w matplotlib"""
        times_left, periods_left = self.analyzer.get_step_periods("left")
        times_right, periods_right = self.analyzer.get_step_periods("right")
        
        # Okresy w czasie
        ax1 = self.fig.add_subplot(2, 2, 1)
        if len(times_left) > 0:
            ax1.plot(times_left, periods_left, 'bo-', label='Lewa stopa', alpha=0.7)
        if len(times_right) > 0:
            ax1.plot(times_right, periods_right, 'ro-', label='Prawa stopa', alpha=0.7)
        ax1.set_title('Okresy kroków w czasie')
        ax1.set_xlabel('Czas [s]')
        ax1.set_ylabel('Okres [s]')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Histogram okresów
        ax2 = self.fig.add_subplot(2, 2, 2)
        if len(periods_left) > 0:
            ax2.hist(periods_left, bins=10, alpha=0.6, label='Lewa stopa', color='blue')
        if len(periods_right) > 0:
            ax2.hist(periods_right, bins=10, alpha=0.6, label='Prawa stopa', color='red')
        ax2.set_title('Rozkład okresów kroków')
        ax2.set_xlabel('Okres [s]')
        ax2.set_ylabel('Częstość')
        ax2.legend()
        
        # Cross-correlation
                # Cross-correlation
        ax3 = self.fig.add_subplot(2, 2, 3)
        if len(periods_left) > 2 and len(periods_right) > 2:
            # Upewnij się, że są tego samego rozmiaru
            min_len = min(len(periods_left), len(periods_right))
            left_trimmed = periods_left[:min_len]
            right_trimmed = periods_right[:min_len]

            correlation = np.correlate(left_trimmed - np.mean(left_trimmed), 
                                       right_trimmed - np.mean(right_trimmed), 
                                       mode='full')
            lags = np.arange(-min_len + 1, min_len)
            ax3.plot(lags, correlation, 'k-')
            ax3.set_title('Korelacja między okresami kroków')
            ax3.set_xlabel('Przesunięcie')
            ax3.set_ylabel('Korelacja')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Zbyt mało danych\ndo obliczenia korelacji',
                     ha='center', va='center', transform=ax3.transAxes)

        # Scatterplot left vs right periods
        ax4 = self.fig.add_subplot(2, 2, 4)
        if len(periods_left) > 2 and len(periods_right) > 2:
            min_len = min(len(periods_left), len(periods_right))
            ax4.scatter(periods_left[:min_len], periods_right[:min_len], alpha=0.7)
            ax4.set_title('Okresy kroków: Lewa vs Prawa')
            ax4.set_xlabel('Lewa stopa [s]')
            ax4.set_ylabel('Prawa stopa [s]')
            ax4.grid(True, alpha=0.3)

            # Współczynnik korelacji
            corr_coef = np.corrcoef(periods_left[:min_len], periods_right[:min_len])[0, 1]
            ax4.text(0.05, 0.95, f'Korelacja: {corr_coef:.2f}', 
                     transform=ax4.transAxes, verticalalignment='top',
                     bbox=dict(boxstyle="round", facecolor='white', alpha=0.7))
        else:
            ax4.text(0.5, 0.5, 'Brak wystarczających danych',
                     ha='center', va='center', transform=ax4.transAxes)
 
    def export_to_browser(self):
        """Eksportuje wykres do HTML i otwiera w przeglądarce"""
        if not self.analyzer:
            messagebox.showwarning("Brak danych", "Najpierw wczytaj dane.")
            return

        plot_type = self.plot_type_var.get()

        fig = None

        try:
            if plot_type == "signals":
                fig = self.plotter.plot_signals_plotly()
            elif plot_type == "step_detection":
                fig = self.plotter.plot_step_detection_plotly()
            elif plot_type == "step_rhythm":
                fig = self.plotter.plot_step_rhythm_plotly()
            elif plot_type == "heatmap":
                fig = self.plotter.plot_heatmap_plotly()
            elif plot_type == "symmetry":
                fig = self.plotter.plot_symmetry_plotly()
            elif plot_type == "correlation":
                fig = self.plotter.plot_correlation_plotly()
            else:
                messagebox.showwarning("Nieznany typ", "Nieznany typ wykresu.")
                return

            # Zapisz do tymczasowego pliku HTML
            with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmpfile:
                pyo.plot(fig, filename=tmpfile.name, auto_open=False)
                webbrowser.open('file://' + tmpfile.name)

        except Exception as e:
            messagebox.showerror("Błąd eksportu", f"Nie udało się wyeksportować wykresu:\n{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = FootPressureGUI(root)
    root.mainloop()