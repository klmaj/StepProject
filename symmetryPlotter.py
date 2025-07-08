import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

class SymmetryVisualizationTools:
    """
    Klasa do wizualizacji wzorców symetrii z SymmetryPatternExtractor
    """
    
    def __init__(self, extractor):
        """
        Inicjalizacja narzędzi wizualizacji
        
        Args:
            extractor: obiekt SymmetryPatternExtractor
        """
        self.extractor = extractor
        self.patterns = None
        self.fingerprint = None
        
        # Konfiguracja stylów
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Kolory dla wizualizacji
        self.colors = {
            'left': '#2E86AB',
            'right': '#A23B72',
            'symmetry': '#F18F01',
            'asymmetry': '#C73E1D',
            'neutral': '#7D8491'
        }
    
    def _ensure_patterns(self):
        """Zapewnia, że wzorce są załadowane"""
        if self.patterns is None:
            self.patterns = self.extractor.extract_core_symmetry_patterns()
            self.fingerprint = self.extractor.create_symmetry_fingerprint()
    
    def plot_basic_pressure_overview(self, figsize=(15, 10)):
        """
        Podstawowy przegląd danych nacisku
        """
        fig, axes = plt.subplots(3, 3, figsize=figsize)
        fig.suptitle('Przegląd Danych Nacisku Stóp', fontsize=16, fontweight='bold')
        
        # 1. Całkowity nacisk w czasie
        total_left = self.extractor.left_foot.sum(axis=1)
        total_right = self.extractor.right_foot.sum(axis=1)
        
        axes[0, 0].plot(self.extractor.time, total_left, 
                       label='Lewa stopa', color=self.colors['left'], linewidth=2)
        axes[0, 0].plot(self.extractor.time, total_right, 
                       label='Prawa stopa', color=self.colors['right'], linewidth=2)
        axes[0, 0].set_title('Całkowity Nacisk w Czasie')
        axes[0, 0].set_xlabel('Czas')
        axes[0, 0].set_ylabel('Nacisk')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Stosunek symetrii
        symmetry_ratio = np.where(total_right != 0, total_left / total_right, 1.0)
        axes[0, 1].plot(self.extractor.time, symmetry_ratio, 
                       color=self.colors['symmetry'], linewidth=2)
        axes[0, 1].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Idealna symetria')
        axes[0, 1].set_title('Stosunek Symetrii (Lewa/Prawa)')
        axes[0, 1].set_xlabel('Czas')
        axes[0, 1].set_ylabel('Stosunek')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Rozkład nacisku średniego
        left_mean = self.extractor.left_foot.mean()
        right_mean = self.extractor.right_foot.mean()
        
        x = np.arange(8)
        width = 0.35
        
        axes[0, 2].bar(x - width/2, left_mean, width, 
                      label='Lewa stopa', color=self.colors['left'], alpha=0.7)
        axes[0, 2].bar(x + width/2, right_mean, width, 
                      label='Prawa stopa', color=self.colors['right'], alpha=0.7)
        axes[0, 2].set_title('Średni Nacisk na Czujnik')
        axes[0, 2].set_xlabel('Czujnik')
        axes[0, 2].set_ylabel('Średni Nacisk')
        axes[0, 2].set_xticks(x)
        axes[0, 2].set_xticklabels([f'S{i+1}' for i in range(8)])
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Heatmapa korelacji między czujnikami
        all_sensors = pd.concat([self.extractor.left_foot, self.extractor.right_foot], axis=1)
        correlation_matrix = all_sensors.corr()
        
        im = axes[1, 0].imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        axes[1, 0].set_title('Korelacja Między Czujnikami')
        axes[1, 0].set_xticks(range(16))
        axes[1, 0].set_yticks(range(16))
        axes[1, 0].set_xticklabels(all_sensors.columns, rotation=45)
        axes[1, 0].set_yticklabels(all_sensors.columns)
        plt.colorbar(im, ax=axes[1, 0])
        
        # 5. Zmienność czujników
        left_std = self.extractor.left_foot.std()
        right_std = self.extractor.right_foot.std()
        
        axes[1, 1].bar(x - width/2, left_std, width, 
                      label='Lewa stopa', color=self.colors['left'], alpha=0.7)
        axes[1, 1].bar(x + width/2, right_std, width, 
                      label='Prawa stopa', color=self.colors['right'], alpha=0.7)
        axes[1, 1].set_title('Zmienność Czujników')
        axes[1, 1].set_xlabel('Czujnik')
        axes[1, 1].set_ylabel('Odchylenie Standardowe')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels([f'S{i+1}' for i in range(8)])
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Histogram stosunku symetrii
        axes[1, 2].hist(symmetry_ratio, bins=30, color=self.colors['symmetry'], 
                       alpha=0.7, edgecolor='black')
        axes[1, 2].axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Idealna symetria')
        axes[1, 2].axvline(x=np.mean(symmetry_ratio), color='orange', linestyle='-', 
                          linewidth=2, label=f'Średnia: {np.mean(symmetry_ratio):.3f}')
        axes[1, 2].set_title('Rozkład Stosunku Symetrii')
        axes[1, 2].set_xlabel('Stosunek Lewa/Prawa')
        axes[1, 2].set_ylabel('Częstość')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        # 7. Trajektoria centrum nacisku
        self._plot_cop_trajectory(axes[2, 0])
        
        # 8. Analiza czasowa dla wybranych czujników
        axes[2, 1].plot(self.extractor.time, self.extractor.left_foot.iloc[:, 0], 
                       label='L1', color=self.colors['left'], linewidth=2)
        axes[2, 1].plot(self.extractor.time, self.extractor.right_foot.iloc[:, 0], 
                       label='R1', color=self.colors['right'], linewidth=2)
        axes[2, 1].plot(self.extractor.time, self.extractor.left_foot.iloc[:, 4], 
                       label='L5', color=self.colors['left'], linewidth=2, linestyle='--')
        axes[2, 1].plot(self.extractor.time, self.extractor.right_foot.iloc[:, 4], 
                       label='R5', color=self.colors['right'], linewidth=2, linestyle='--')
        axes[2, 1].set_title('Przykładowe Czujniki w Czasie')
        axes[2, 1].set_xlabel('Czas')
        axes[2, 1].set_ylabel('Nacisk')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        # 9. Boxplot porównawczy
        box_data = []
        box_labels = []
        for i in range(8):
            box_data.append(self.extractor.left_foot.iloc[:, i])
            box_labels.append(f'L{i+1}')
            box_data.append(self.extractor.right_foot.iloc[:, i])
            box_labels.append(f'R{i+1}')
        
        bp = axes[2, 2].boxplot(box_data, labels=box_labels, patch_artist=True)
        
        # Kolorowanie boxplotów
        for i, patch in enumerate(bp['boxes']):
            if i % 2 == 0:  # Lewe stopy
                patch.set_facecolor(self.colors['left'])
                patch.set_alpha(0.7)
            else:  # Prawe stopy
                patch.set_facecolor(self.colors['right'])
                patch.set_alpha(0.7)
        
        axes[2, 2].set_title('Rozkład Nacisku - Boxplot')
        axes[2, 2].set_xlabel('Czujnik')
        axes[2, 2].set_ylabel('Nacisk')
        axes[2, 2].tick_params(axis='x', rotation=45)
        axes[2, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_cop_trajectory(self, ax):
        """Pomocnicza funkcja do rysowania trajektorii COP"""
        # Uproszczone pozycje czujników
        sensor_positions = np.array([
            [0, 0], [1, 0], [2, 0], [3, 0],  # Przód
            [0, 1], [1, 1], [2, 1], [3, 1]   # Tył
        ])
        
        def calculate_cop(foot_data):
            cop = []
            for i in range(len(foot_data)):
                pressures = foot_data.iloc[i].values
                total_pressure = np.sum(pressures)
                
                if total_pressure > 0:
                    cop_x = np.sum(pressures * sensor_positions[:, 0]) / total_pressure
                    cop_y = np.sum(pressures * sensor_positions[:, 1]) / total_pressure
                    cop.append([cop_x, cop_y])
                else:
                    cop.append([0, 0])
            return np.array(cop)
        
        left_cop = calculate_cop(self.extractor.left_foot)
        right_cop = calculate_cop(self.extractor.right_foot)
        
        ax.plot(left_cop[:, 0], left_cop[:, 1], 
               color=self.colors['left'], alpha=0.7, linewidth=2, label='Lewa stopa')
        ax.plot(right_cop[:, 0], right_cop[:, 1], 
               color=self.colors['right'], alpha=0.7, linewidth=2, label='Prawa stopa')
        ax.set_title('Trajektoria Centrum Nacisku')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_symmetry_patterns_detailed(self, figsize=(20, 15)):
        """
        Szczegółowa wizualizacja wzorców symetrii
        """
        self._ensure_patterns()
        
        fig, axes = plt.subplots(4, 4, figsize=figsize)
        fig.suptitle('Szczegółowa Analiza Wzorców Symetrii', fontsize=18, fontweight='bold')
        
        # 1. Globalny wzorzec symetrii
        global_pattern = self.patterns['global_symmetry']
        metrics = ['mean', 'std', 'median', 'iqr', 'skewness', 'kurtosis']
        values = [global_pattern[metric] for metric in metrics]
        
        axes[0, 0].bar(metrics, values, color=self.colors['symmetry'], alpha=0.7)
        axes[0, 0].set_title('Globalny Wzorzec Symetrii')
        axes[0, 0].set_ylabel('Wartość')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Korelacje sensorów
        correlations = [self.patterns['sensor_patterns'][f'sensor_{i+1}']['correlation'] 
                       for i in range(8)]
        
        colors = ['green' if corr > 0.5 else 'orange' if corr > 0.3 else 'red' 
                 for corr in correlations]
        
        axes[0, 1].bar(range(8), correlations, color=colors, alpha=0.7)
        axes[0, 1].set_title('Korelacje Czujników L-R')
        axes[0, 1].set_xlabel('Czujnik')
        axes[0, 1].set_ylabel('Korelacja')
        axes[0, 1].set_xticks(range(8))
        axes[0, 1].set_xticklabels([f'S{i+1}' for i in range(8)])
        axes[0, 1].axhline(y=0.5, color='orange', linestyle='--', alpha=0.7)
        axes[0, 1].axhline(y=0.3, color='red', linestyle='--', alpha=0.7)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Stosunek symetrii dla czujników
        ratios = [self.patterns['sensor_patterns'][f'sensor_{i+1}']['symmetry_ratio_mean'] 
                 for i in range(8)]
        
        axes[0, 2].bar(range(8), ratios, color=self.colors['asymmetry'], alpha=0.7)
        axes[0, 2].axhline(y=1.0, color='green', linestyle='--', linewidth=2, 
                          label='Idealna symetria')
        axes[0, 2].set_title('Stosunek Symetrii Czujników')
        axes[0, 2].set_xlabel('Czujnik')
        axes[0, 2].set_ylabel('Stosunek L/R')
        axes[0, 2].set_xticks(range(8))
        axes[0, 2].set_xticklabels([f'S{i+1}' for i in range(8)])
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Centrum nacisku
        cop_pattern = self.patterns['cop_patterns']
        cop_metrics = ['cop_asymmetry_magnitude_mean', 'cop_asymmetry_magnitude_std',
                      'cop_path_length_left', 'cop_path_length_right']
        cop_values = [cop_pattern[metric] for metric in cop_metrics]
        
        axes[0, 3].bar(range(len(cop_metrics)), cop_values, 
                      color=self.colors['neutral'], alpha=0.7)
        axes[0, 3].set_title('Wzorzec Centrum Nacisku')
        axes[0, 3].set_ylabel('Wartość')
        axes[0, 3].set_xticks(range(len(cop_metrics)))
        axes[0, 3].set_xticklabels(['Asym\nMean', 'Asym\nStd', 'Path\nLeft', 'Path\nRight'])
        axes[0, 3].grid(True, alpha=0.3)
        
        # 5. Wzorzec dynamiczny
        dynamic_pattern = self.patterns['dynamic_patterns']
        
        trend_asym = [dynamic_pattern['trends'][f'sensor_{i+1}']['trend_asymmetry'] 
                     for i in range(8)]
        
        axes[1, 0].bar(range(8), trend_asym, color=self.colors['left'], alpha=0.7)
        axes[1, 0].set_title('Asymetria Trendów')
        axes[1, 0].set_xlabel('Czujnik')
        axes[1, 0].set_ylabel('Asymetria Trendu')
        axes[1, 0].set_xticks(range(8))
        axes[1, 0].set_xticklabels([f'S{i+1}' for i in range(8)])
        axes[1, 0].grid(True, alpha=0.3)
        
        # 6. Wzorzec fazowy
        phase_pattern = self.patterns['phase_patterns']
        phase_metrics = ['number_of_cycles', 'cycle_regularity', 
                        'average_cycle_length', 'cycle_length_std']
        phase_values = [phase_pattern[metric] for metric in phase_metrics]
        
        axes[1, 1].bar(range(len(phase_metrics)), phase_values, 
                      color=self.colors['right'], alpha=0.7)
        axes[1, 1].set_title('Wzorzec Fazowy')
        axes[1, 1].set_ylabel('Wartość')
        axes[1, 1].set_xticks(range(len(phase_metrics)))
        axes[1, 1].set_xticklabels(['Cycles', 'Regularity', 'Avg\nLength', 'Length\nStd'])
        axes[1, 1].grid(True, alpha=0.3)
        
        # 7. Dominujące częstotliwości
        spectral_pattern = self.patterns['spectral_patterns']
        freq_asym = [spectral_pattern[f'sensor_{i+1}']['dominant_freq_asymmetry'] 
                    for i in range(8)]
        
        axes[1, 2].bar(range(8), freq_asym, color=self.colors['symmetry'], alpha=0.7)
        axes[1, 2].set_title('Asymetria Częstotliwości')
        axes[1, 2].set_xlabel('Czujnik')
        axes[1, 2].set_ylabel('Asymetria Częst.')
        axes[1, 2].set_xticks(range(8))
        axes[1, 2].set_xticklabels([f'S{i+1}' for i in range(8)])
        axes[1, 2].grid(True, alpha=0.3)
        
        # 8. Wzorzec topologiczny
        topology_pattern = self.patterns['topology_patterns']
        topo_metrics = ['left_entropy', 'right_entropy', 'entropy_asymmetry', 
                       'kl_divergence']
        topo_values = [topology_pattern[metric] for metric in topo_metrics]
        
        axes[1, 3].bar(range(len(topo_metrics)), topo_values, 
                      color=self.colors['asymmetry'], alpha=0.7)
        axes[1, 3].set_title('Wzorzec Topologiczny')
        axes[1, 3].set_ylabel('Wartość')
        axes[1, 3].set_xticks(range(len(topo_metrics)))
        axes[1, 3].set_xticklabels(['L\nEntropy', 'R\nEntropy', 'Entropy\nAsym', 'KL\nDiv'])
        axes[1, 3].grid(True, alpha=0.3)
        
        # 9-16. Szczegółowe analizy dla wybranych czujników
        selected_sensors = [0, 3, 4, 7]  # Reprezentatywne czujniki
        
        for idx, sensor_idx in enumerate(selected_sensors):
            row = 2 + idx // 4
            col = idx % 4
            
            # Dane czujnika
            left_data = self.extractor.left_foot.iloc[:, sensor_idx]
            right_data = self.extractor.right_foot.iloc[:, sensor_idx]
            
            # Scatter plot
            axes[row, col].scatter(left_data, right_data, 
                                 color=self.colors['neutral'], alpha=0.6)
            
            # Linia idealnej symetrii
            max_val = max(left_data.max(), right_data.max())
            axes[row, col].plot([0, max_val], [0, max_val], 
                              'r--', linewidth=2, label='Idealna symetria')
            
            # Regresja liniowa
            slope, intercept, r_value, p_value, std_err = stats.linregress(left_data, right_data)
            line_x = np.linspace(0, max_val, 100)
            line_y = slope * line_x + intercept
            axes[row, col].plot(line_x, line_y, 
                              color=self.colors['symmetry'], linewidth=2, 
                              label=f'Regresja (r={r_value:.3f})')
            
            axes[row, col].set_title(f'Czujnik {sensor_idx+1} - Symetria')
            axes[row, col].set_xlabel('Lewa stopa')
            axes[row, col].set_ylabel('Prawa stopa')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_fingerprint_radar(self, figsize=(12, 8)):
        """
        Wykres radarowy odcisku palca symetrii
        """
        self._ensure_patterns()
        
        # Normalizacja danych odcisku palca
        fingerprint_data = self.fingerprint.copy()
        
        # Usuń wartości nieskończone i NaN
        for key, value in fingerprint_data.items():
            if not np.isfinite(value):
                fingerprint_data[key] = 0
        
        # Normalizacja do zakresu 0-1
        scaler = StandardScaler()
        values = list(fingerprint_data.values())
        normalized_values = scaler.fit_transform(np.array(values).reshape(-1, 1)).flatten()
        
        # Przeskalowanie do zakresu 0-1
        min_val = normalized_values.min()
        max_val = normalized_values.max()
        if max_val != min_val:
            normalized_values = (normalized_values - min_val) / (max_val - min_val)
        
        # Przygotowanie danych do wykresu radarowego
        categories = list(fingerprint_data.keys())
        values = normalized_values
        
        # Liczba zmiennych
        N = len(categories)
        
        # Kąty dla każdej zmiennej
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Zamknięcie koła
        
        # Dodanie pierwszej wartości na końcu
        values = np.concatenate([values, [values[0]]])
        
        # Tworzenie wykresu
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
        
        # Rysowanie obszaru
        ax.plot(angles, values, 'o-', linewidth=2, color=self.colors['symmetry'])
        ax.fill(angles, values, alpha=0.25, color=self.colors['symmetry'])
        
        # Etykiety
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([cat.replace('_', '\n') for cat in categories])
        
        # Ustawienia osi
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.grid(True)
        
        # Tytuł
        plt.title('Odcisk Palca Symetrii - Wykres Radarowy', 
                 size=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.show()
    
    def plot_sensor_heatmap(self, figsize=(12, 8)):
        """
        Heatmapa przedstawiająca układ czujników na stopach
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Średnie wartości dla każdego czujnika
        left_mean = self.extractor.left_foot.mean().values
        right_mean = self.extractor.right_foot.mean().values
        
        # Układ czujników (4x2 - przód i tył stopy)
        left_grid = left_mean.reshape(2, 4)
        right_grid = right_mean.reshape(2, 4)
        
        # Heatmapa lewej stopy
        im1 = ax1.imshow(left_grid, cmap='Blues', aspect='auto')
        ax1.set_title('Lewa Stopa - Średni Nacisk', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Pozycja (przód-tył)')
        ax1.set_ylabel('Pozycja (bok)')
        
        # Dodanie wartości na heatmapie
        for i in range(2):
            for j in range(4):
                ax1.text(j, i, f'{left_grid[i, j]:.1f}', 
                        ha='center', va='center', fontweight='bold')
        
        # Heatmapa prawej stopy
        im2 = ax2.imshow(right_grid, cmap='Reds', aspect='auto')
        ax2.set_title('Prawa Stopa - Średni Nacisk', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Pozycja (przód-tył)')
        ax2.set_ylabel('Pozycja (bok)')
        
        # Dodanie wartości na heatmapie
        for i in range(2):
            for j in range(4):
                ax2.text(j, i, f'{right_grid[i, j]:.1f}', 
                        ha='center', va='center', fontweight='bold')
        
        # Kolorbar
        fig.colorbar(im1, ax=ax1, shrink=0.6)
        fig.colorbar(im2, ax=ax2, shrink=0.6)
        
        plt.tight_layout()
        plt.show()
    
    def plot_time_series_analysis(self, figsize=(15, 10)):
        """
        Analiza szeregów czasowych dla wybranych czujników
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Analiza Szeregów Czasowych', fontsize=16, fontweight='bold')
        
        # Wybór reprezentatywnych czujników
        sensors_to_plot = [0, 3, 4, 7]  # Narożniki + środek
        
        for idx, sensor_idx in enumerate(sensors_to_plot):
            row = idx // 2
            col = idx % 2
            
            left_data = self.extractor.left_foot.iloc[:, sensor_idx]
            right_data = self.extractor.right_foot.iloc[:, sensor_idx]
            
            axes[row, col].plot(self.extractor.time, left_data, 
                              color=self.colors['left'], linewidth=2, 
                              label=f'L{sensor_idx+1}', alpha=0.8)
            axes[row, col].plot(self.extractor.time, right_data, 
                              color=self.colors['right'], linewidth=2, 
                              label=f'R{sensor_idx+1}', alpha=0.8)
            
            # Średnia ruchoma
            window_size = min(20, len(left_data) // 10)
            if window_size > 1:
                left_ma = left_data.rolling(window=window_size).mean()
                right_ma = right_data.rolling(window=window_size).mean()
                
                axes[row, col].plot(self.extractor.time, left_ma, 
                                  color=self.colors['left'], linewidth=3, 
                                  linestyle='--', alpha=0.7)
                axes[row, col].plot(self.extractor.time, right_ma, 
                                  color=self.colors['right'], linewidth=3, 
                                  linestyle='--', alpha=0.7)
            
            axes[row, col].set_title(f'Czujnik {sensor_idx+1}')
            axes[row, col].set_xlabel('Czas')
            axes[row, col].set_ylabel('Nacisk')
            axes[row, col].legend()
            