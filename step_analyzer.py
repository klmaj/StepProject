import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

class FootSensorAnalyzer:
    def __init__(self, time, left_foot, right_foot, sheet_name=""):
        self.time = np.array(time)
        self.left = np.array(left_foot)
        self.right = np.array(right_foot)
        self.sheet_name = sheet_name
        self.sampling_rate = self._calculate_sampling_rate()
        
        # Nazwy czujników (możesz dostosować do swojego układu)
        self.sensor_names = [
            '1', '2', '3', '4', 
            '5', '6', '7', '8'
        ]
    
    def _calculate_sampling_rate(self):
        """Oblicza częstotliwość próbkowania"""
        print(len(self.time))
        if len(self.time) > 1:
            dt = np.mean(np.diff(self.time))
            return 1.0 / dt
        return None
    
    
    def detect_steps(self, foot='left', threshold_percent=20, min_step_time=0.3):
        """Wykrywa kroki na podstawie sumy sygnałów z wszystkich czujników"""
        if foot == 'left':
            data = self.left
        else:
            data = self.right
        
        # Suma sygnałów ze wszystkich czujników
        total_force = np.sum(data, axis=1)
        
        # Próg jako procent maksymalnej wartości
        threshold = np.max(total_force) * (threshold_percent / 100)
        
        # Znajdź miejsca gdzie siła przekracza próg
        above_threshold = total_force > threshold
        
        # Znajdź początki i końce kroków
        step_starts = []
        step_ends = []
        in_step = False
        
        for i, above in enumerate(above_threshold):
            if above and not in_step:
                step_starts.append(i)
                in_step = True
            elif not above and in_step:
                step_ends.append(i)
                in_step = False
        
        # Filtruj kroki które są za krótkie
        print(self.sampling_rate)
        min_samples = int(min_step_time * self.sampling_rate)
        valid_steps = []
        
        for start, end in zip(step_starts, step_ends):
            if end - start > min_samples:
                valid_steps.append((start, end))
        
        return valid_steps, total_force
    
    def analyze_step_phases(self, foot='left', step_indices=None):
        """Analizuje fazy kroku: kontakt pięty, pełny kontakt, odbicie palców"""
        if step_indices is None:
            step_indices, _ = self.detect_steps(foot)
        
        if foot == 'left':
            data = self.left
        else:
            data = self.right
        
        step_phases = []
        
        for start_idx, end_idx in step_indices:
            step_data = data[start_idx:end_idx]
            step_time = self.time[start_idx:end_idx]
            
            # Analiza faz na podstawie rozkładu aktywności czujników
            heel_sensors = np.mean(step_data[:, 0:2], axis=1)  # Pięta
            mid_sensors = np.mean(step_data[:, 2:4], axis=1)   # Środek
            toe_sensors = np.mean(step_data[:, 4:8], axis=1)   # Palce
            
            # Znajdź maksima dla każdej strefy
            heel_max_idx = np.argmax(heel_sensors)
            toe_max_idx = np.argmax(toe_sensors)
            
            phases = {
                'heel_contact': heel_max_idx,
                'toe_off': toe_max_idx,
                'step_duration': step_time[-1] - step_time[0],
                'heel_force_max': np.max(heel_sensors),
                'toe_force_max': np.max(toe_sensors),
                'total_force_max': np.max(np.sum(step_data, axis=1))
            }
            
            step_phases.append(phases)
        
        return step_phases
    
    def calculate_pressure_distribution(self, foot='left'):
        """Oblicza rozkład nacisku dla każdego czujnika"""
        if foot == 'left':
            data = self.left
        else:
            data = self.right
        
        # Statystyki dla każdego czujnika
        stats = {}
        for i, sensor_name in enumerate(self.sensor_names):
            sensor_data = data[:, i]
            stats[sensor_name] = {
                'mean': np.mean(sensor_data),
                'max': np.max(sensor_data),
                'std': np.std(sensor_data),
                'total_impulse': np.trapz(sensor_data, self.time)
            }
        
        return stats
    
    def plot_raw_data(self, figsize=(15, 10)):
        """Wykres surowych danych z wszystkich czujników"""
        fig, axes = plt.subplots(2, 4, figsize=figsize)
        fig.suptitle(f'Surowe dane z czujników - {self.sheet_name}', fontsize=16)
        
        for i, sensor_name in enumerate(self.sensor_names):
            row = i // 4
            col = i % 4
            
            axes[row, col].plot(self.time, self.left[:, i], 'b-', label='Lewa stopa', linewidth=1)
            axes[row, col].plot(self.time, self.right[:, i], 'r-', label='Prawa stopa', linewidth=1)
            axes[row, col].set_title(f'Czujnik: {sensor_name}')
            axes[row, col].set_xlabel('Czas [s]')
            axes[row, col].set_ylabel('Siła')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        """Generuje podstawowy raport z analizy"""
        print(f"=== RAPORT ANALIZY CZUJNIKÓW STOPY ===")
        print(f"Arkusz: {self.sheet_name}")
        print(f"Czas trwania pomiaru: {self.time[-1] - self.time[0]:.2f} s")
        print(f"Częstotliwość próbkowania: {self.sampling_rate:.1f} Hz")
        print(f"Liczba próbek: {len(self.time)}")
        
        # Analiza kroków
        left_steps, _ = self.detect_steps('left')
        right_steps, _ = self.detect_steps('right')
        
        print(f"\n=== ANALIZA KROKÓW ===")
        print(f"Kroki lewa stopa: {len(left_steps)}")
        print(f"Kroki prawa stopa: {len(right_steps)}")
        
        if len(left_steps) > 0:
            left_phases = self.analyze_step_phases('left', left_steps)
            avg_step_duration = np.mean([phase['step_duration'] for phase in left_phases])
            print(f"Średni czas kroku (lewa): {avg_step_duration:.3f} s")
        
        if len(right_steps) > 0:
            right_phases = self.analyze_step_phases('right', right_steps)
            avg_step_duration = np.mean([phase['step_duration'] for phase in right_phases])
            print(f"Średni czas kroku (prawa): {avg_step_duration:.3f} s")
        
        # Rozkład nacisku
        print(f"\n=== ROZKŁAD NACISKU ===")
        left_dist = self.calculate_pressure_distribution('left')
        right_dist = self.calculate_pressure_distribution('right')
        
        print("Lewa stopa - najwyższe średnie naciski:")
        sorted_left = sorted(left_dist.items(), key=lambda x: x[1]['mean'], reverse=True)
        for sensor, stats in sorted_left[:3]:
            print(f"  {sensor}: {stats['mean']:.2f} (max: {stats['max']:.2f})")
        
        print("Prawa stopa - najwyższe średnie naciski:")
        sorted_right = sorted(right_dist.items(), key=lambda x: x[1]['mean'], reverse=True)
        for sensor, stats in sorted_right[:3]:
            print(f"  {sensor}: {stats['mean']:.2f} (max: {stats['max']:.2f})")

    def generate_report(self):
        """Generuje podstawowy raport z analizy"""
        print(f"=== RAPORT ANALIZY CZUJNIKÓW STOPY ===")
        print(f"Arkusz: {self.sheet_name}")
        print(f"Czas trwania pomiaru: {self.time[-1] - self.time[0]:.2f} s")
        print(f"Częstotliwość próbkowania: {self.sampling_rate:.1f} Hz")
        print(f"Liczba próbek: {len(self.time)}")
        
        # Analiza kroków
        left_steps, _ = self.detect_steps('left')
        right_steps, _ = self.detect_steps('right')
        
        print(f"\n=== ANALIZA KROKÓW ===")
        print(f"Kroki lewa stopa: {len(left_steps)}")
        print(f"Kroki prawa stopa: {len(right_steps)}")
        
        if len(left_steps) > 0:
            left_phases = self.analyze_step_phases('left', left_steps)
            avg_step_duration = np.mean([phase['step_duration'] for phase in left_phases])
            print(f"Średni czas kroku (lewa): {avg_step_duration:.3f} s")
        
        if len(right_steps) > 0:
            right_phases = self.analyze_step_phases('right', right_steps)
            avg_step_duration = np.mean([phase['step_duration'] for phase in right_phases])
            print(f"Średni czas kroku (prawa): {avg_step_duration:.3f} s")
        
        # Rozkład nacisku
        print(f"\n=== ROZKŁAD NACISKU ===")
        left_dist = self.calculate_pressure_distribution('left')
        right_dist = self.calculate_pressure_distribution('right')
        
        print("Lewa stopa - najwyższe średnie naciski:")
        sorted_left = sorted(left_dist.items(), key=lambda x: x[1]['mean'], reverse=True)
        for sensor, stats in sorted_left[:3]:
            print(f"  {sensor}: {stats['mean']:.2f} (max: {stats['max']:.2f})")
        
        print("Prawa stopa - najwyższe średnie naciski:")
        sorted_right = sorted(right_dist.items(), key=lambda x: x[1]['mean'], reverse=True)
        for sensor, stats in sorted_right[:3]:
            print(f"  {sensor}: {stats['mean']:.2f} (max: {stats['max']:.2f})")