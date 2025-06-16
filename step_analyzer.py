import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class FootSensorAnalyzer:
    def __init__(self, time, left_foot, right_foot, sheet_name=""):
        self.time = np.array(time)
        self.left = np.array(left_foot)
        self.right = np.array(right_foot)
        self.sheet_name = sheet_name
        self.left_sensors = self.left  # 8 czujników lewej stopy
        self.right_sensors = self.right  # 8 czujników prawej stopy
        self.left_total = None   # suma sygnałów lewej stopy
        self.right_total = None  # suma sygnałów prawej stopy
        self.steps_left = []
        self.steps_right = []
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

    def trim_signal_edges(self, signal: np.ndarray, time: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Usuwa tylko początkowe i końcowe zera z sygnału i odpowiadającego czasu, ale zostawia zera w środku"""
        # Szukamy pierwszego i ostatniego indeksu, gdzie sygnał jest różny od zera
        non_zero_indices = np.flatnonzero(signal != 0)
    
        if len(non_zero_indices) == 0:
            return np.array([]), np.array([])
    
        start_idx = max(non_zero_indices[0]-1,0)
        end_idx = min(non_zero_indices[-1] + 2, len(signal))  # +1 żeby zachować ten ostatni indeks
    
        # Zwracamy fragmenty sygnału i czasu — ze środkowymi zerami, ale bez zer z przodu i z tyłu
        return signal[start_idx:end_idx], time[start_idx:end_idx]

    def find_step_edges(self, signal: np.ndarray, time: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # najpierw przytnij sygnał
        signal_trimmed, time_trimmed = self.trim_signal_edges(signal, time)
    
        # teraz znajdź początek i koniec kroków na przyciętym sygnale
        step_starts = np.where((signal_trimmed[:-1] == 0) & (signal_trimmed[1:] > 0))[0] + 1
        step_ends = np.where((signal_trimmed[:-1] > 0) & (signal_trimmed[1:] == 0))[0] + 1
    
        return step_starts, step_ends, time_trimmed

    def get_step_periods(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Oblicza okresy kroków (czas pomiędzy kolejnymi krokami) dla lewej lub prawej stopy.

        Parametry:
        - side: 'left' lub 'right'

        Zwraca:
        - times: czasy rozpoczęcia kroków
        - periods: czas trwania kolejnych kroków
        """

        if side == 'left':
            signal = self.left[:,0]
        elif side == 'right':
            signal = self.right[:,0]
        else:
            raise ValueError("side must be 'left' or 'right'")

        times = self.time

        step_starts, _, trimmed_time = self.find_step_edges(signal, times)

        step_times = trimmed_time[step_starts]
        periods = np.diff(step_times)

        return step_times[:-1], periods  # Ostatni start nie ma kolejnego kroku

    def mean_max_pressure_per_step(self, side='left'):
        """
        Oblicza średnią maksymalną wartość nacisku na każdym czujniku dla kroków danej stopy.

        Zwraca:
        - mean_max_per_sensor: np.ndarray, shape (liczba_czujników,), średnie maksima z kroków
        """

        if side == 'left':
            data = self.left  # zakładam, że shape (czas, czujniki)
        elif side == 'right':
            data = self.right
        else:
            raise ValueError("side musi być 'left' lub 'right'")

        # Użyj sygnału z jednego czujnika do wykrywania kroków, np. pierwszego
        signal = data[:, 0]
        time = self.time

        step_starts, step_ends, time_trimmed = self.find_step_edges(signal, time)

        # Jeśli kroki się nie dopasowują (np. różna długość), obetnij
        n_steps = min(len(step_starts), len(step_ends))

        max_values_per_step = []

        for i in range(n_steps):
            start_idx = step_starts[i]
            end_idx = step_ends[i]

            # Wyciągnij fragment danych dla kroku (wszystkie czujniki)
            step_data = data[start_idx:end_idx, :]  # shape (czas_w_kroku, czujniki)

            # Maksimum nacisku na każdym czujniku w tym kroku
            max_per_sensor = np.max(step_data, axis=0)  # shape (czujniki,)
            max_values_per_step.append(max_per_sensor)

        max_values_per_step = np.array(max_values_per_step)  # shape (liczba_kroków, czujniki)

        # Oblicz średnią maksymalną wartość per czujnik (średnia po krokach)
        mean_max_per_sensor = np.mean(max_values_per_step, axis=0)

        return mean_max_per_sensor


    def analyze_multi_sensor_symmetry(self):
        """Zaawansowana analiza symetrii dla wszystkich czujników"""
        symmetry = {}
        
        # Symetria dla każdego czujnika
        sensor_symmetries = []
        for i in range(8):
            left_mean = np.mean(self.left_sensors[:, i])
            right_mean = np.mean(self.right_sensors[:, i])
            
            if left_mean + right_mean > 0:
                sym_index = 2 * abs(left_mean - right_mean) / (left_mean + right_mean) * 100
                sensor_symmetries.append(sym_index)
            else:
                sensor_symmetries.append(0)
                
        symmetry['sensor_symmetry_indices'] = dict(zip(self.sensor_names, sensor_symmetries))
        
        # Ogólna symetria sumaryczna
        total_sym = 2 * abs(np.mean(self.left_total) - np.mean(self.right_total)) / (np.mean(self.left_total) + np.mean(self.right_total)) * 100
        symmetry['total_symmetry_index'] = total_sym
        
        # Korelacja między odpowiadającymi sobie czujnikami
        sensor_correlations = []
        for i in range(8):
            corr = np.corrcoef(self.left_sensors[:, i], self.right_sensors[:, i])[0, 1]
            sensor_correlations.append(corr)
            
        symmetry['sensor_correlations'] = dict(zip(self.sensor_names, sensor_correlations))
        
        # PCA dla analizy głównych wzorców asymetrii
        combined_sensors = np.column_stack([self.left_sensors, self.right_sensors])
        scaler = StandardScaler()
        scaled_sensors = scaler.fit_transform(combined_sensors)
        
        pca = PCA(n_components=4)
        pca_result = pca.fit_transform(scaled_sensors)
        
        symmetry['pca_explained_variance'] = pca.explained_variance_ratio_
        symmetry['pca_components'] = pca.components_
        
        return symmetry