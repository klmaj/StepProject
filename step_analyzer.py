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
        
        min_samples = int(min_step_time * self.sampling_rate)
        valid_steps = []
        
        for start, end in zip(step_starts, step_ends):
            if end - start > min_samples:
                valid_steps.append((start, end))
        
        return valid_steps, total_force
    
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

    def get_average_step_period(self, side: str) -> float:
        _, periods = self.get_step_periods(side)
        return np.mean(periods)

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
            avg_step_duration = self.get_average_step_period('left')
            print(f"Średni czas kroku (lewa): {avg_step_duration:.3f} s")
        
        if len(right_steps) > 0:
            avg_step_duration = self.get_average_step_period('left')
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


    def find_step_edges(self, signal: np.ndarray, time: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        step_starts = np.where((signal[:-1] == 0) & (signal[1:] !=0))[0] + 1
        step_ends = np.where((signal[:-1] != 0) & (signal[1:] == 0))[0] + 1
    
        return step_starts, step_ends, time

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
    
    def compute_cross_correlation_matrix(self):
        """
        Oblicza macierz korelacji pomiędzy czujnikami lewej i prawej stopy (8x8).
        Zwraca:
            - corr_matrix: macierz 8x8 z wartościami korelacji Pearsona
        """
        corr_matrix = np.zeros((8, 8))
        for i in range(8):
            for j in range(8):
                corr, _ = pearsonr(self.right[:, i], self.left[:, j])
                corr_matrix[i, j] = corr
        return corr_matrix
    
    def compute_symmetry_indices(self):
        """
        Oblicza globalny i czujnikowy wskaźnik symetrii.
        Zwraca:
            {
                'global_symmetry': float,
                'sensor_symmetry': dict (np. {'1': 12.5, ..., '8': 4.3})
            }
        """
        left_total = np.sum(self.left, axis=1)
        right_total = np.sum(self.right, axis=1)

        # Globalna symetria
        mean_left = np.mean(left_total)
        mean_right = np.mean(right_total)

        if (mean_left + mean_right) == 0:
            global_symmetry = 0.0
        else:
            global_symmetry = 100 * 2 * abs(mean_left - mean_right) / (mean_left + mean_right)

        # Symetria dla każdego czujnika
        sensor_symmetry = {}
        for i in range(8):
            l_mean = np.mean(self.left[:, i])
            r_mean = np.mean(self.right[:, i])
            if (l_mean + r_mean) == 0:
                index = 0.0
            else:
                index = 100 * 2 * abs(l_mean - r_mean) / (l_mean + r_mean)
            sensor_symmetry[str(i + 1)] = index

        return {
            'global_symmetry': global_symmetry,
            'sensor_symmetry': sensor_symmetry
        }


    def correlation_left_right_by_step(self, step_range=(0, 0), method='mean'):
        """
        Oblicza korelację między lewą i prawą stopą w krokach od n do m,
        na podstawie średnich lub maksymalnych sygnałów z czujników.

        Params:
            analyzer: FootSensorAnalyzer
            step_range: (n, m) — indeksy kroków (włącznie)
            method: 'mean' lub 'max' — jak agregować dane w kroku

        Returns:
            dict: { 'czujnik_1': korelacja, ..., 'czujnik_8': korelacja }
        """
        left_steps, _ = self.detect_steps('left')
        right_steps, _ = self.detect_steps('right')

        n, m = step_range
        results = {f'czujnik_{i+1}': [] for i in range(8)}

        for step_idx in range(n, m + 1):
            if step_idx >= len(left_steps) or step_idx >= len(right_steps):
                continue  # pomiń jeśli brakuje kroku

            l_start, l_end = left_steps[step_idx]
            r_start, r_end = right_steps[step_idx]

            left = self.left[l_start:l_end]
            right = self.right[r_start:r_end]

            for i in range(8):
                l_sig = left[:, i]
                r_sig = right[:, i]

                if method == 'mean':
                    l_val = np.mean(l_sig)
                    r_val = np.mean(r_sig)
                elif method == 'max':
                    l_val = np.max(l_sig)
                    r_val = np.max(r_sig)
                else:
                    raise ValueError("method must be 'mean' or 'max'")

                results[f'czujnik_{i+1}'].append((l_val, r_val))

        # Teraz policz korelację dla każdej pary sygnałów
        corr_result = {}
        for sensor, pairs in results.items():
            if len(pairs) >= 2:
                l_vals, r_vals = zip(*pairs)
                corr, _ = pearsonr(l_vals, r_vals)
                corr_result[sensor] = corr
            else:
                corr_result[sensor] = np.nan  # za mało danych

        return corr_result


    def full_correlation_matrix_by_step(self, step_range=(0, 0), method='mean'):
        """
        Oblicza pełną macierz korelacji 8x8 między czujnikami lewej i prawej stopy
        w krokach od n do m (włącznie), na podstawie średnich lub maksymalnych wartości.

        Zwraca:
            macierz numpy (8x8) z korelacjami
        """
        left_steps, _ = self.detect_steps('left')
        right_steps, _ = self.detect_steps('right')
        n, m = step_range

        left_features = []
        right_features = []

        for step_idx in range(n, m + 1):
            if step_idx >= len(left_steps) or step_idx >= len(right_steps):
                continue

            l_start, l_end = left_steps[step_idx]
            r_start, r_end = right_steps[step_idx]

            left = self.left[l_start:l_end]
            right = self.right[r_start:r_end]

            if method == 'mean':
                left_vec = np.mean(left, axis=0)
                right_vec = np.mean(right, axis=0)
            elif method == 'max':
                left_vec = np.max(left, axis=0)
                right_vec = np.max(right, axis=0)
            else:
                raise ValueError("method must be 'mean' or 'max'")

            left_features.append(left_vec)
            right_features.append(right_vec)

        left_mat = np.array(left_features)
        right_mat = np.array(right_features)

        corr_matrix = np.zeros((8, 8))
        for i in range(8):
            for j in range(8):
                l_col = left_mat[:, i]
                r_col = right_mat[:, j]
                if len(l_col) >= 2 and np.std(l_col) > 0 and np.std(r_col) > 0:
                    corr, _ = pearsonr(l_col, r_col)
                    corr_matrix[i, j] = corr
                else:
                    corr_matrix[i, j] = np.nan

        return corr_matrix

    def full_si_map(self, left_mat, right_mat):
        """
        Oblicza pełną macierz Symmetry Index (8x8) dla czujników lewej i prawej stopy.
        SI = ((L - P) / ((L + P)/2)) * 100%
        """
        si_matrix = np.zeros((8, 8))

        for i in range(8):  # czujniki lewej stopy
            for j in range(8):  # czujniki prawej stopy
                l = left_mat[:, i]
                r = right_mat[:, j]
                mask = (~np.isnan(l)) & (~np.isnan(r))
                if np.sum(mask) == 0:
                    si_matrix[i, j] = np.nan
                else:
                    l_valid = l[mask]
                    r_valid = r[mask]
                    si_vals = ((l_valid - r_valid) / ((l_valid + r_valid) / 2)) * 100
                    si_matrix[i, j] = np.mean(si_vals)

        return si_matrix

    def full_mapd_map(self, left_mat, right_mat):
        """
        Oblicza pełną macierz MAPD (8x8) między czujnikami lewej i prawej stopy.
        MAPD = mean( |L - P| / ((L + P)/2) ) * 100%
        """
        mapd_matrix = np.zeros((8, 8))

        for i in range(8):
            for j in range(8):
                l = left_mat[:, i]
                r = right_mat[:, j]
                mask = (~np.isnan(l)) & (~np.isnan(r))
                if np.sum(mask) == 0:
                    mapd_matrix[i, j] = np.nan
                else:
                    l_valid = l[mask]
                    r_valid = r[mask]
                    mapd_vals = (np.abs(l_valid - r_valid) / ((l_valid + r_valid) / 2)) * 100
                    mapd_matrix[i, j] = np.mean(mapd_vals)

        return mapd_matrix

    def get_max_signal_matrix(self, step_range=(0, 0)):
        left_steps, _ = self.detect_steps('left')
        right_steps, _ = self.detect_steps('right')
        n, m = step_range

        left_features = []
        right_features = []

        for step_idx in range(n, m + 1):
            if step_idx < len(left_steps) and step_idx < len(right_steps):
                l_start, l_end = left_steps[step_idx]
                r_start, r_end = right_steps[step_idx]
                
                # Weź fragment sygnału z danego kroku
                left = self.left[l_start:l_end]
                right = self.right[r_start:r_end]
                
                # Użyj np. wartości maksymalnych (możesz zmienić na mean)
                left_vec = np.max(left, axis=0)
                right_vec = np.max(right, axis=0)

                left_features.append(left_vec)
                right_features.append(right_vec)

        # 3. Konwersja do macierzy (kroki x 8)
        left_mat = np.array(left_features)
        right_mat = np.array(right_features)

        return left_mat, right_mat
    
    def compute_global_signals(self):
        """
        Oblicza globalny nacisk lewej i prawej stopy w czasie
        jako sumę sygnałów z 8 czujników (czyli siłę całkowitą).
        
        Returns:
            global_left, global_right – 1D numpy arrays (czas,)
        """
        global_left = np.sum(self.left, axis=1)
        global_right = np.sum(self.right, axis=1)
        return global_left, global_right

    def compute_global_mapd(self, global_left, global_right):
        """
        Oblicza MAPD (Mean Absolute Percentage Difference) w czasie
        oraz jego średnią wartość dla całego sygnału.

        Returns:
            mapd_time – MAPD w czasie (1D array)
            mapd_mean – średni MAPD (float)
        """
        denominator = (global_left + global_right) / 2
        mask = denominator > 0  # unikamy dzielenia przez 0
        mapd_time = np.zeros_like(global_left)
        mapd_time[mask] = (np.abs(global_left[mask] - global_right[mask]) / denominator[mask]) * 100
        mapd_mean = np.mean(mapd_time[mask])
        return mapd_time, mapd_mean
