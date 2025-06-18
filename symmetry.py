import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class EnhancedFootSensorAnalyzer:
    def __init__(self, time, left_foot, right_foot, sheet_name=""):
        self.time = np.array(time)
        self.left = np.array(left_foot)
        self.right = np.array(right_foot)
        self.sheet_name = sheet_name
        self.left_sensors = self.left
        self.right_sensors = self.right
        self.sampling_rate = self._calculate_sampling_rate()
        
        # Pozycje czujników na stopie (w mm, układ anatomiczny)
        # Zakładam układ: pięta -> palce, przyśrodkowo -> boczno
        self.sensor_positions = np.array([
            [0, 30],    # 1: Pięta przyśrodkowa
            [0, -30],   # 2: Pięta boczna
            [60, 30],   # 3: Środstopie przyśrodkowe
            [60, -30],  # 4: Środstopie boczne
            [120, 30],  # 5: Przodstopie przyśrodkowe
            [120, -30], # 6: Przodstopie boczne
            [180, 0],   # 7: Paluch
            [170, -20]  # 8: Palce 2-5
        ])
        
        self.sensor_names = ['1', '2', '3', '4', '5', '6', '7', '8']
        
    def _calculate_sampling_rate(self):
        """Oblicza częstotliwość próbkowania"""
        if len(self.time) > 1:
            dt = np.mean(np.diff(self.time))
            return 1.0 / dt
        return None
    
    def calculate_cop_trajectory(self, foot='both'):
        """
        Oblicza trajektorię centrum nacisku (Center of Pressure)
        
        Args:
            foot: 'left', 'right', lub 'both'
            
        Returns:
            dict z trajektoriami COP dla wybranej stopy/stóp
        """
        results = {}
        
        if foot in ['left', 'both']:
            cop_x_left, cop_y_left = self._calculate_single_cop(self.left)
            results['left'] = {
                'cop_x': cop_x_left,
                'cop_y': cop_y_left,
                'velocity': self._calculate_cop_velocity(cop_x_left, cop_y_left),
                'path_length': self._calculate_path_length(cop_x_left, cop_y_left),
                'sway_area': self._calculate_sway_area(cop_x_left, cop_y_left)
            }
        
        if foot in ['right', 'both']:
            cop_x_right, cop_y_right = self._calculate_single_cop(self.right)
            results['right'] = {
                'cop_x': cop_x_right,
                'cop_y': cop_y_right,
                'velocity': self._calculate_cop_velocity(cop_x_right, cop_y_right),
                'path_length': self._calculate_path_length(cop_x_right, cop_y_right),
                'sway_area': self._calculate_sway_area(cop_x_right, cop_y_right)
            }
        
        return results
    
    def _calculate_single_cop(self, foot_data):
        """Oblicza COP dla jednej stopy"""
        cop_x = []
        cop_y = []
        
        for i in range(len(foot_data)):
            forces = foot_data[i]
            total_force = np.sum(forces)
            
            if total_force > 0:
                # Średnia ważona pozycji czujników
                cop_x_i = np.sum(forces * self.sensor_positions[:, 0]) / total_force
                cop_y_i = np.sum(forces * self.sensor_positions[:, 1]) / total_force
                cop_x.append(cop_x_i)
                cop_y.append(cop_y_i)
            else:
                cop_x.append(np.nan)
                cop_y.append(np.nan)
        
        return np.array(cop_x), np.array(cop_y)
    
    def _calculate_cop_velocity(self, cop_x, cop_y):
        """Oblicza prędkość przemieszczania się COP"""
        # Usuń NaN
        valid_indices = ~(np.isnan(cop_x) | np.isnan(cop_y))
        if np.sum(valid_indices) < 2:
            return np.array([])
        
        valid_x = cop_x[valid_indices]
        valid_y = cop_y[valid_indices]
        valid_time = self.time[valid_indices]
        
        dx_dt = np.gradient(valid_x, valid_time)
        dy_dt = np.gradient(valid_y, valid_time)
        
        velocity = np.sqrt(dx_dt**2 + dy_dt**2)
        return velocity
    
    def _calculate_path_length(self, cop_x, cop_y):
        """Oblicza całkowitą długość ścieżki COP"""
        valid_indices = ~(np.isnan(cop_x) | np.isnan(cop_y))
        if np.sum(valid_indices) < 2:
            return 0
        
        valid_x = cop_x[valid_indices]
        valid_y = cop_y[valid_indices]
        
        distances = np.sqrt(np.diff(valid_x)**2 + np.diff(valid_y)**2)
        return np.sum(distances)
    
    def _calculate_sway_area(self, cop_x, cop_y):
        """Oblicza obszar wychwiań COP (elipsa zawierająca 95% punktów)"""
        valid_indices = ~(np.isnan(cop_x) | np.isnan(cop_y))
        if np.sum(valid_indices) < 3:
            return 0
        
        valid_x = cop_x[valid_indices]
        valid_y = cop_y[valid_indices]
        
        # Macierz kowariancji
        cov_matrix = np.cov(valid_x, valid_y)
        eigenvals, eigenvecs = np.linalg.eig(cov_matrix)
        
        # Półosie elipsy dla 95% danych (chi-square dla 2 DOF, p=0.95)
        chi2_95 = 5.991
        semi_major = np.sqrt(chi2_95 * np.max(eigenvals))
        semi_minor = np.sqrt(chi2_95 * np.min(eigenvals))
        
        area = np.pi * semi_major * semi_minor
        return area
    
    def analyze_foot_correlation(self, method='pearson'):
        """
        Analizuje korelację między lewą a prawą stopą
        
        Args:
            method: 'pearson' lub 'spearman'
            
        Returns:
            dict z wynikami analizy korelacji
        """
        results = {
            'sensor_correlations': {},
            'total_force_correlation': 0,
            'cop_correlations': {},
            'temporal_correlations': {},
            'method': method
        }
        
        # 1. Korelacja dla każdego czujnika
        for i, sensor_name in enumerate(self.sensor_names):
            left_sensor = self.left[:, i]
            right_sensor = self.right[:, i]
            
            if method == 'pearson':
                corr, p_value = pearsonr(left_sensor, right_sensor)
            else:
                corr, p_value = spearmanr(left_sensor, right_sensor)
            
            results['sensor_correlations'][sensor_name] = {
                'correlation': corr,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        
        # 2. Korelacja sił całkowitych
        left_total = np.sum(self.left, axis=1)
        right_total = np.sum(self.right, axis=1)
        
        if method == 'pearson':
            total_corr, total_p = pearsonr(left_total, right_total)
        else:
            total_corr, total_p = spearmanr(left_total, right_total)
        
        results['total_force_correlation'] = {
            'correlation': total_corr,
            'p_value': total_p,
            'significant': total_p < 0.05
        }
        
        # 3. Korelacja COP
        cop_data = self.calculate_cop_trajectory('both')
        if 'left' in cop_data and 'right' in cop_data:
            left_cop_x = cop_data['left']['cop_x']
            left_cop_y = cop_data['left']['cop_y']
            right_cop_x = cop_data['right']['cop_x']
            right_cop_y = cop_data['right']['cop_y']
            
            # Usuń NaN przed korelacją
            valid_mask = ~(np.isnan(left_cop_x) | np.isnan(left_cop_y) | 
                          np.isnan(right_cop_x) | np.isnan(right_cop_y))
            
            if np.sum(valid_mask) > 10:
                if method == 'pearson':
                    cop_x_corr, cop_x_p = pearsonr(left_cop_x[valid_mask], right_cop_x[valid_mask])
                    cop_y_corr, cop_y_p = pearsonr(left_cop_y[valid_mask], right_cop_y[valid_mask])
                else:
                    cop_x_corr, cop_x_p = spearmanr(left_cop_x[valid_mask], right_cop_x[valid_mask])
                    cop_y_corr, cop_y_p = spearmanr(left_cop_y[valid_mask], right_cop_y[valid_mask])
                
                results['cop_correlations'] = {
                    'x_direction': {'correlation': cop_x_corr, 'p_value': cop_x_p},
                    'y_direction': {'correlation': cop_y_corr, 'p_value': cop_y_p}
                }
        
        # 4. Korelacja z opóźnieniem czasowym (cross-correlation)
        results['temporal_correlations'] = self._calculate_cross_correlation(left_total, right_total)
        
        return results
    
    def _calculate_cross_correlation(self, signal1, signal2):
        """Oblicza korelację krzyżową z różnymi opóźnieniami"""
        # Normalizacja sygnałów
        signal1_norm = (signal1 - np.mean(signal1)) / np.std(signal1)
        signal2_norm = (signal2 - np.mean(signal2)) / np.std(signal2)
        
        # Korelacja krzyżowa
        cross_corr = np.correlate(signal1_norm, signal2_norm, mode='full')
        
        # Opóźnienia w próbkach
        lags = np.arange(-len(signal2) + 1, len(signal1))
        
        # Znajdź maksymalną korelację i odpowiadające opóźnienie
        max_corr_idx = np.argmax(np.abs(cross_corr))
        max_correlation = cross_corr[max_corr_idx]
        optimal_lag = lags[max_corr_idx]
        
        # Przelicz na czas
        optimal_lag_time = optimal_lag / self.sampling_rate
        
        return {
            'max_correlation': max_correlation,
            'optimal_lag_samples': optimal_lag,
            'optimal_lag_time': optimal_lag_time,
            'cross_correlation': cross_corr,
            'lags': lags
        }
    
    def analyze_symmetry(self):
        """
        Kompleksowa analiza symetrii między stopami
        
        Returns:
            dict z różnymi wskaźnikami symetrii
        """
        results = {
            'force_symmetry': {},
            'temporal_symmetry': {},
            'spatial_symmetry': {},
            'cop_symmetry': {},
            'overall_asymmetry_index': 0
        }
        
        # 1. Symetria sił
        results['force_symmetry'] = self._analyze_force_symmetry()
        
        # 2. Symetria czasowa (timing)
        results['temporal_symmetry'] = self._analyze_temporal_symmetry()
        
        # 3. Symetria przestrzenna (rozkład nacisku)
        results['spatial_symmetry'] = self._analyze_spatial_symmetry()
        
        # 4. Symetria COP
        results['cop_symmetry'] = self._analyze_cop_symmetry()
        
        # 5. Ogólny wskaźnik asymetrii
        results['overall_asymmetry_index'] = self._calculate_overall_asymmetry(results)
        
        return results
    
    def _analyze_force_symmetry(self):
        """Analiza symetrii sił"""
        left_total = np.sum(self.left, axis=1)
        right_total = np.sum(self.right, axis=1)
        
        # Wskaźnik symetrii = 2 * |L - R| / (L + R) * 100%
        mean_left = np.mean(left_total)
        mean_right = np.mean(right_total)
        
        if mean_left + mean_right > 0:
            symmetry_index = 2 * abs(mean_left - mean_right) / (mean_left + mean_right) * 100
        else:
            symmetry_index = 0
        
        # Symetria dla każdego czujnika
        sensor_symmetries = {}
        for i, sensor_name in enumerate(self.sensor_names):
            left_sensor = np.mean(self.left[:, i])
            right_sensor = np.mean(self.right[:, i])
            
            if left_sensor + right_sensor > 0:
                sensor_sym = 2 * abs(left_sensor - right_sensor) / (left_sensor + right_sensor) * 100
            else:
                sensor_sym = 0
            
            sensor_symmetries[sensor_name] = sensor_sym
        
        return {
            'total_force_symmetry_index': symmetry_index,
            'sensor_symmetries': sensor_symmetries,
            'left_dominance': mean_left > mean_right,
            'dominance_ratio': max(mean_left, mean_right) / min(mean_left, mean_right) if min(mean_left, mean_right) > 0 else float('inf')
        }
    
    def _analyze_temporal_symmetry(self):
        """Analiza symetrii czasowej"""
        # Wykryj kroki dla obu stóp
        left_steps, _ = self.detect_steps('left')
        right_steps, _ = self.detect_steps('right')
        
        if len(left_steps) == 0 or len(right_steps) == 0:
            return {'error': 'Nie wykryto kroków dla jednej lub obu stóp'}
        
        # Czasy trwania kroków
        left_durations = [(self.time[end] - self.time[start]) for start, end in left_steps]
        right_durations = [(self.time[end] - self.time[start]) for start, end in right_steps]
        
        mean_left_duration = np.mean(left_durations)
        mean_right_duration = np.mean(right_durations)
        
        # Wskaźnik symetrii czasowej
        if mean_left_duration + mean_right_duration > 0:
            temporal_symmetry = 2 * abs(mean_left_duration - mean_right_duration) / (mean_left_duration + mean_right_duration) * 100
        else:
            temporal_symmetry = 0
        
        return {
            'temporal_symmetry_index': temporal_symmetry,
            'left_step_duration_mean': mean_left_duration,
            'right_step_duration_mean': mean_right_duration,
            'left_step_count': len(left_steps),
            'right_step_count': len(right_steps),
            'step_count_symmetry': abs(len(left_steps) - len(right_steps))
        }
    
    def _analyze_spatial_symmetry(self):
        """Analiza symetrii przestrzennej rozkładu nacisku"""
        left_means = np.mean(self.left, axis=0)
        right_means = np.mean(self.right, axis=0)
        
        # Mapowanie czujników dla symetrii (1->2, 3->4, 5->6, 7->8)
        symmetric_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]  # indeksy czujników
        
        spatial_asymmetries = {}
        for i, (left_idx, right_idx) in enumerate(symmetric_pairs):
            left_force = left_means[left_idx]
            right_force = right_means[right_idx]
            
            if left_force + right_force > 0:
                asymmetry = 2 * abs(left_force - right_force) / (left_force + right_force) * 100
            else:
                asymmetry = 0
            
            region_names = ['heel', 'midfoot', 'forefoot', 'toes']
            spatial_asymmetries[region_names[i]] = asymmetry
        
        return {
            'regional_asymmetries': spatial_asymmetries,
            'mean_spatial_asymmetry': np.mean(list(spatial_asymmetries.values()))
        }
    
    def _analyze_cop_symmetry(self):
        """Analiza symetrii centrum nacisku"""
        cop_data = self.calculate_cop_trajectory('both')
        
        if 'left' not in cop_data or 'right' not in cop_data:
            return {'error': 'Nie można obliczyć COP dla obu stóp'}
        
        left_cop = cop_data['left']
        right_cop = cop_data['right']
        
        # Porównaj parametry COP
        path_length_symmetry = 2 * abs(left_cop['path_length'] - right_cop['path_length']) / (left_cop['path_length'] + right_cop['path_length']) * 100
        sway_area_symmetry = 2 * abs(left_cop['sway_area'] - right_cop['sway_area']) / (left_cop['sway_area'] + right_cop['sway_area']) * 100
        
        return {
            'path_length_symmetry': path_length_symmetry,
            'sway_area_symmetry': sway_area_symmetry,
            'left_path_length': left_cop['path_length'],
            'right_path_length': right_cop['path_length'],
            'left_sway_area': left_cop['sway_area'],
            'right_sway_area': right_cop['sway_area']
        }
    
    def _calculate_overall_asymmetry(self, symmetry_results):
        """Oblicza ogólny wskaźnik asymetrii"""
        asymmetry_scores = []
        
        # Zbierz wszystkie wskaźniki asymetrii
        if 'total_force_symmetry_index' in symmetry_results.get('force_symmetry', {}):
            asymmetry_scores.append(symmetry_results['force_symmetry']['total_force_symmetry_index'])
        
        if 'temporal_symmetry_index' in symmetry_results.get('temporal_symmetry', {}):
            asymmetry_scores.append(symmetry_results['temporal_symmetry']['temporal_symmetry_index'])
        
        if 'mean_spatial_asymmetry' in symmetry_results.get('spatial_symmetry', {}):
            asymmetry_scores.append(symmetry_results['spatial_symmetry']['mean_spatial_asymmetry'])
        
        if asymmetry_scores:
            return np.mean(asymmetry_scores)
        else:
            return 0
    
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
    
    def generate_comprehensive_report(self):
        """Generuje kompleksowy raport z wszystkich analiz"""
        print(f"=" * 60)
        print(f"KOMPLEKSOWY RAPORT ANALIZY BIOMECHANICZNEJ STOPY")
        print(f"Arkusz: {self.sheet_name}")
        print(f"=" * 60)
        
        # Podstawowe informacje
        print(f"\n📊 PODSTAWOWE INFORMACJE:")
        print(f"Czas trwania pomiaru: {self.time[-1] - self.time[0]:.2f} s")
        print(f"Częstotliwość próbkowania: {self.sampling_rate:.1f} Hz")
        print(f"Liczba próbek: {len(self.time)}")
        
        # Analiza kroków
        left_steps, _ = self.detect_steps('left')
        right_steps, _ = self.detect_steps('right')
        print(f"\n👣 ANALIZA KROKÓW:")
        print(f"Kroki lewa stopa: {len(left_steps)}")
        print(f"Kroki prawa stopa: {len(right_steps)}")
        
        # Analiza korelacji
        print(f"\n🔗 ANALIZA KORELACJI:")
        correlation_results = self.analyze_foot_correlation()
        total_corr = correlation_results['total_force_correlation']['correlation']
        print(f"Korelacja sił całkowitych: {total_corr:.3f}")
        
        # Najwyższe korelacje czujników
        sensor_corrs = [(name, data['correlation']) for name, data in correlation_results['sensor_correlations'].items()]
        sensor_corrs.sort(key=lambda x: abs(x[1]), reverse=True)
        print(f"Najwyższe korelacje czujników:")
        for name, corr in sensor_corrs[:3]:
            print(f"  Czujnik {name}: {corr:.3f}")
        
        # Analiza symetrii
        print(f"\n⚖️ ANALIZA SYMETRII:")
        symmetry_results = self.analyze_symmetry()
        overall_asymmetry = symmetry_results['overall_asymmetry_index']
        print(f"Ogólny wskaźnik asymetrii: {overall_asymmetry:.1f}%")
        
        if 'force_symmetry' in symmetry_results:
            force_sym = symmetry_results['force_symmetry']['total_force_symmetry_index']
            print(f"Asymetria sił: {force_sym:.1f}%")
        
        if 'temporal_symmetry' in symmetry_results:
            temporal_sym = symmetry_results['temporal_symmetry']['temporal_symmetry_index']
            print(f"Asymetria czasowa: {temporal_sym:.1f}%")
        
        # Analiza COP
        print(f"\n🎯 ANALIZA CENTRUM NACISKU (COP):")
        cop_data = self.calculate_cop_trajectory('both')
        if 'left' in cop_data and 'right' in cop_data:
            left_path = cop_data['left']['path_length']
            right_path = cop_data['right']['path_length']
            left_area = cop_data['left']['sway_area']
            right_area = cop_data['right']['sway_area']
            
            print(f"Długość ścieżki COP - lewa: {left_path:.1f} mm")
            print(f"Długość ścieżki COP - prawa: {right_path:.1f} mm")
            print(f"Obszar wychwiań - lewa: {left_area:.1f} mm²")
            print(f"Obszar wychwiań - prawa: {right_area:.1f} mm²")
        
        # Interpretacja wyników
        print(f"\n📋 INTERPRETACJA:")
        self._interpret_results(overall_asymmetry, total_corr)
        
        return {
            'correlation': correlation_results,
            'symmetry': symmetry_results,
            'cop': cop_data
        }
    
    def _interpret_results(self, asymmetry_index, correlation):
        """Interpretuje wyniki analiz"""
        print("Ocena symetrii:")
        if asymmetry_index < 5:
            print("  ✅ Bardzo dobra symetria")
        elif asymmetry_index < 10:
            print("  ✅ Dobra symetria")
        elif asymmetry_index < 20:
            print("  ⚠️ Umiarkowana asymetria")
        else:
            print("  ❌ Znacząca asymetria - wymaga uwagi")
        
        print("Ocena koordynacji (korelacja):")
        if abs(correlation) > 0.8:
            print("  ✅ Bardzo dobra koordynacja między stopami")
        elif abs(correlation) > 0.6:
            print("  ✅ Dobra koordynacja")
        elif abs(correlation) > 0.4:
            print("  ⚠️ Umiarkowana koordynacja")
        else:
            print("  ❌ Słaba koordynacja między stopami")