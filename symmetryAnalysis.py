import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import savgol_filter, find_peaks
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances
from sklearn.cluster import KMeans
from symmetryPlotter import SymmetryVisualizationTools
import json
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SymmetryPatternExtractor:
    """
    Klasa do ekstrakcji wzorców symetrii z danych czujników nacisku stóp
    i ich porównywania między różnymi osobami/sesjami
    """
    
    def __init__(self, data_file=None,  sheet_index=0):
        """
        Inicjalizacja ekstraktora wzorców
        
        Args:
            data_file: ścieżka do pliku CSV
            data_array: numpy array z danymi (alternatywa do pliku)
        """
        xls = pd.read_excel(data_file, sheet_name=None, engine="openpyxl")
        self.sheet_names = list(xls.keys())
        sheet_name = self.sheet_names[sheet_index]
        self.data = xls[sheet_name]
        self.time = self.data.iloc[:, 0]
        self.left_foot = self.data.iloc[:, 1:9]
        self.right_foot = self.data.iloc[:, 9:17]
        
        # Nazwy czujników
        self.left_sensors = [f'L{i}' for i in range(1, 9)]
        self.right_sensors = [f'R{i}' for i in range(1, 9)]
        
        self.left_foot.columns = self.left_sensors
        self.right_foot.columns = self.right_sensors
        
        # Inicjalizacja wzorców
        self.symmetry_patterns = {}
        
    def extract_core_symmetry_patterns(self):
        """
        Ekstraktuje podstawowe wzorce symetrii nadające się do porównywania
        
        Returns:
            dict: Słownik z kluczowymi wzorcami symetrii
        """
        patterns = {}
        
        # 1. WZORZEC GLOBALNY - całkowita symetria
        total_left = self.left_foot.sum(axis=1)
        total_right = self.right_foot.sum(axis=1)
        
        # Unikanie dzielenia przez zero
        global_symmetry = np.where(total_right != 0, total_left / total_right, 1.0)
        patterns['global_symmetry'] = {
            'mean': np.mean(global_symmetry),
            'std': np.std(global_symmetry),
            'median': np.median(global_symmetry),
            'iqr': np.percentile(global_symmetry, 75) - np.percentile(global_symmetry, 25),
            'skewness': stats.skew(global_symmetry),
            'kurtosis': stats.kurtosis(global_symmetry)
        }
        
        # 2. WZORZEC SENSOROWY - symetria dla każdego czujnika
        sensor_patterns = {}
        for i in range(8):
            left_vals = self.left_foot.iloc[:, i]
            right_vals = self.right_foot.iloc[:, i]
            
            # Stosunek symetrii
            sensor_ratio = np.where(right_vals != 0, left_vals / right_vals, 1.0)
            
            # Różnica znormalizowana
            total_sensor = left_vals + right_vals
            normalized_diff = np.where(total_sensor != 0, 
                                     (left_vals - right_vals) / total_sensor, 0.0)
            
            sensor_patterns[f'sensor_{i+1}'] = {
                'symmetry_ratio_mean': np.mean(sensor_ratio),
                'symmetry_ratio_std': np.std(sensor_ratio),
                'normalized_diff_mean': np.mean(normalized_diff),
                'normalized_diff_std': np.std(normalized_diff),
                'correlation': stats.pearsonr(left_vals, right_vals)[0] if len(left_vals) > 1 else 0,
                'cross_correlation_max': np.max(np.correlate(left_vals, right_vals, mode='full'))
            }
        
        patterns['sensor_patterns'] = sensor_patterns
        
        # 3. WZORZEC CENTRUM NACISKU (COP)
        cop_patterns = self._extract_cop_patterns()
        patterns['cop_patterns'] = cop_patterns
        
        # 4. WZORZEC DYNAMICZNY - zmienność w czasie
        dynamic_patterns = self._extract_dynamic_patterns()
        patterns['dynamic_patterns'] = dynamic_patterns
        
        # 5. WZORZEC FAZOWY - analiza cykli chodu
        phase_patterns = self._extract_phase_patterns()
        patterns['phase_patterns'] = phase_patterns
        
        # 6. WZORZEC SPEKTRALNY - analiza częstotliwościowa
        spectral_patterns = self._extract_spectral_patterns()
        patterns['spectral_patterns'] = spectral_patterns
        
        # 7. WZORZEC TOPOLOGICZNY - rozkład nacisku
        topology_patterns = self._extract_topology_patterns()
        patterns['topology_patterns'] = topology_patterns
        
        self.symmetry_patterns = patterns
        return patterns
    
    def _extract_cop_patterns(self):
        """Ekstraktuje wzorce centrum nacisku"""
        # Pozycje czujników (dostosuj do rzeczywistego układu)
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
        
        left_cop = calculate_cop(self.left_foot)
        right_cop = calculate_cop(self.right_foot)
        
        # Asymetria COP
        cop_asymmetry = left_cop - right_cop
        
        return {
            'left_cop_mean': np.mean(left_cop, axis=0),
            'right_cop_mean': np.mean(right_cop, axis=0),
            'cop_asymmetry_mean': np.mean(cop_asymmetry, axis=0),
            'cop_asymmetry_std': np.std(cop_asymmetry, axis=0),
            'cop_asymmetry_magnitude_mean': np.mean(np.linalg.norm(cop_asymmetry, axis=1)),
            'cop_asymmetry_magnitude_std': np.std(np.linalg.norm(cop_asymmetry, axis=1)),
            'cop_path_length_left': np.sum(np.linalg.norm(np.diff(left_cop, axis=0), axis=1)),
            'cop_path_length_right': np.sum(np.linalg.norm(np.diff(right_cop, axis=0), axis=1))
        }
    
    def _extract_dynamic_patterns(self):
        """Ekstraktuje wzorce dynamiczne"""
        # Zmienność w czasie
        left_variability = self.left_foot.std(axis=0)
        right_variability = self.right_foot.std(axis=0)
        
        # Trendy
        trends = {}
        for i in range(8):
            left_trend = np.polyfit(range(len(self.left_foot)), self.left_foot.iloc[:, i], 1)[0]
            right_trend = np.polyfit(range(len(self.right_foot)), self.right_foot.iloc[:, i], 1)[0]
            trends[f'sensor_{i+1}'] = {
                'left_trend': left_trend,
                'right_trend': right_trend,
                'trend_asymmetry': abs(left_trend - right_trend)
            }
        
        return {
            'variability_asymmetry': np.mean(np.abs(left_variability - right_variability)),
            'variability_correlation': stats.pearsonr(left_variability, right_variability)[0],
            'trends': trends,
            'global_trend_asymmetry': np.mean([trends[f'sensor_{i+1}']['trend_asymmetry'] for i in range(8)])
        }
    
    def _extract_phase_patterns(self):
        """Ekstraktuje wzorce fazowe (cykle chodu)"""
        # Suma nacisku jako wskaźnik cykli
        total_pressure = self.left_foot.sum(axis=1) + self.right_foot.sum(axis=1)
        
        # Wygładzenie sygnału
        if len(total_pressure) > 10:
            smoothed = savgol_filter(total_pressure, 
                                   window_length=min(11, len(total_pressure)//3*2+1), 
                                   polyorder=3)
        else:
            smoothed = total_pressure
        
        # Znalezienie pików (kroków)
        peaks, properties = find_peaks(smoothed, height=np.mean(smoothed), distance=len(smoothed)//10)
        
        # Analiza cykli
        if len(peaks) > 1:
            cycle_lengths = np.diff(peaks)
            cycle_regularity = np.std(cycle_lengths) / np.mean(cycle_lengths) if np.mean(cycle_lengths) > 0 else 0
        else:
            cycle_lengths = []
            cycle_regularity = 0
        
        return {
            'number_of_cycles': len(peaks),
            'cycle_regularity': cycle_regularity,
            'average_cycle_length': np.mean(cycle_lengths) if len(cycle_lengths) > 0 else 0,
            'cycle_length_std': np.std(cycle_lengths) if len(cycle_lengths) > 0 else 0
        }
    
    def _extract_spectral_patterns(self):
        """Ekstraktuje wzorce spektralne"""
        spectral_features = {}
        
        for i in range(8):
            left_signal = self.left_foot.iloc[:, i]
            right_signal = self.right_foot.iloc[:, i]
            
            # FFT
            left_fft = np.fft.fft(left_signal)
            right_fft = np.fft.fft(right_signal)
            
            # Spektrum mocy
            left_power = np.abs(left_fft)**2
            right_power = np.abs(right_fft)**2
            
            # Dominująca częstotliwość
            freqs = np.fft.fftfreq(len(left_signal))
            left_dominant_freq = freqs[np.argmax(left_power[1:len(left_power)//2])+1]
            right_dominant_freq = freqs[np.argmax(right_power[1:len(right_power)//2])+1]
            
            spectral_features[f'sensor_{i+1}'] = {
                'dominant_freq_asymmetry': abs(left_dominant_freq - right_dominant_freq),
                'power_asymmetry': abs(np.mean(left_power) - np.mean(right_power)),
                'spectral_correlation': stats.pearsonr(left_power, right_power)[0] if len(left_power) > 1 else 0
            }
        
        return spectral_features
    
    def _extract_topology_patterns(self):
        """Ekstraktuje wzorce topologiczne rozkładu nacisku"""
        # Średnie wartości dla każdego czujnika
        left_mean = self.left_foot.mean()
        right_mean = self.right_foot.mean()
        
        # Normalizacja do sumy = 1 (rozkład prawdopodobieństwa)
        left_normalized = left_mean / left_mean.sum() if left_mean.sum() > 0 else left_mean
        right_normalized = right_mean / right_mean.sum() if right_mean.sum() > 0 else right_mean
        
        # Entropia rozkładu
        def entropy(p):
            p = p[p > 0]  # Usuń zera
            return -np.sum(p * np.log2(p))
        
        left_entropy = entropy(left_normalized)
        right_entropy = entropy(right_normalized)
        
        # Dystans KL (Kullback-Leibler)
        def kl_divergence(p, q):
            p = p + 1e-10  # Unikaj dzielenia przez zero
            q = q + 1e-10
            return np.sum(p * np.log2(p / q))
        
        kl_div = kl_divergence(left_normalized, right_normalized)
        
        return {
            'left_entropy': left_entropy,
            'right_entropy': right_entropy,
            'entropy_asymmetry': abs(left_entropy - right_entropy),
            'kl_divergence': kl_div,
            'topology_correlation': stats.pearsonr(left_normalized, right_normalized)[0],
            'max_pressure_asymmetry': abs(np.argmax(left_normalized) - np.argmax(right_normalized))
        }
    
    def create_symmetry_fingerprint(self):
        """
        Tworzy 'odcisk palca' symetrii - zwięzły wektor cech charakterystycznych
        
        Returns:
            dict: Słownik z kluczowymi cechami symetrii
        """
        if not self.symmetry_patterns:
            self.extract_core_symmetry_patterns()
        
        fingerprint = {}
        
        # 1. Globalne cechy symetrii
        fingerprint['global_symmetry_mean'] = self.symmetry_patterns['global_symmetry']['mean']
        fingerprint['global_symmetry_std'] = self.symmetry_patterns['global_symmetry']['std']
        fingerprint['global_symmetry_skewness'] = self.symmetry_patterns['global_symmetry']['skewness']
        
        # 2. Średnie cechy sensorowe
        sensor_ratios = [self.symmetry_patterns['sensor_patterns'][f'sensor_{i+1}']['symmetry_ratio_mean'] 
                        for i in range(8)]
        sensor_correlations = [self.symmetry_patterns['sensor_patterns'][f'sensor_{i+1}']['correlation'] 
                              for i in range(8)]
        
        fingerprint['sensor_asymmetry_mean'] = np.mean(np.abs(np.array(sensor_ratios) - 1.0))
        fingerprint['sensor_asymmetry_std'] = np.std(np.abs(np.array(sensor_ratios) - 1.0))
        fingerprint['sensor_correlation_mean'] = np.mean(sensor_correlations)
        fingerprint['sensor_correlation_std'] = np.std(sensor_correlations)
        
        # 3. Cechy centrum nacisku
        fingerprint['cop_asymmetry_magnitude'] = self.symmetry_patterns['cop_patterns']['cop_asymmetry_magnitude_mean']
        fingerprint['cop_path_asymmetry'] = abs(
            self.symmetry_patterns['cop_patterns']['cop_path_length_left'] - 
            self.symmetry_patterns['cop_patterns']['cop_path_length_right']
        )
        
        # 4. Cechy dynamiczne
        fingerprint['variability_asymmetry'] = self.symmetry_patterns['dynamic_patterns']['variability_asymmetry']
        fingerprint['trend_asymmetry'] = self.symmetry_patterns['dynamic_patterns']['global_trend_asymmetry']
        
        # 5. Cechy fazowe
        fingerprint['cycle_regularity'] = self.symmetry_patterns['phase_patterns']['cycle_regularity']
        fingerprint['number_of_cycles'] = self.symmetry_patterns['phase_patterns']['number_of_cycles']
        
        # 6. Cechy topologiczne
        fingerprint['entropy_asymmetry'] = self.symmetry_patterns['topology_patterns']['entropy_asymmetry']
        fingerprint['kl_divergence'] = self.symmetry_patterns['topology_patterns']['kl_divergence']
        
        return fingerprint
    
    def save_patterns(self, filename):
        """Zapisuje wzorce do pliku"""
        if not self.symmetry_patterns:
            self.extract_core_symmetry_patterns()
        
        save_data = {
            'timestamp': datetime.now().isoformat(),
            'patterns': self.symmetry_patterns,
            'fingerprint': self.create_symmetry_fingerprint()
        }
        
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        print(f"Wzorce zapisane do: {filename}")
    
    def load_patterns(self, filename):
        """Wczytuje wzorce z pliku"""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self.symmetry_patterns = data['patterns']
        return data
    
    def compare_with_patterns(self, other_patterns_file):
        """
        Porównuje obecne wzorce z innymi wzorcami
        
        Args:
            other_patterns_file: ścieżka do pliku z innymi wzorcami
            
        Returns:
            dict: Wyniki porównania
        """
        if not self.symmetry_patterns:
            self.extract_core_symmetry_patterns()
        
        # Wczytaj wzorce do porównania
        other_data = self.load_patterns(other_patterns_file)
        other_patterns = other_data['patterns']
        
        # Porównaj odciski palców
        current_fingerprint = self.create_symmetry_fingerprint()
        other_fingerprint = other_data['fingerprint']
        
        comparison = {}
        
        # Oblicz różnice dla każdej cechy
        for key in current_fingerprint:
            if key in other_fingerprint:
                diff = abs(current_fingerprint[key] - other_fingerprint[key])
                comparison[key] = {
                    'current': current_fingerprint[key],
                    'other': other_fingerprint[key],
                    'difference': diff,
                    'relative_difference': diff / (abs(other_fingerprint[key]) + 1e-10)
                }
        
        # Oblicz ogólne podobieństwo
        differences = [comp['relative_difference'] for comp in comparison.values()]
        overall_similarity = 1 / (1 + np.mean(differences))  # Im mniejsze różnice, tym większe podobieństwo
        
        comparison['overall_similarity'] = overall_similarity
        comparison['similarity_score'] = overall_similarity * 100  # Procent podobieństwa
        
        return comparison

class SymmetryDatabase:
    """
    Klasa do zarządzania bazą wzorców symetrii
    """
    
    def __init__(self, db_filename='symmetry_database.pkl'):
        self.db_filename = db_filename
        self.patterns_db = self._load_database()
    
    def _load_database(self):
        """Wczytuje bazę danych"""
        try:
            with open(self.db_filename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return {}
    
    def _save_database(self):
        """Zapisuje bazę danych"""
        with open(self.db_filename, 'wb') as f:
            pickle.dump(self.patterns_db, f)
    
    def add_pattern(self, pattern_id, extractor, metadata=None):
        """
        Dodaje wzorzec do bazy
        
        Args:
            pattern_id: unikalny identyfikator wzorca
            extractor: obiekt SymmetryPatternExtractor
            metadata: dodatkowe informacje (wiek, płeć, stan zdrowia, etc.)
        """
        if not extractor.symmetry_patterns:
            extractor.extract_core_symmetry_patterns()
        
        self.patterns_db[pattern_id] = {
            'timestamp': datetime.now().isoformat(),
            'patterns': extractor.symmetry_patterns,
            'fingerprint': extractor.create_symmetry_fingerprint(),
            'metadata': metadata or {}
        }
        
        self._save_database()
        print(f"Wzorzec {pattern_id} dodany do bazy")
    
    def find_similar_patterns(self, extractor, top_k=5):
        """
        Znajduje najbardziej podobne wzorce w bazie
        
        Args:
            extractor: obiekt SymmetryPatternExtractor do porównania
            top_k: liczba najbardziej podobnych wzorców do zwrócenia
            
        Returns:
            list: Lista podobnych wzorców z wynikami podobieństwa
        """
        if not extractor.symmetry_patterns:
            extractor.extract_core_symmetry_patterns()
        
        current_fingerprint = extractor.create_symmetry_fingerprint()
        similarities = []
        
        for pattern_id, pattern_data in self.patterns_db.items():
            other_fingerprint = pattern_data['fingerprint']
            
            # Oblicz podobieństwo
            differences = []
            for key in current_fingerprint:
                if key in other_fingerprint:
                    diff = abs(current_fingerprint[key] - other_fingerprint[key])
                    rel_diff = diff / (abs(other_fingerprint[key]) + 1e-10)
                    differences.append(rel_diff)
            
            if differences:
                similarity = 1 / (1 + np.mean(differences))
                similarities.append({
                    'pattern_id': pattern_id,
                    'similarity': similarity,
                    'similarity_score': similarity * 100,
                    'metadata': pattern_data['metadata']
                })
        
        # Sortuj według podobieństwa
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similarities[:top_k]
    
    def get_pattern_statistics(self):
        """Zwraca statystyki bazy wzorców"""
        if not self.patterns_db:
            return "Baza jest pusta"
        
        stats = {
            'total_patterns': len(self.patterns_db),
            'metadata_summary': {}
        }
        
        # Analiza metadanych
        for pattern_data in self.patterns_db.values():
            metadata = pattern_data['metadata']
            for key, value in metadata.items():
                if key not in stats['metadata_summary']:
                    stats['metadata_summary'][key] = {}
                
                if value not in stats['metadata_summary'][key]:
                    stats['metadata_summary'][key][value] = 0
                stats['metadata_summary'][key][value] += 1
        
        return stats

# Funkcje pomocnicze do wizualizacji porównań
def plot_symmetry_comparison(extractor1, extractor2, labels=['Wzorzec 1', 'Wzorzec 2']):
    """
    Porównuje wizualnie dwa wzorce symetrii
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Porównanie Wzorców Symetrii', fontsize=16)
    
    # Przygotuj dane
    patterns1 = extractor1.extract_core_symmetry_patterns()
    patterns2 = extractor2.extract_core_symmetry_patterns()
    
    fingerprint1 = extractor1.create_symmetry_fingerprint()
    fingerprint2 = extractor2.create_symmetry_fingerprint()
    
    # 1. Porównanie globalnej symetrii
    global_metrics = ['mean', 'std', 'skewness']
    values1 = [patterns1['global_symmetry'][metric] for metric in global_metrics]
    values2 = [patterns2['global_symmetry'][metric] for metric in global_metrics]
    
    x = np.arange(len(global_metrics))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, values1, width, label=labels[0], alpha=0.7)
    axes[0, 0].bar(x + width/2, values2, width, label=labels[1], alpha=0.7)
    axes[0, 0].set_xlabel('Metryki')
    axes[0, 0].set_ylabel('Wartość')
    axes[0, 0].set_title('Globalna Symetria')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(global_metrics)
    axes[0, 0].legend()
    
    # 2. Porównanie korelacji sensorów
    correlations1 = [patterns1['sensor_patterns'][f'sensor_{i+1}']['correlation'] for i in range(8)]
    correlations2 = [patterns2['sensor_patterns'][f'sensor_{i+1}']['correlation'] for i in range(8)]
    
    x_sensors = np.arange(8)
    axes[0, 1].bar(x_sensors - width/2, correlations1, width, label=labels[0], alpha=0.7)
    axes[0, 1].bar(x_sensors + width/2, correlations2, width, label=labels[1], alpha=0.7)
    axes[0, 1].set_xlabel('Czujnik')
    axes[0, 1].set_ylabel('Korelacja')
    axes[0, 1].set_title('Korelacje Czujników')
    axes[0, 1].set_xticks(x_sensors)
    axes[0, 1].set_xticklabels([f'S{i+1}' for i in range(8)])
    axes[0, 1].legend()
    
    # 3. Porównanie odcisków palców
    common_keys = set(fingerprint1.keys()) & set(fingerprint2.keys())
    common_keys = sorted(list(common_keys))
    
    if len(common_keys) > 10:
        common_keys = common_keys[:10]  # Ograniczenie do 10 najważniejszych
    
    fp_values1 = [fingerprint1[key] for key in common_keys]
    fp_values2 = [fingerprint2[key] for key in common_keys]
    
    # Normalizacja dla lepszej wizualizacji
    scaler = StandardScaler()
    fp_values1_norm = scaler.fit_transform(np.array(fp_values1).reshape(-1, 1)).flatten()
    fp_values2_norm = scaler.fit_transform(np.array(fp_values2).reshape(-1, 1)).flatten()
    
    x_fp = np.arange(len(common_keys))
    axes[0, 2].bar(x_fp - width/2, fp_values1_norm, width, label=labels[0], alpha=0.7)
    axes[0, 2].bar(x_fp + width/2, fp_values2_norm, width, label=labels[1], alpha=0.7)
    axes[0, 2].set_xlabel('Cechy')
    axes[0, 2].set_ylabel('Wartość (znormalizowana)')
    axes[0, 2].set_title('Odcisk Palca Symetrii')
    axes[0, 2].set_xticks(x_fp)
    axes[0, 2].set_xticklabels([key.replace('_', '\n') for key in common_keys], rotation=45, ha='right')
    axes[0, 2].legend()
    
    # 4-6. Dodatkowe porównania
    # Można dodać więcej szczegółowych porównań
    
    plt.tight_layout()
    plt.show()

# Przykład użycia
if __name__ == "__main__":
    print("=== SYSTEM EKSTRAKCJI WZORCÓW SYMETRII ===\n")
    
    # 1. Utworzenie ekstraktora
    extractor = SymmetryPatternExtractor("Dane2025-06-06.xlsx")
    
    # 2. Ekstrakcja wzorców
    patterns = extractor.extract_core_symmetry_patterns()
    
    # 3. Utworzenie odcisku palca
    fingerprint = extractor.create_symmetry_fingerprint()
    
    print("=== ODCISK PALCA SYMETRII ===")
    for key, value in fingerprint.items():
        print(f"{key}: {value:.6f}")
    
    # 4. Zapisanie wzorców
    extractor.save_patterns('wzorzec_osoby_1.json')
    
    # 5. Utworzenie bazy danych
    db = SymmetryDatabase()
    
    # 6. Dodanie wzorca do bazy
    db.add_pattern('osoba_1_zdrowa', extractor, {
        'wiek': 30,
        'płeć': 'K',
        'stan_zdrowia': 'zdrowa',
        'data_pomiaru': '2024-01-01'
    })
    
    print(f"\n=== STATYSTYKI BAZY DANYCH ===")
    stats = db.get_pattern_statistics()
    print(f"Liczba wzorców: {stats['total_patterns']}")
    
    # 7. Przykład porównania (jeśli masz drugi plik)
    # extractor2 = SymmetryPatternExtractor('test-dane2.txt')
    # similarities = db.find_similar_patterns(extractor2)
    # print("Najbardziej podobne wzorce:", similarities)
    # 1. Inicjalizacja z obiektem SymmetryPatternExtractor
    vis_tools = SymmetryVisualizationTools(extractor)

    # 2. Generowanie podstawowego przeglądu
    vis_tools.plot_basic_pressure_overview()

    # 3. Szczegółowa analiza wzorców
    vis_tools.plot_symmetry_patterns_detailed()

    # 4. Specjalistyczne wizualizacje
    vis_tools.plot_fingerprint_radar()
    vis_tools.plot_sensor_heatmap()
    vis_tools.plot_time_series_analysis() 
   