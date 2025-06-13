import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks, butter, filtfilt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class MultiSensorGaitAnalyzer:
    def __init__(self, sampling_rate=100):
        """
        Analizator wzorców chodu z wieloma czujnikami
        
        Args:
            sampling_rate (int): Częstotliwość próbkowania w Hz
        """
        self.sampling_rate = sampling_rate
        self.data = None
        self.time = None
        self.left_sensors = None  # 8 czujników lewej stopy
        self.right_sensors = None  # 8 czujników prawej stopy
        self.left_total = None   # suma sygnałów lewej stopy
        self.right_total = None  # suma sygnałów prawej stopy
        self.steps_left = []
        self.steps_right = []
        self.sensor_positions = [
            'Heel_medial', 'Heel_lateral', 'Midfoot_medial', 'Midfoot_lateral',
            'Forefoot_medial', 'Forefoot_lateral', 'Toe_big', 'Toe_small'
        ]
        
    def load_data_from_text(self, filepath_or_data, separator=','):
        """
        Wczytanie danych z pliku tekstowego lub stringa
        Format: czas, 8_czujników_lewa_stopa, 8_czujników_prawa_stopa
        """
        if isinstance(filepath_or_data, str) and '\n' in filepath_or_data:
            # Dane jako string
            lines = filepath_or_data.strip().split('\n')
            data_rows = []
            for line in lines:
                if line.strip():
                    row = [float(x.replace(',', '.')) for x in line.split()]
                    data_rows.append(row)
            data_array = np.array(data_rows)
        else:
            # Dane z pliku
            data_array = np.loadtxt(filepath_or_data, delimiter=separator)
            
        # Sprawdzenie formatu danych
        if data_array.shape[1] != 17:  # 1 czas + 8 + 8 czujników
            raise ValueError(f"Oczekiwano 17 kolumn (czas + 8 + 8 czujników), otrzymano {data_array.shape[1]}")
            
        # Utworzenie DataFrame
        columns = ['time']
        for i in range(8):
            columns.append(f'left_{self.sensor_positions[i]}')
        for i in range(8):
            columns.append(f'right_{self.sensor_positions[i]}')
            
        self.data = pd.DataFrame(data_array, columns=columns)
        self.time = self.data['time'].values
        
        # Wydzielenie sygnałów z czujników
        self.left_sensors = self.data.iloc[:, 1:9].values  # kolumny 1-8
        self.right_sensors = self.data.iloc[:, 9:17].values  # kolumny 9-16
        
        # Obliczenie sumarycznych sygnałów dla każdej stopy
        self.left_total = np.sum(self.left_sensors, axis=1)
        self.right_total = np.sum(self.right_sensors, axis=1)
        
        print(f"Dane wczytane: {len(self.data)} próbek")
        print(f"Czas trwania: {self.time[-1] - self.time[0]:.2f}s")
        print(f"Częstotliwość próbkowania rzeczywista: {len(self.data)/(self.time[-1] - self.time[0]):.1f} Hz")
 
    def load_data_from_excel(self, filepath, sheet_name=0):
        """
        Wczytanie danych z pliku Excel (.xlsx) z możliwością wyboru arkusza.
        Zakłada format: czas, 8 czujników lewej stopy, 8 czujników prawej stopy

        Args:
        filepath (str): Ścieżka do pliku .xlsx
        sheet_name (int lub str): Indeks lub nazwa arkusza Excel
        """
        df = pd.read_excel(filepath, sheet_name=sheet_name, header=None)

        if df.shape[1] < 17:
            raise ValueError(f"Za mało kolumn: oczekiwano co najmniej 17, otrzymano {df.shape[1]}")

        # Ucinamy do pierwszych 17 kolumn
        df = df.iloc[:, :17]

        columns = ['time'] + \
                [f'left_{pos}' for pos in self.sensor_positions] + \
                [f'right_{pos}' for pos in self.sensor_positions]

        self.data = pd.DataFrame(df.values, columns=columns)
        self.time = self.data['time'].values
        self.left_sensors = self.data.iloc[:, 1:9].values
        self.right_sensors = self.data.iloc[:, 9:17].values
        self.left_total = np.sum(self.left_sensors, axis=1)
        self.right_total = np.sum(self.right_sensors, axis=1)

        print(f"Wczytano dane z arkusza: '{sheet_name}'")
        print(f"Liczba próbek: {len(self.data)}")
        print(f"Czas trwania: {self.time[-1] - self.time[0]:.2f}s")
        print(f"Częstotliwość próbkowania rzeczywista: {len(self.data)/(self.time[-1] - self.time[0]):.1f} Hz")
        
    def detect_steps(self, min_height_percentile=70, min_distance_sec=0.3):
        """Wykrywanie kroków na podstawie sumarycznych sygnałów"""
        min_height_left = np.percentile(self.left_total, min_height_percentile)
        min_height_right = np.percentile(self.right_total, min_height_percentile)
        min_distance = int(min_distance_sec * self.sampling_rate)
        
        # Wykrywanie pików
        peaks_left, _ = find_peaks(self.left_total, 
                                 height=min_height_left, 
                                 distance=min_distance)
        peaks_right, _ = find_peaks(self.right_total,
                                  height=min_height_right,
                                  distance=min_distance)
        
        self.steps_left = peaks_left
        self.steps_right = peaks_right
        
        print(f"Wykryto kroków: lewa stopa - {len(peaks_left)}, prawa stopa - {len(peaks_right)}")
        return peaks_left, peaks_right
    
    def analyze_pressure_distribution(self):
        """Analiza rozkładu nacisku na powierzchni stopy"""
        analysis = {}
        
        # Średni rozkład nacisku dla każdego czujnika
        left_means = np.mean(self.left_sensors, axis=0)
        right_means = np.mean(self.right_sensors, axis=0)
        
        analysis['left_pressure_distribution'] = dict(zip(self.sensor_positions, left_means))
        analysis['right_pressure_distribution'] = dict(zip(self.sensor_positions, right_means))
        
        # Centrum nacisku (Center of Pressure - CoP)
        # Założenie: czujniki rozmieszczone w siatce 2x4
        sensor_coords = np.array([
            [0, 3], [1, 3],    # Heel medial, lateral
            [0, 2], [1, 2],    # Midfoot medial, lateral  
            [0, 1], [1, 1],    # Forefoot medial, lateral
            [0, 0], [1, 0]     # Toe big, small
        ])
        
        cop_left_x = []
        cop_left_y = []
        cop_right_x = []
        cop_right_y = []
        
        for i in range(len(self.left_sensors)):
            # Centrum nacisku lewa stopa
            if np.sum(self.left_sensors[i]) > 0:
                cop_x = np.sum(sensor_coords[:, 0] * self.left_sensors[i]) / np.sum(self.left_sensors[i])
                cop_y = np.sum(sensor_coords[:, 1] * self.left_sensors[i]) / np.sum(self.left_sensors[i])
                cop_left_x.append(cop_x)
                cop_left_y.append(cop_y)
            else:
                cop_left_x.append(0.5)
                cop_left_y.append(1.5)
                
            # Centrum nacisku prawa stopa
            if np.sum(self.right_sensors[i]) > 0:
                cop_x = np.sum(sensor_coords[:, 0] * self.right_sensors[i]) / np.sum(self.right_sensors[i])
                cop_y = np.sum(sensor_coords[:, 1] * self.right_sensors[i]) / np.sum(self.right_sensors[i])
                cop_right_x.append(cop_x)
                cop_right_y.append(cop_y)
            else:
                cop_right_x.append(0.5)
                cop_right_y.append(1.5)
        
        analysis['cop_left'] = {'x': np.array(cop_left_x), 'y': np.array(cop_left_y)}
        analysis['cop_right'] = {'x': np.array(cop_right_x), 'y': np.array(cop_right_y)}
        
        # Trajektoria centrum nacisku
        analysis['cop_path_length_left'] = self._calculate_path_length(cop_left_x, cop_left_y)
        analysis['cop_path_length_right'] = self._calculate_path_length(cop_right_x, cop_right_y)
        
        return analysis
        
    def _calculate_path_length(self, x, y):
        """Obliczenie długości trajektorii"""
        return np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
    
    def analyze_gait_phases(self):
        """Analiza faz chodu na podstawie wzorców czujników"""
        phases = {}
        
        if len(self.steps_left) < 2 or len(self.steps_right) < 2:
            return phases
            
        # Analiza dla lewej stopy
        left_cycles = []
        for i in range(len(self.steps_left) - 1):
            start_idx = self.steps_left[i]
            end_idx = self.steps_left[i + 1]
            cycle_data = self.left_sensors[start_idx:end_idx, :]
            left_cycles.append(cycle_data)
            
        # Analiza dla prawej stopy
        right_cycles = []
        for i in range(len(self.steps_right) - 1):
            start_idx = self.steps_right[i]
            end_idx = self.steps_right[i + 1]
            cycle_data = self.right_sensors[start_idx:end_idx, :]
            right_cycles.append(cycle_data)
        
        # Identyfikacja faz na podstawie wzorców aktywacji czujników
        phases['left_heel_contact'] = self._detect_heel_contact(left_cycles, 'left')
        phases['right_heel_contact'] = self._detect_heel_contact(right_cycles, 'right')
        phases['left_toe_off'] = self._detect_toe_off(left_cycles, 'left')
        phases['right_toe_off'] = self._detect_toe_off(right_cycles, 'right')
        
        return phases
    
    def _detect_heel_contact(self, cycles, foot):
        """Wykrywanie momentu kontaktu pięty"""
        heel_contacts = []
        heel_sensors = [0, 1]  # Heel medial, lateral
        
        for cycle in cycles:
            if len(cycle) > 0:
                heel_signal = np.mean(cycle[:, heel_sensors], axis=1)
                # Moment pierwszego znaczącego wzrostu sygnału z pięty
                threshold = np.max(heel_signal) * 0.1
                contact_idx = np.where(heel_signal > threshold)[0]
                if len(contact_idx) > 0:
                    heel_contacts.append(contact_idx[0] / len(cycle))  # znormalizowane
                    
        return heel_contacts
    
    def _detect_toe_off(self, cycles, foot):
        """Wykrywanie momentu oderwania palców"""
        toe_offs = []
        toe_sensors = [6, 7]  # Toe big, small
        
        for cycle in cycles:
            if len(cycle) > 0:
                toe_signal = np.mean(cycle[:, toe_sensors], axis=1)
                # Moment gdy sygnał z palców spada poniżej progu
                threshold = np.max(toe_signal) * 0.1
                above_threshold = toe_signal > threshold
                if np.any(above_threshold):
                    last_above = np.where(above_threshold)[0][-1]
                    toe_offs.append(last_above / len(cycle))  # znormalizowane
                    
        return toe_offs
    
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
                
        symmetry['sensor_symmetry_indices'] = dict(zip(self.sensor_positions, sensor_symmetries))
        
        # Ogólna symetria sumaryczna
        total_sym = 2 * abs(np.mean(self.left_total) - np.mean(self.right_total)) / (np.mean(self.left_total) + np.mean(self.right_total)) * 100
        symmetry['total_symmetry_index'] = total_sym
        
        # Korelacja między odpowiadającymi sobie czujnikami
        sensor_correlations = []
        for i in range(8):
            corr = np.corrcoef(self.left_sensors[:, i], self.right_sensors[:, i])[0, 1]
            sensor_correlations.append(corr)
            
        symmetry['sensor_correlations'] = dict(zip(self.sensor_positions, sensor_correlations))
        
        # PCA dla analizy głównych wzorców asymetrii
        combined_sensors = np.column_stack([self.left_sensors, self.right_sensors])
        scaler = StandardScaler()
        scaled_sensors = scaler.fit_transform(combined_sensors)
        
        pca = PCA(n_components=4)
        pca_result = pca.fit_transform(scaled_sensors)
        
        symmetry['pca_explained_variance'] = pca.explained_variance_ratio_
        symmetry['pca_components'] = pca.components_
        
        return symmetry
    
    def detect_advanced_anomalies(self):
        """Zaawansowane wykrywanie anomalii na podstawie wzorców wielu czujników"""
        anomalies = []
        window_size = min(100, len(self.data) // 20)
        
        for i in range(0, len(self.data) - window_size, window_size//2):
            window_left = self.left_sensors[i:i+window_size, :]
            window_right = self.right_sensors[i:i+window_size, :]
            
            # Sprawdzenie wzorca aktywacji czujników
            left_pattern = np.mean(window_left, axis=0)
            right_pattern = np.mean(window_right, axis=0)
            
            # Nietypowy rozkład nacisku
            left_cv = np.std(left_pattern) / np.mean(left_pattern) if np.mean(left_pattern) > 0 else 0
            right_cv = np.std(right_pattern) / np.mean(right_pattern) if np.mean(right_pattern) > 0 else 0
            
            if left_cv > 1.5 or right_cv > 1.5:
                anomalies.append({
                    'time': self.time[i + window_size//2],
                    'type': 'unusual_pressure_pattern',
                    'value': max(left_cv, right_cv),
                    'description': 'Nietypowy rozkład nacisku między czujnikami'
                })
            
            # Brak aktywności w głównych czujnikach
            main_sensors = [0, 1, 4, 5, 6]  # heel, forefoot, toe
            left_main_activity = np.mean(left_pattern[main_sensors])
            right_main_activity = np.mean(right_pattern[main_sensors])
            
            if left_main_activity < 0.1 * np.mean(self.left_total) or right_main_activity < 0.1 * np.mean(self.right_total):
                anomalies.append({
                    'time': self.time[i + window_size//2],
                    'type': 'low_main_sensor_activity',
                    'value': min(left_main_activity, right_main_activity),
                    'description': 'Niska aktywność głównych czujników'
                })
            
            # Asymetria między odpowiadającymi sobie czujnikami
            sensor_asymmetries = []
            for j in range(8):
                if left_pattern[j] + right_pattern[j] > 0:
                    asym = abs(left_pattern[j] - right_pattern[j]) / (left_pattern[j] + right_pattern[j])
                    sensor_asymmetries.append(asym)
                    
            if sensor_asymmetries and np.mean(sensor_asymmetries) > 0.5:
                anomalies.append({
                    'time': self.time[i + window_size//2],
                    'type': 'high_sensor_asymmetry',
                    'value': np.mean(sensor_asymmetries),
                    'description': 'Wysoka asymetria między czujnikami'
                })
                
        return anomalies
    
    def plot_comprehensive_analysis(self, figsize=(20, 15)):
        """Kompleksowa wizualizacja dla danych wieloczujnikowych"""
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. Sygnały sumaryczne z wykrytymi krokami
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(self.time, self.left_total, 'b-', label='Lewa stopa (suma)', alpha=0.7, linewidth=2)
        ax1.plot(self.time, self.right_total, 'r-', label='Prawa stopa (suma)', alpha=0.7, linewidth=2)
        
        if len(self.steps_left) > 0:
            ax1.plot(self.time[self.steps_left], self.left_total[self.steps_left], 'bo', markersize=8)
        if len(self.steps_right) > 0:
            ax1.plot(self.time[self.steps_right], self.right_total[self.steps_right], 'ro', markersize=8)
            
        ax1.set_xlabel('Czas [s]')
        ax1.set_ylabel('Siła nacisku (suma)')
        ax1.set_title('Sygnały sumaryczne z wykrytymi krokami')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Mapa ciepła czujników - lewa stopa
        ax2 = fig.add_subplot(gs[0, 2])
        left_means = np.mean(self.left_sensors, axis=0).reshape(4, 2)
        im1 = ax2.imshow(left_means, cmap='Reds', aspect='auto')
        ax2.set_title('Rozkład nacisku - Lewa stopa')
        ax2.set_xticks([0, 1])
        ax2.set_xticklabels(['Medial', 'Lateral'])
        ax2.set_yticks([0, 1, 2, 3])
        ax2.set_yticklabels(['Toe', 'Forefoot', 'Midfoot', 'Heel'])
        plt.colorbar(im1, ax=ax2)
        
        # 3. Mapa ciepła czujników - prawa stopa
        ax3 = fig.add_subplot(gs[0, 3])
        right_means = np.mean(self.right_sensors, axis=0).reshape(4, 2)
        im2 = ax3.imshow(right_means, cmap='Blues', aspect='auto')
        ax3.set_title('Rozkład nacisku - Prawa stopa')
        ax3.set_xticks([0, 1])
        ax3.set_xticklabels(['Medial', 'Lateral'])
        ax3.set_yticks([0, 1, 2, 3])
        ax3.set_yticklabels(['Toe', 'Forefoot', 'Midfoot', 'Heel'])
        plt.colorbar(im2, ax=ax3)
        
        # 4. Trajektoria centrum nacisku
        pressure_analysis = self.analyze_pressure_distribution()
        ax4 = fig.add_subplot(gs[1, :2])
        ax4.plot(pressure_analysis['cop_left']['x'], pressure_analysis['cop_left']['y'], 
                'b-', alpha=0.7, label='Lewa stopa')
        ax4.plot(pressure_analysis['cop_right']['x'], pressure_analysis['cop_right']['y'], 
                'r-', alpha=0.7, label='Prawa stopa')
        ax4.set_xlabel('Pozycja X')
        ax4.set_ylabel('Pozycja Y (Heel→Toe)')
        ax4.set_title('Trajektoria Centrum Nacisku (CoP)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_aspect('equal')
        
        # 5. Symetria dla każdego czujnika
        symmetry = self.analyze_multi_sensor_symmetry()
        ax5 = fig.add_subplot(gs[1, 2:])
        sensor_names = list(symmetry['sensor_symmetry_indices'].keys())
        symmetry_values = list(symmetry['sensor_symmetry_indices'].values())
        
        bars = ax5.bar(range(len(sensor_names)), symmetry_values, 
                      color=['red' if x > 20 else 'orange' if x > 10 else 'green' for x in symmetry_values])
        ax5.set_xlabel('Czujniki')
        ax5.set_ylabel('Indeks asymetrii [%]')
        ax5.set_title('Asymetria dla każdego czujnika')
        ax5.set_xticks(range(len(sensor_names)))
        ax5.set_xticklabels(sensor_names, rotation=45, ha='right')
        ax5.axhline(y=10, color='orange', linestyle='--', alpha=0.7, label='Próg 10%')
        ax5.axhline(y=20, color='red', linestyle='--', alpha=0.7, label='Próg 20%')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Analiza PCA
        ax6 = fig.add_subplot(gs[2, :2])
        explained_var = symmetry['pca_explained_variance']
        ax6.bar(range(len(explained_var)), explained_var * 100)
        ax6.set_xlabel('Składowe główne')
        ax6.set_ylabel('Wyjaśniona wariancja [%]')
        ax6.set_title('Analiza głównych składowych (PCA)')
        ax6.grid(True, alpha=0.3)
        
        # 7. Korelacje między czujnikami
        ax7 = fig.add_subplot(gs[2, 2:])
        correlations = list(symmetry['sensor_correlations'].values())
        colors = ['red' if x < 0.5 else 'orange' if x < 0.7 else 'green' for x in correlations]
        bars = ax7.bar(range(len(sensor_names)), correlations, color=colors)
        ax7.set_xlabel('Czujniki')
        ax7.set_ylabel('Korelacja L-R')
        ax7.set_title('Korelacja między odpowiadającymi czujnikami')
        ax7.set_xticks(range(len(sensor_names)))
        ax7.set_xticklabels(sensor_names, rotation=45, ha='right')
        ax7.axhline(y=0.7, color='green', linestyle='--', alpha=0.7)
        ax7.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7)
        ax7.grid(True, alpha=0.3)
        
        # 8. Analiza czasowa wybranych czujników
        ax8 = fig.add_subplot(gs[3, :])
        # Pokaż sygnały z kluczowych czujników
        key_sensors = [0, 4, 6]  # Heel medial, Forefoot medial, Toe big
        colors = ['purple', 'green', 'orange']
        
        for i, (sensor_idx, color) in enumerate(zip(key_sensors, colors)):
            ax8.plot(self.time, self.left_sensors[:, sensor_idx], 
                    color=color, alpha=0.7, linestyle='-', 
                    label=f'L-{self.sensor_positions[sensor_idx]}')
            ax8.plot(self.time, self.right_sensors[:, sensor_idx], 
                    color=color, alpha=0.7, linestyle='--',
                    label=f'R-{self.sensor_positions[sensor_idx]}')
        
        ax8.set_xlabel('Czas [s]')
        ax8.set_ylabel('Siła nacisku')
        ax8.set_title('Sygnały z kluczowych czujników')
        ax8.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax8.grid(True, alpha=0.3)
        
        plt.suptitle('Kompleksowa Analiza Chodu - Dane Wieloczujnikowe', fontsize=16, y=0.98)
        plt.tight_layout()
        return fig
    
    def generate_comprehensive_report(self):
        """Szczegółowy raport z analizy wieloczujnikowej"""
        print("=== SZCZEGÓŁOWY RAPORT ANALIZY CHODU ===\n")
        
        # Podstawowe informacje
        duration = self.time[-1] - self.time[0]
        actual_sampling_rate = len(self.data) / duration
        print(f"Czas nagrania: {duration:.2f} sekund")
        print(f"Liczba próbek: {len(self.data)}")
        print(f"Częstotliwość próbkowania: {actual_sampling_rate:.1f} Hz\n")
        
        # Wykryte kroki
        print("--- WYKRYTE KROKI ---")
        print(f"Lewa stopa: {len(self.steps_left)} kroków")
        print(f"Prawa stopa: {len(self.steps_right)} kroków")
        
        if len(self.steps_left) > 1:
            left_cadence = 60 / np.mean(np.diff(self.time[self.steps_left]))
            print(f"Kadencja lewa stopa: {left_cadence:.1f} kroków/min")
            
        if len(self.steps_right) > 1:
            right_cadence = 60 / np.mean(np.diff(self.time[self.steps_right]))
            print(f"Kadencja prawa stopa: {right_cadence:.1f} kroków/min")
        print()
        
        # Analiza rozkładu nacisku
        print("--- ROZKŁAD NACISKU ---")
        pressure_dist = self.analyze_pressure_distribution()
        
        print("Średni nacisk na czujniki - Lewa stopa:")
        for sensor, value in pressure_dist['left_pressure_distribution'].items():
            print(f"  {sensor}: {value:.3f}")
            
        print("\nŚredni nacisk na czujniki - Prawa stopa:")
        for sensor, value in pressure_dist['right_pressure_distribution'].items():
            print(f"  {sensor}: {value:.3f}")
            
        print(f"\nDługość trajektorii CoP - Lewa: {pressure_dist['cop_path_length_left']:.3f}")
        print(f"Długość trajektorii CoP - Prawa: {pressure_dist['cop_path_length_right']:.3f}")
        print()
        
        # Analiza symetrii
        print("--- ANALIZA SYMETRII ---")
        symmetry = self.analyze_multi_sensor_symmetry()
        
        print(f"Ogólny indeks symetrii: {symmetry['total_symmetry_index']:.2f}%")
        print("\nIndeksy symetrii dla poszczególnych czujników:")
        for sensor, value in symmetry['sensor_symmetry_indices'].items():
            status = "🔴 WYSOKA" if value > 20 else "🟡 ŚREDNIA" if value > 10 else "🟢 NISKA"
            print(f"  {sensor}: {value:.2f}% ({status})")
            
        print("\nKorelacje między odpowiadającymi czujnikami:")
        for sensor, value in symmetry['sensor_correlations'].items():
            status = "🔴 NISKA" if value < 0.5 else "🟡 ŚREDNIA" if value < 0.7 else "🟢 WYSOKA"
            print(f"  {sensor}: {value:.3f} ({status})")

        print("\n--- FAZY CHODU ---")
        phases = self.analyze_gait_phases()
        for phase_name, timings in phases.items():
            avg_time = np.mean(timings) if timings else None
            if avg_time is not None:
                print(f"{phase_name}: średnio {avg_time*100:.1f}% cyklu")
            else:
                print(f"{phase_name}: brak danych")

        print("\n--- ANOMALIE ---")
        anomalies = self.detect_advanced_anomalies()
        if anomalies:
            for anomaly in anomalies:
                print(f"[{anomaly['time']:.2f}s] {anomaly['type']} - {anomaly['description']} (wartość: {anomaly['value']:.3f})")
        else:
            print("Brak istotnych anomalii wykrytych w danych")

        print("\n=== KONIEC RAPORTU ===")
   

# Krok 1: Utwórz obiekt analizatora
analyzer = MultiSensorGaitAnalyzer(sampling_rate=100)

# Krok 2: Wczytaj dane z pliku tekstowego (np. CSV lub TXT)
# analyzer.load_data_from_text("dane.txt", separator=",")
analyzer.load_data_from_excel("WybraneDaneGotowe1.xlsx", sheet_name="2")

# Krok 3: Wstępna filtracja sygnałów
#analyzer.preprocess_data(lowpass_freq=10, highpass_freq=0.5)

# Krok 4: Wykrywanie kroków
steps_left, steps_right = analyzer.detect_steps()

# Krok 5: Analiza rozkładu nacisku
pressure_results = analyzer.analyze_pressure_distribution()

# Krok 6: Analiza faz chodu
phases = analyzer.analyze_gait_phases()

# Krok 7: Analiza symetrii
symmetry = analyzer.analyze_multi_sensor_symmetry()

# Krok 8: Wykrywanie anomalii
anomalies = analyzer.detect_advanced_anomalies()

# Krok 9: Wizualizacja danych
fig = analyzer.plot_comprehensive_analysis()
fig.savefig("analiza_chodu.png")

# Krok 10: Generowanie raportu
analyzer.generate_comprehensive_report()
