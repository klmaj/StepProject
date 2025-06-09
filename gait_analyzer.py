import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import plotly.graph_objs as go

class GaitAnalyzer:
    def __init__(self, time, left_df, right_df):
        self.time = time
        self.left = left_df
        self.right = right_df

    def get_combined_signal(self, foot: str):
        """Zwraca sygnał sumaryczny dla lewej lub prawej stopy."""
        if foot == "left":
            return self.left.sum(axis=1)
        elif foot == "right":
            return self.right.sum(axis=1)
        else:
            raise ValueError("foot must be 'left' or 'right'")

    def detect_steps(self, signal, prominence=20, distance=30):
        """Detekcja kroków na podstawie sygnału sumarycznego."""
        peaks, _ = find_peaks(signal, prominence=prominence, distance=distance)
        return self.time.iloc[peaks].values

    def compute_step_periods(self, step_times):
        """Zwraca okresy kroków oraz odpowiadające im czasy (start kolejnych kroków)."""
        periods = np.diff(step_times)
        times = step_times[:-1]
        return times, periods

    # Przycięcie sygnału: usuń początkowe i końcowe zera
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
            signal = self.left.iloc[:,0].values 
        elif side == 'right':
            signal = self.right.iloc[:,0].values
        else:
            raise ValueError("side must be 'left' or 'right'")

        times = self.time.values

        step_starts, _, trimmed_time = self.find_step_edges(signal, times)

        step_times = trimmed_time[step_starts]
        periods = np.diff(step_times)

        return step_times[:-1], periods  # Ostatni start nie ma kolejnego kroku

    def detect_step_phases(self, threshold: float = 10.0) -> pd.DataFrame:
        """
        Wykrywa fazy chodu: stance (podpora), swing (przenoszenie), double support (dwupodporowa).
        Zwraca DataFrame z kolumnami: time, left, right, global (faza łączna).
        """
        left_signal = self.left.sum(axis=1).values  
        right_signal = self.right.sum(axis=1).values
        times = self.time.values

        left_stance = left_signal > threshold
        right_stance = right_signal > threshold

        phases = []

        for i in range(len(times)):
            phase_left = 'stance' if left_stance[i] else 'swing'
            phase_right = 'stance' if right_stance[i] else 'swing'

            if left_stance[i] and right_stance[i]:
                global_phase = 'double_support'
            elif left_stance[i] or right_stance[i]:
                global_phase = 'single_support'
            else:
                global_phase = 'no_contact'

            phases.append({
                'time': times[i],
                'left': phase_left,
                'right': phase_right,
                'global': global_phase
            })

        return pd.DataFrame(phases)
        
    def symmetry_index(self, a, b):
        return 100 * (a - b) / ((a + b) / 2 + 1e-6)

    def get_summary(self):
        steps_L = self.count_steps(self.left_contact)
        steps_P = self.count_steps(self.right_contact)
        dur_L = self.contact_durations(self.left_contact)
        dur_P = self.contact_durations(self.right_contact)
        mean_dur_L = np.mean(dur_L)
        mean_dur_P = np.mean(dur_P)
        #mean_press_L = self.left.mean()
        #mean_press_P = self.right.mean()

        return {
            "steps": (steps_L, steps_P, self.symmetry_index(steps_L, steps_P)),
            "duration": (mean_dur_L, mean_dur_P, self.symmetry_index(mean_dur_L, mean_dur_P)),
            #pressure": (mean_press_L, mean_press_P, self.symmetry_index(mean_press_L, mean_press_P)),
        }
