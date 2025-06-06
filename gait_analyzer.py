import numpy as np
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

    def plot_step_periods_both_feet(self, prominence=20, distance=30):
        """Główna funkcja rysująca rytm chodu obu stóp."""
        
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
