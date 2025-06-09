import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks
import numpy as np
from gait_analyzer import GaitAnalyzer


class GaitPlotter:
    def __init__(self, time, analyzer:GaitAnalyzer):
        self.analyzer = analyzer
        self.time = time
        self.left = analyzer.left
        self.right = analyzer.right
        

    def plot_signals(self, czas_max=None):
        # Filtrowanie danych wg zakresu czasu
        if czas_max is not None:
            mask = self.time <= czas_max
            time_plot = self.time[mask]
            left_plot = self.left[mask]
            right_plot = self.right[mask]
            title = f"Dane do {czas_max} s"
        else:
            time_plot = self.time
            left_plot = self.left
            right_plot = self.right
            title = "Pełne dane"

        # Kolory
        colors = ['red', 'yellow', 'blue', 'darkgreen', 'brown', 'cyan', 'magenta', 'lightgreen']

        # Wykres
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=["Lewa stopa", "Prawa stopa"]
        )

        # Lewa stopa
        for i in range(left_plot.shape[1]):
            fig.add_trace(
                go.Scatter(x=time_plot, y=left_plot.iloc[:, i], mode='lines',
                           name=f'L{i+1}', line=dict(color=colors[i])),
                row=1, col=1
            )

        # Prawa stopa
        for i in range(right_plot.shape[1]):
            fig.add_trace(
                go.Scatter(x=time_plot, y=right_plot.iloc[:, i], mode='lines',
                           name=f'P{i+1}', line=dict(color=colors[i])),
                row=2, col=1
            )

        # Suwak czasu na dolnej osi
        fig.update_xaxes(
            rangeslider=dict(visible=True),
            row=2, col=1
        )

        # Interaktywność i styl
        fig.update_layout(
            height=900,
            title=title,
            hovermode="x unified",
            template="plotly_white",
            dragmode='zoom'
        )

        fig.update_yaxes(title_text="Lewa stopa F [N]", fixedrange=False, row=1, col=1)
        fig.update_yaxes(title_text="Prawa stopa F [N]", fixedrange=False, row=2, col=1)
        fig.update_xaxes(title_text="Czas [s]", row=2, col=1)

        fig.show()

    def plot_step_periods(self):
        """
        Rysuje wykres okresu kroku (czas między kolejnymi zerami) w funkcji czasu,
        osobno dla lewej i prawej stopy, na jednym wykresie.
        """
        # Dane z analizatora
        times_left, periods_left = self.analyzer.get_step_periods("left")
        times_right, periods_right = self.analyzer.get_step_periods("right")

        # Tworzenie wykresu interaktywnego
        fig = go.Figure()

        # Lewa stopa – okresy kroków
        fig.add_trace(go.Scatter(
            x=times_left,
            y=periods_left,
            mode='markers',
            name='Lewa stopa',
            marker=dict(color='blue')
        ))

        # Prawa stopa – okresy kroków
        fig.add_trace(go.Scatter(
            x=times_right,
            y=periods_right,
            mode='markers',
            name='Prawa stopa',
            marker=dict(color='red')
        ))

        # Średnia lewa
        mean_left = np.mean(periods_left)
        fig.add_trace(go.Scatter(
            x=[times_left.min(), times_left.max()],
            y=[mean_left, mean_left],
            mode='lines',
            name=f'Średnia lewa ({mean_left:.2f}s)',
            line=dict(color='blue', dash='solid', width=1)
        ))

        # Średnia prawa
        mean_right = np.mean(periods_right)
        fig.add_trace(go.Scatter(
            x=[times_right.min(), times_right.max()],
            y=[mean_right, mean_right],
            mode='lines',
            name=f'Średnia prawa ({mean_right:.2f}s)',
            line=dict(color='red', dash='solid', width=1)
        ))

        fig.update_layout(
            title="Okresy kroków w czasie",
            xaxis_title="Czas [s]",
            yaxis_title="Okres kroku [s]",
            hovermode="x unified",
            template="plotly_white",
            height=500
        )

        fig.update_yaxes(range=[0, 1.6])

        fig.show()

    def plot_test_step_peaks(self):
        signal = self.analyzer.get_combined_signal("left")
        peaks, _ = find_peaks(signal, prominence=20, distance=30)
        times = self.analyzer.time.iloc[peaks]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.analyzer.time, y=signal, mode='lines', name='sygnał'))
        fig.add_trace(go.Scatter(x=times, y=signal.iloc[peaks], mode='markers', name='kroki', marker=dict(color='red')))
        fig.show()

    def plot_step_phases(self, threshold: float = 10.0):
        phases_df = self.analyzer.detect_step_phases(threshold)

        times = phases_df['time']

        # Kolory do faz
        color_map = {
            'stance': 'green',
            'swing': 'red',
            'double_support': 'blue',
            'single_support': 'orange',
            'no_contact': 'gray'
        }

        fig = go.Figure()

        # Lewa stopa - kolorowanie punktów lub obszaru na osi czasu
        fig.add_trace(go.Scatter(
            x=times,
            y=[1]*len(times),
            mode='markers',
            marker=dict(
                color=phases_df['left'].map(color_map),
                size=8,
                symbol='square'
            ),
            name='Lewa stopa'
        ))

        # Prawa stopa - umieszczona na osi y=0.5 żeby się nie nakładało
        fig.add_trace(go.Scatter(
            x=times,
            y=[0.5]*len(times),
            mode='markers',
            marker=dict(
                color=phases_df['right'].map(color_map),
                size=8,
                symbol='square'
            ),
            name='Prawa stopa'
        ))

        # Globalna faza (dwupodporowa itp.) na osi y=1.5
        fig.add_trace(go.Scatter(
            x=times,
            y=[1.5]*len(times),
            mode='markers',
            marker=dict(
                color=phases_df['global'].map(color_map),
                size=8,
                symbol='circle'
            ),
            name='Faza globalna'
        ))

        fig.update_layout(
            title="Fazy chodu w czasie",
            yaxis=dict(
                tickvals=[0.5, 1, 1.5],
                ticktext=['Prawa stopa', 'Lewa stopa', 'Faza globalna'],
                range=[0, 2]
            ),
            xaxis_title="Czas [s]",
            height=400,
            template="plotly_white",
            showlegend=True
        )

        fig.show()
