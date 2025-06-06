import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks
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

    def plot_steps_period(self, prominence, distance):
        # Lewa stopa
        left_signal = self.analyzer.get_combined_signal("left")
        left_step_times = self.analyzer.detect_steps(left_signal, prominence, distance)
        times_left, periods_left = self.analyzer.compute_step_periods(left_step_times)

        # Prawa stopa
        right_signal = self.analyzer.get_combined_signal("right")
        right_step_times = self.analyzer.detect_steps(right_signal, prominence, distance)
        times_right, periods_right = self.analyzer.compute_step_periods(right_step_times)

        # Rysowanie
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=times_left,
            y=periods_left,
            mode='markers+lines',
            name='Lewa stopa',
            marker=dict(color='blue', symbol='circle')
        ))

        fig.add_trace(go.Scatter(
            x=times_right,
            y=periods_right,
            mode='markers+lines',
            name='Prawa stopa',
            marker=dict(color='red', symbol='x')
        ))

        fig.update_layout(
            title="Okresy kroków w czasie (lewa i prawa stopa)",
            xaxis_title="Czas [s]",
            yaxis_title="Okres kroku [s]",
            template="plotly_white",
            height=500
        )

        fig.show()


    def plot_test_step_peaks(self):
        signal = self.analyzer.get_combined_signal("left")
        peaks, _ = find_peaks(signal, prominence=20, distance=30)
        times = self.analyzer.time.iloc[peaks]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.analyzer.time, y=signal, mode='lines', name='sygnał'))
        fig.add_trace(go.Scatter(x=times, y=signal.iloc[peaks], mode='markers', name='kroki', marker=dict(color='red')))
        fig.show()
