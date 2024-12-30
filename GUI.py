import random
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QLabel, QPushButton, QComboBox, QTableWidget, QTableWidgetItem, QHBoxLayout, QWidget, QHeaderView
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QColor, QBrush
from stable_baselines3 import DQN
from datetime import datetime, timedelta
import pyqtgraph as pg
import numpy as np
import matplotlib.pyplot as plt
from airport_environment import AirTrafficEnv

# Generate a random takeoff time within the next 5 hours
def generate_random_takeoff_time():
    current_time = datetime.now()
    random_minutes = random.randint(0, 5 * 60) 
    departure_time = current_time + timedelta(minutes=random_minutes)
    return departure_time.strftime("%H:%M:%S")

class AirTrafficGUI(QMainWindow):
    def __init__(self, rl_model, env):
        super().__init__()
        self.setWindowTitle("Air Traffic Management System")
        self.setGeometry(100, 100, 800, 600)

        # RL Model and Environment
        self.rl_model = rl_model
        self.env = env
        self.state = self.env.reset()
        self.total_rewards = 0

        self.initUI()
        self.update_flight_table()
        self.update_environment_info()

        # Timer for RL Model decision-making
        self.rl_timer = QTimer()
        self.rl_timer.timeout.connect(self.run_rl_step)
        self.rl_timer.start(2000)  # RL model decision every 2 seconds

        # Timer for real-time flight updates
        self.flight_timer = QTimer()
        self.flight_timer.timeout.connect(self.update_flight_status)
        self.flight_timer.start(5000)  # Update every 5 seconds

    def initUI(self):
        main_layout = QVBoxLayout()

        # Weather Control
        weather_layout = QHBoxLayout()
        weather_label = QLabel("Weather:")
        self.weather_combo = QComboBox()
        if isinstance(self.env.weather_conditions, list):
            self.weather_combo.addItems(self.env.weather_conditions)
        self.weather_combo.setCurrentText(self.env.weather if isinstance(self.env.weather, str) else "Default")
        weather_layout.addWidget(weather_label)
        weather_layout.addWidget(self.weather_combo)
        self.weather_combo.currentTextChanged.connect(self.change_weather)

        # Time of Day Control
        time_layout = QHBoxLayout()
        time_label = QLabel("Time of Day:")
        self.time_combo = QComboBox()
        if isinstance(self.env.time_of_day, list):
            self.time_combo.addItems(self.env.time_of_day)
            self.time_combo.setCurrentText(self.env.time_of_day[0] if self.env.time_of_day else "Default")
        self.time_combo.currentTextChanged.connect(self.change_time_of_day)
        time_layout.addWidget(time_label)
        time_layout.addWidget(self.time_combo)

        # Emergency Flight Button
        self.emergency_button = QPushButton("Mark Selected Flight as Emergency")
        self.emergency_button.clicked.connect(self.mark_emergency)

        # Flight Table
        self.flight_table = QTableWidget()
        self.flight_table.setColumnCount(6)
        self.flight_table.setHorizontalHeaderLabels(
            ["Flight ID", "Airline", "Status", "Gate/Runway", "Emergency", "Takeoff Time / Gate"]
        )
        self.flight_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # Environment Info
        self.environment_info = QLabel()
        self.environment_info.setStyleSheet("font-size: 14px;")

        # Total Rewards Display
        self.rewards_label = QLabel(f"Total Rewards: {self.total_rewards}")
        self.rewards_label.setStyleSheet("font-size: 16px; font-weight: bold; color: green;")

        # Adding widgets to layout
        main_layout.addLayout(weather_layout)
        main_layout.addLayout(time_layout)
        main_layout.addWidget(self.emergency_button)
        main_layout.addWidget(self.flight_table)
        main_layout.addWidget(self.environment_info)
        main_layout.addWidget(self.rewards_label)

        # Central widget
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def change_weather(self, selected_weather):
        
        if selected_weather in self.env.weather_conditions:
            self.env.weather = selected_weather
            self.update_environment_info()

    def change_time_of_day(self, time_of_day):
        
        if time_of_day in self.env.time_of_day:
            self.env.current_time_of_day = time_of_day
            self.update_environment_info()

    def mark_emergency(self):
        
        """Mark the selected flight as emergency."""
        
        selected_row = self.flight_table.currentRow()
        if selected_row != -1:
            flight_id = self.flight_table.item(selected_row, 0).text()
            for flight in self.env.aircraft_list:
                if flight["id"] == flight_id:
                    flight["status"] = "Emergency"
                    self.env.emergency_flights += 1
                    break
            self.update_flight_table()
            self.update_environment_info()

    def run_rl_step(self):
    
        action, _ = self.rl_model.predict(self.state)
        self.state, reward, done, info = self.env.step(action)
        self.total_rewards += reward
        self.update_flight_table()
        self.update_environment_info()
        self.rewards_label.setText(f"Total Rewards: {self.total_rewards}")

        if done:
            self.state = self.env.reset()
            self.total_rewards = 0

    def update_flight_status(self):
        """Update the status of flights dynamically."""
        for flight in self.env.aircraft_list:
            if flight["status"] != "Emergency":
                flight["status"] = random.choice(["On Time", "Delayed"])
                flight["gate_runway"] = random.choice(["Gate 1", "Runway 2", "Gate 3"])
                
                if flight["status"] == "On Time":
                    # Assign a random takeoff time for on-time flights within the next 5 hours
                    flight["takeoff_time"] = generate_random_takeoff_time()
                    flight["takeoff_gate"] = random.choice(["Gate 1", "Gate 2", "Runway 1", "Runway 2"])
                elif flight["status"] == "Delayed":
                    flight["takeoff_time"] = "Delayed"
                    flight["takeoff_gate"] = "N/A"
                    
            # Simulating departure
            if flight["status"] == "On Time" and random.random() < 0.1:
                flight["status"] = "Departed"
                flight["takeoff_time"] = "Departed"
                flight["takeoff_gate"] = "N/A"
        self.update_flight_table()

    def update_flight_table(self):
        self.flight_table.setRowCount(len(self.env.aircraft_list))
        for row, flight in enumerate(self.env.aircraft_list):
            self.flight_table.setItem(row, 0, QTableWidgetItem(flight["id"]))
            self.flight_table.setItem(row, 1, QTableWidgetItem(flight["airline"]))

            # Status column with color coding
            status_item = QTableWidgetItem(flight["status"])
            if flight["status"] == "On Time":
                status_item.setBackground(QColor("green"))
                status_item.setForeground(QBrush(QColor("white")))
            elif flight["status"] == "Delayed":
                status_item.setBackground(QColor("yellow"))
                status_item.setForeground(QBrush(QColor("black")))
            elif flight["status"] == "Emergency":
                status_item.setBackground(QColor("red"))
                status_item.setForeground(QBrush(QColor("white")))
            self.flight_table.setItem(row, 2, status_item)

            self.flight_table.setItem(row, 3, QTableWidgetItem(flight.get("gate_runway", "N/A")))
            self.flight_table.setItem(row, 4, QTableWidgetItem("Yes" if flight["status"] == "Emergency" else "No"))

            if flight["status"] == "Emergency":
                takeoff_info = "Emergency - No Gate/Time"
            else:
                takeoff_info = f"{flight.get('takeoff_time', 'N/A')} / {flight.get('takeoff_gate', 'N/A')}"
            
            self.flight_table.setItem(row, 5, QTableWidgetItem(takeoff_info))

    def update_environment_info(self):
        """Display the current environment information."""
        info = f"""
        <b>Current Weather:</b> {self.env.weather}<br>
        <b>Time of Day:</b> {self.env.current_time_of_day}<br>
        <b>Available Runways:</b> {sum(self.env.runway_availability)}<br>
        <b>Available Gates:</b> {sum(self.env.gate_availability)}<br>
        <b>Emergency Flights:</b> {self.env.emergency_flights}
        """
        self.environment_info.setText(info)

if __name__ == "__main__":
    
    env = AirTrafficEnv()
    rl_model = DQN.load("air_traffic_model")

    app = QApplication(sys.argv)

    # Create the dashboard window
    dashboard = AirTrafficGUI(rl_model, env)
    dashboard.show()

    sys.exit(app.exec_())
