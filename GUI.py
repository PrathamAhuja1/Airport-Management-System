import random
import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QLabel, 
                            QPushButton, QComboBox, QTableWidget, QTableWidgetItem, 
                            QHBoxLayout, QWidget, QHeaderView, QFrame, QGridLayout,
                            QGroupBox, QStyleFactory)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QColor, QBrush, QFont
from stable_baselines3 import DQN
from datetime import datetime, timedelta
import numpy as np
from airport_environment import AirTrafficEnv


class AirTrafficGUI(QMainWindow):
    
    def __init__(self, model, env):
        super().__init__()
        self.rl_model = model
        self.env = env
        self.state = env.reset()
        self.total_rewards = 0.0
        
        self.setWindowTitle('Air Traffic Control Simulator')
        self.setGeometry(100, 100, 1200, 800)
        self.initUI()
        
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_flight_status)
        self.update_timer.start(12000)
        
        self.rl_timer = QTimer()
        self.rl_timer.timeout.connect(self.run_rl_step)
        self.rl_timer.start(1000)
        
        self.alert_timer = QTimer()
        self.alert_timer.timeout.connect(self.check_alerts)
        self.alert_timer.start(2000)
        
        
    def initUI(self):
        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)

        # Header section #
        header_group = QGroupBox("Airport Control Panel")
        header_layout = QGridLayout()

        # Weather Control #
        weather_label = QLabel("Weather Conditions:")
        weather_label.setFont(QFont('Arial', 10, QFont.Bold))
        self.weather_combo = QComboBox()
        self.weather_combo.addItems(self.env.weather_conditions)
        self.weather_combo.setCurrentText(self.env.weather)
        header_layout.addWidget(weather_label, 0, 0)
        header_layout.addWidget(self.weather_combo, 0, 1)
        self.weather_combo.currentTextChanged.connect(self.change_weather)

        # Time of Day Control #
        time_label = QLabel("Time of Day:")
        time_label.setFont(QFont('Arial', 10, QFont.Bold))
        self.time_combo = QComboBox()
        self.time_combo.addItems(self.env.time_of_day)
        self.time_combo.setCurrentText(self.env.current_time_of_day)
        header_layout.addWidget(time_label, 0, 2)
        header_layout.addWidget(self.time_combo, 0, 3)
        self.time_combo.currentTextChanged.connect(self.change_time_of_day)

        # Emergency Button #
        self.emergency_button = QPushButton("Mark Selected Flight as Emergency")
        self.emergency_button.setStyleSheet("""
            QPushButton {
                background-color: #ff4444;
                color: white;
                padding: 5px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #ff0000;
            }
        """)
        header_layout.addWidget(self.emergency_button, 0, 4)
        self.emergency_button.clicked.connect(self.mark_emergency)
        header_group.setLayout(header_layout)

        # Alert Panel #
        alert_group = QGroupBox("System Alerts")
        alert_layout = QVBoxLayout()
        self.alert_panel = QLabel("System Operating Normally")
        self.alert_panel.setStyleSheet("""
            QLabel {
                background-color: #e8f5e9;
                color: #2e7d32;
                padding: 10px;
                border: 1px solid #81c784;
                border-radius: 5px;
            }
        """)
        alert_layout.addWidget(self.alert_panel)
        alert_group.setLayout(alert_layout)

        # Flight Table #
        table_group = QGroupBox("Flight Information")
        table_layout = QVBoxLayout()
        self.flight_table = QTableWidget()
        self.flight_table.setColumnCount(7)
        self.flight_table.setHorizontalHeaderLabels([
            "Flight ID", "Airline", "Status", 
            "Flight Status", "Gate/Runway", "Schedule", "Emergency"
        ])
        self.flight_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table_layout.addWidget(self.flight_table)
        table_group.setLayout(table_layout)

        # Environment Info #
        info_group = QGroupBox("Environment Statistics")
        info_layout = QVBoxLayout()
        self.environment_info = QLabel()
        self.rewards_label = QLabel(f"Total Rewards: {self.total_rewards}")
        info_layout.addWidget(self.environment_info)
        info_layout.addWidget(self.rewards_label)
        info_group.setLayout(info_layout)

        main_layout.addWidget(header_group)
        main_layout.addWidget(alert_group)
        main_layout.addWidget(table_group)
        main_layout.addWidget(info_group)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)


    def get_flight_status_color(self, flight_status):
        
        """Get color for flight status (On Time/Delayed/Emergency)."""
        
        status_colors = {
            "On Time": (QColor(76, 175, 80), QColor("white")),  
            "Delayed": (QColor(255, 152, 0), QColor("black")),  
            "Emergency": (QColor(244, 67, 54), QColor("white")) 
        }
        return status_colors.get(flight_status, (QColor("white"), QColor("black")))



    def get_operation_status_color(self, status):
        
        """Get color for operation status (Landing/Taxiing/etc)."""
        
        status_colors = {
            "Landing": (QColor(3, 169, 244), QColor("white")),        
            "Taxiing to Gate": (QColor(255, 193, 7), QColor("black")), 
            "At Gate": (QColor(76, 175, 80), QColor("white")),      
            "Taxiing to Runway": (QColor(255, 152, 0), QColor("black")), 
            "Taking Off": (QColor(33, 150, 243), QColor("white")),    
            "Departed": (QColor(158, 158, 158), QColor("white"))      
        }
        return status_colors.get(status, (QColor("white"), QColor("black")))



    def update_flight_table(self):
        self.flight_table.setRowCount(len(self.env.aircraft_list))

        for row, flight in enumerate(self.env.aircraft_list):
            self.flight_table.setItem(row, 0, QTableWidgetItem(flight["id"]))
            
            
            self.flight_table.setItem(row, 1, QTableWidgetItem(flight["airline"]))
            
        
            status_item = QTableWidgetItem(flight["status"])
            bg_color, text_color = self.get_operation_status_color(flight["status"])
            status_item.setBackground(bg_color)
            status_item.setForeground(QBrush(text_color))
            self.flight_table.setItem(row, 2, status_item)
            

            flight_status_item = QTableWidgetItem(flight["flight_status"])
            bg_color, text_color = self.get_flight_status_color(flight["flight_status"])
            flight_status_item.setBackground(bg_color)
            flight_status_item.setForeground(QBrush(text_color))
            self.flight_table.setItem(row, 3, flight_status_item)
            

            location_text = "—"
            if flight["status"] in ["Landing", "Taking Off", "Taxiing to Runway"] and flight["runway"] is not None:
                if flight["status"] == "Taxiing to Runway":
                    location_text = f"Runway-{flight['runway']} (Assigned)"
                else:
                    position = "Landing" if flight["status"] == "Landing" else "Takeoff"
                    location_text = f"Runway-{flight['runway']} ({position})"
            elif flight["status"] in ["At Gate", "Taxiing to Gate"] and flight["id"] in self.env.flight_gates:
                gate_num = self.env.flight_gates[flight["id"]]
                location_text = f"Gate-{gate_num}"
            
            location_item = QTableWidgetItem(location_text)
            location_item.setTextAlignment(Qt.AlignCenter)
            self.flight_table.setItem(row, 4, location_item)
            
    
            self.flight_table.setItem(row, 5, QTableWidgetItem(flight["scheduled_time"]))
            

            emergency_item = QTableWidgetItem("EMERGENCY" if flight["emergency"] else "—")
            if flight["emergency"]:
                emergency_item.setBackground(QColor(244, 67, 54))
                emergency_item.setForeground(QBrush(QColor("white")))
            self.flight_table.setItem(row, 6, emergency_item)

            for col in range(7):
                if self.flight_table.item(row, col):
                    self.flight_table.item(row, col).setTextAlignment(Qt.AlignCenter)        
            
            
            
            
    def check_alerts(self):
        
        """Check for system alerts and emergencies"""
        
        alerts = []
        
        emergency_flights = [f for f in self.env.aircraft_list if f["emergency"]]
        if emergency_flights:
            alerts.append(f"⚠️ {len(emergency_flights)} Emergency Flights Active!")
            
    
        if self.env.weather in ['storm', 'fog']:
            alerts.append(f"⚠️ Adverse Weather: {self.env.weather}")
            

        if self.env.occupied_gates >= self.env.num_gates * 0.9:
            alerts.append("⚠️ Gate Capacity Critical!")
            

        if alerts:
            self.alert_panel.setText("\n".join(alerts))
            self.alert_panel.setStyleSheet("""
                QLabel {
                    background-color: #ffebee;
                    color: #c62828;
                    padding: 10px;
                    border: 1px solid #ef5350;
                    border-radius: 5px;
                    font-weight: bold;
                }
            """)
        else:
            self.alert_panel.setText("System Operating Normally")
            self.alert_panel.setStyleSheet("""
                QLabel {
                    background-color: #e8f5e9;
                    color: #2e7d32;
                    padding: 10px;
                    border: 1px solid #81c784;
                    border-radius: 5px;
                }
            """)        



    def update_environment_info(self):
        
        """Update the environment information display."""
        
        info = f"""
        <table style='width: 100%; margin: 10px;'>
            <tr>
                <td><b>Current Weather:</b></td>
                <td>{self.env.weather}</td>
                <td><b>Time of Day:</b></td>
                <td>{self.env.current_time_of_day}</td>
            </tr>
            <tr>
                <td><b>Available Gates:</b></td>
                <td>{self.env.num_gates - self.env.occupied_gates} / {self.env.num_gates}</td>
                <td><b>Occupied Gates:</b></td>
                <td>{self.env.occupied_gates}</td>
            </tr>
            <tr>
                <td><b>Available Runways:</b></td>
                <td>{self.env.num_runways - len([q for q in self.env.runway_queue.values() if q])} / {self.env.num_runways}</td>
                <td><b>Emergency Flights:</b></td>
                <td>{self.env.emergency_flights}</td>
            </tr>
            <tr>
                <td><b>Total Active Flights:</b></td>
                <td>{len(self.env.aircraft_list)}</td>
                <td><b>Weather Status:</b></td>
                <td>{'Normal' if self.env.weather not in ['storm', 'fog'] else 'Adverse Conditions'}</td>
            </tr>
        </table>
        """
        self.environment_info.setText(info)



    def change_weather(self, selected_weather):
        self.env.weather = selected_weather
        self.update_environment_info()



    def change_time_of_day(self, time_of_day):
        self.env.current_time_of_day = time_of_day
        self.update_environment_info()



    def mark_emergency(self):
        selected_row = self.flight_table.currentRow()
        if selected_row != -1:
            flight_id = self.flight_table.item(selected_row, 0).text()
            for flight in self.env.aircraft_list:
                if flight["id"] == flight_id and not flight["emergency"]:
                    flight["status"] = "Emergency"
                    flight["flight_status"] = "Emergency"
                    flight["emergency"] = True
                    self.env.emergency_flights += 1
                    break
            self.update_flight_table()
            self.update_environment_info()



    def run_rl_step(self):
        """Execute one step of the RL model."""
        action, _ = self.rl_model.predict(self.state)
        self.state, reward, done, info = self.env.step(action)
        self.total_rewards += reward
        self.rewards_label.setText(f"Total Rewards: {self.total_rewards:,.2f}")
        self.update_flight_table()
        self.update_environment_info()



    def update_flight_status(self):
        
        """Update flight statuses based on environment step."""
        
        self.state = self.env.reset() 
        self.update_flight_table()
        self.update_environment_info()



def main():
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create('Fusion'))
    env = AirTrafficEnv()
    model = DQN.load("air_traffic_model")
    window = AirTrafficGUI(model, env)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()