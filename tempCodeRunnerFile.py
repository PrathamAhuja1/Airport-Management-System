table_group = QGroupBox("Flight Information")
        table_layout = QVBoxLayout()
        self.flight_table = QTableWidget()
        self.flight_table.setColumnCount(8)
        self.flight_table.setHorizontalHeaderLabels([
            "Flight ID", "Airline", "Status", "Assignment", 
            "Flight Status", "Gate/Runway", "Schedule", "Emergency"
        ])
        self.flight_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table_layout.addWidget(self.flight_table)
        table_group.setLayout(table_layout)