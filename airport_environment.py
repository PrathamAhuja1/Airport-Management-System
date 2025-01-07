import random
import numpy as np
import gym
from gym import spaces
from datetime import datetime, timedelta



class AirTrafficEnv(gym.Env):
    def __init__(self):
        super(AirTrafficEnv, self).__init__()
        self.action_space = spaces.Discrete(15)
        self.observation_space = spaces.Box(low=0, high=10, shape=(6,), dtype=np.float32)

        self.weather_conditions = ['sunny', 'clear', 'cloudy', 'storm', 'fog']
        self.time_of_day = ['day', 'night']
        self.current_time_of_day = 'day'
        self.weather = 'clear'
        self.num_runways = 3
        self.num_gates = 12
        self.capacity = 15
        self.occupied_gates = 0
        self.emergency_flights = 0
        self.current_emergency_flights = 0
        self.total_emergency_flights = 0
        
        self.aircraft_list = []
        self.flight_gates = {}
        self.runway_queue = {i: [] for i in range(self.num_runways)}
        self.preserved_gates = {}
        self.last_reset_time = datetime.now()
        self.runway_availability = [True] * self.num_runways
        
        self.emergency_history = []
        self.delay_history = []
        
        self.aircraft_list = self.generate_aircraft()
        self.state = self.get_state()



    def reset(self):
        current_time = datetime.now()
        if (current_time - self.last_reset_time).seconds >= 12:
            self.emergency_flights = 0
            new_aircraft = self.generate_aircraft()
            
            active_gates = {flight_id: gate for flight_id, gate in self.preserved_gates.items() 
                          if any(f["id"] == flight_id and f["status"] != "Departed" 
                               for f in self.aircraft_list)}
            self.preserved_gates = active_gates
            self.flight_gates = active_gates.copy()
            
            self.aircraft_list = new_aircraft
            self.runway_availability = [True] * self.num_runways
            self.last_reset_time = current_time
            
        self.update_occupied_gates()
        self.state = self.get_state()
        return self.state



    def generate_aircraft(self):
        airlines = ['Air India', 'Emirates', 'Delta', 'Lufthansa', 'Qatar Airways']
        aircraft = []
        current_time = datetime.now()
        
        for i in range(15):
            is_emergency = random.random() < 0.02
            if is_emergency:
                self.emergency_flights += 1
                self.current_emergency_flights += 1
            
            time_offset = random.randint(0, 5 * 60)
            scheduled_time = current_time + timedelta(minutes=time_offset)
            
            is_delayed = not is_emergency and random.random() < 0.15
            status = "Emergency" if is_emergency else ("Delayed" if is_delayed else "On Time")
            
            arriving = is_emergency or random.random() < 0.5
            stage = 0 if arriving else 2
            scheduled_time_str = f"{'Landing' if arriving else 'Takeoff'}: {scheduled_time.strftime('%H:%M')}"
            
            aircraft.append({
                "id": f"{random.choice(airlines)[:2].upper()}-{i + 100}",
                "airline": random.choice(airlines),
                "status": "Emergency Landing" if is_emergency else ("Landing" if arriving else "At Gate"),
                "flight_status": status,
                "gate": None,
                "runway": None,
                "assignment": None,
                "emergency": is_emergency,
                "taxi_start_time": None,
                "scheduled_time": scheduled_time_str,
                "processing_stage": stage,
                "priority": 1 if is_emergency else 0,
                "fuel_status": "Critical" if is_emergency else "Normal",
                "last_update": current_time
            })
        
        aircraft.sort(key=lambda x: (-x["priority"], x["last_update"]))
        return aircraft
    
    
    
    def handle_emergencies(self):
        
        """Priority handling for emergency flights"""
        
        emergency_flights = [f for f in self.aircraft_list if f["emergency"]]
        for flight in emergency_flights:

            if flight["status"] == "Emergency Landing":
                
                runway = min(self.runway_queue.keys(), 
                           key=lambda r: len(self.runway_queue[r]))
                
                
                if self.runway_queue[runway]:
                    displaced_flight = next((f for f in self.aircraft_list 
                                          if f["id"] == self.runway_queue[runway][0]), None)
                    if displaced_flight and not displaced_flight["emergency"]:
                        displaced_flight["assignment"] = None
                        self.runway_queue[runway].pop(0)
                
                # Assigning runway to emergency flight#
                
                if flight["id"] not in self.runway_queue[runway]:
                    self.runway_queue[runway].insert(0, flight["id"])
                    flight["assignment"] = f"Runway {runway} (Emergency)"

    def calculate_reward(self, flight):
        
        """Calculate reward based on conditions and flight handling"""
        
        base_reward = 1.0
        
        # Extra Reward for adverse conditions #
        if self.weather in ['storm', 'fog']:
            base_reward *= 2.0
        if self.current_time_of_day == 'night':
            base_reward *= 1.5
            
        # Assigning reward for handling emergency flights #
        if flight["emergency"]:
            base_reward *= 3.0
            
        # Penalty for delays #
        if flight["flight_status"] == "Delayed":
            base_reward *= 0.5
            
        return base_reward

    def update_occupied_gates(self):
        
        """Updating occupied gates count"""
        
        self.occupied_gates = len(set(self.flight_gates.values()))


    def release_gate(self, flight_id):
        
        if flight_id in self.flight_gates:
            self.flight_gates.pop(flight_id)
            self.preserved_gates.pop(flight_id, None)
            self.update_occupied_gates()

    def assign_runway(self, flight):
        
        """Assign a runway to a flight"""
        
        # Finding runway with shortest queue #
        runway = min(self.runway_queue.keys(), 
                    key=lambda r: len(self.runway_queue[r]))
        
        if flight["id"] not in [id for queue in self.runway_queue.values() for id in queue]:
            self.runway_queue[runway].append(flight["id"])
            flight["runway"] = runway
            flight["assignment"] = f"Runway {runway}"

    def step(self, action):
        reward = 0
        done = False

        for flight in self.aircraft_list:
            if not flight["emergency"]:
            
                emergency_chance = 0.001
                if self.weather in ['storm', 'fog']:
                    emergency_chance *= 2
                if flight["flight_status"] == "Delayed":
                    emergency_chance *= 1.5
                
                if random.random() < emergency_chance:
                    flight["emergency"] = True
                    flight["status"] = "Emergency"
                    flight["flight_status"] = "Emergency"
                    flight["priority"] = 1
                    self.total_emergency_flights += 1
                    self.current_emergency_flights += 1
                    self.emergency_history.append({
                        "time": datetime.now(),
                        "flight_id": flight["id"],
                        "cause": "In-flight Emergency"
                    })

        # Priority handling for emergencies #
        self.handle_emergencies()
        
        for flight in self.aircraft_list:
            old_status = flight["status"]
            self.progress_flight_status(flight)
            
            if old_status != flight["status"]:
                reward += self.calculate_reward(flight)

        # Updating active flights #
        self.aircraft_list = [f for f in self.aircraft_list if f["status"] != "Departed"]
        while len(self.aircraft_list) < 15:
            new_aircraft = self.generate_aircraft()[0]
            self.aircraft_list.append(new_aircraft)
        
        self.aircraft_list.sort(key=lambda x: (-x["priority"], x["last_update"]))

        self.state = self.get_state()
        info = self.get_info()
        return self.state, reward, done, info
    
    def get_info(self):
        
        """Get detailed environment information"""
        
        return {
            "weather": self.weather,
            "time_of_day": self.current_time_of_day,
            "runway_queue": self.runway_queue,
            "flight_gates": self.flight_gates,
            "total_emergency_flights": self.total_emergency_flights,
            "current_emergency_flights": self.current_emergency_flights,
            "occupied_gates": self.occupied_gates,
            "emergency_history": self.emergency_history[-5:],
            "delay_history": self.delay_history[-5:]
        }
    
    def assign_gate(self, flight):
        
        """Assign an available gate to a flight"""
        
        if flight["id"] in self.flight_gates:
            return self.flight_gates[flight["id"]]
            
        # Finding first available gate #
        used_gates = set(self.flight_gates.values())
        available_gates = set(range(self.num_gates)) - used_gates
        
        if available_gates:
            gate = min(available_gates)
            self.flight_gates[flight["id"]] = gate
            self.preserved_gates[flight["id"]] = gate
            self.update_occupied_gates()
            return gate
        return None
    
    
    

    def progress_flight_status(self, flight):
        
        """Progress flight through stages with realistic timing"""
        
        if flight["emergency"]:
            return

        current_time = datetime.now()
        
        if flight["status"] == "Landing":
            if not flight["runway"]:
                self.assign_runway(flight)
            if random.random() < 0.15:
                flight["status"] = "Taxiing to Gate"
                flight["taxi_start_time"] = current_time
                assigned_gate = self.assign_gate(flight)
                if flight["runway"] is not None:
                    queue = self.runway_queue[flight["runway"]]
                    if queue and queue[0] == flight["id"]:
                        queue.pop(0)
                    flight["runway"] = None
                
        elif flight["status"] == "Taxiing to Gate":
            if flight["taxi_start_time"] and \
            (current_time - flight["taxi_start_time"]).seconds >= 120:
                if flight["id"] in self.flight_gates:
                    flight["status"] = "At Gate"
                    
        elif flight["status"] == "At Gate":
            if random.random() < 0.1:
                self.release_gate(flight["id"])
                flight["status"] = "Taxiing to Runway"
                flight["taxi_start_time"] = current_time
                self.assign_runway(flight)
                
        elif flight["status"] == "Taxiing to Runway":
            if flight["taxi_start_time"] and \
            (current_time - flight["taxi_start_time"]).seconds >= 120:
                if flight["runway"] is not None:
                    flight["status"] = "Taking Off"
                
        elif flight["status"] == "Taking Off":
            runway = flight["runway"]
            if runway is not None and \
            self.runway_queue[runway] and \
            self.runway_queue[runway][0] == flight["id"]:
                if random.random() < 0.2:
                    self.runway_queue[runway].pop(0)
                    flight["status"] = "Departed"
                    flight["runway"] = None

    def get_state(self):
        
        """Return current state"""
        
        return np.array([
            self.weather_conditions.index(self.weather),
            self.time_of_day.index(self.current_time_of_day),
            len(self.aircraft_list),
            self.emergency_flights,
            self.num_runways - len([q for q in self.runway_queue.values() if q]),
            self.num_gates - self.occupied_gates
        ], dtype=np.float32)