import random
import numpy as np
import gym
from gym import spaces


class AirTrafficEnv(gym.Env):
    """Enhanced Custom Environment for Air Traffic Management."""

    def __init__(self):
        super(AirTrafficEnv, self).__init__()

        #Assign runways, assign gates, clear for taxiing/takeoff
        
        self.action_space = spaces.Discrete(15)  

        self.observation_space = spaces.Box(low=0, high=10, shape=(6,), dtype=np.float32)

        # Airport parameters
        
        self.weather_conditions = ['sunny', 'clear', 'cloudy', 'storm', 'fog']
        self.time_of_day = ['day', 'night']
        self.current_time_of_day = 'day'
        self.runways = 3
        self.gates = 5
        self.capacity = 10

    
        self.reset()

    def reset(self):
        
        """Reseting the environment to an initial state."""
        
        self.weather = random.choice(self.weather_conditions)
        self.current_time_of_day = random.choice(self.time_of_day)
        self.emergency_factor = 'low'
        self.emergency_flights = 0
        self.aircraft_list = self.generate_aircraft()
        self.runway_availability = [True] * self.runways
        self.gate_availability = [True] * self.gates
        self.state = self.get_state()
        return self.state

    def generate_aircraft(self):
        
        airlines = ['Air India', 'Emirates', 'Delta', 'Lufthansa', 'Qatar Airways']
        return [
            {
                "id": f"{airline[:2].upper()}-{i + 100}",
                "airline": airline,
                "status": random.choice(["On Time", "Delayed", "Emergency"]),
                "delayed": random.choice([True, False])
            }
            for i, airline in enumerate(random.choices(airlines, k=10))
        ]

    def get_state(self):
        
        """Returning the current state of the environment."""
        
        state = np.array([
            self.weather_conditions.index(self.weather),
            self.time_of_day.index(self.current_time_of_day),
            len(self.aircraft_list),
            self.emergency_flights,
            sum(self.runway_availability),
            sum(self.gate_availability)
        ], dtype=np.float32)
        return state

    def get_weather_conditions(self):
        
        """Adjusting parameters based on weather conditions."""
        
        if self.weather in ['storm', 'fog']:
            visibility = 0  # Low visibility
            capacity = max(3, self.capacity - 5)  # Reduced capacity
        elif self.weather == 'cloudy':
            visibility = 5
            capacity = max(4, self.capacity - 2)
        else:
            visibility = 10  # High visibility
            capacity = self.capacity  # Full capacity

        return capacity, visibility

    def get_emergency_factor(self):
        
        factor = {
            'low': 1,
            'medium': 2,
            'high': 5
        }
        return factor[self.emergency_factor]

    def step(self, action):
        
        reward = 0
        done = False

        # Capacity and visibility adjustments
        
        capacity, visibility = self.get_weather_conditions()

        if random.random() < 0.1:  # 10% chance of emergency
            self.emergency_flights += 1
            self.emergency_factor = random.choice(['low', 'medium', 'high'])
            reward += self.get_emergency_factor() * 5

    
        if action < self.runways:  # Assign runway
            if self.runway_availability[action]:
                self.runway_availability[action] = False
                reward += 15  # Reward for assigning a free runway
            else:
                reward -= 2  # Light penalty for occupied runway (not too harsh)

        elif action < self.runways + self.gates:  # Assign gate
            gate_index = action - self.runways
            if self.gate_availability[gate_index]:
                self.gate_availability[gate_index] = False
                reward += 10  # Reward for successful gate assignment
            else:
                reward -= 1  # Light penalty for occupied gate

        else:  # Handle taxiing or takeoff clearance
            if visibility > 3:
                reward += 10  # Positive reward for successful taxiing/takeoff in good conditions
            else:
                reward -= 5  # Penalize for attempting taxiing/takeoff in low visibility

        if random.random() < 0.05:
            self.weather = random.choice(self.weather_conditions)
        if random.random() < 0.05:
            self.current_time_of_day = random.choice(self.time_of_day)

        #termination conditions
        if self.emergency_flights >= 5 or len(self.aircraft_list) == 0:
            done = True


        self.state = self.get_state()

        # Monitoring information
        info = {
            "weather": self.weather,
            "time_of_day": self.current_time_of_day,
            "runway_availability": self.runway_availability,
            "gate_availability": self.gate_availability,
            "emergency_flights": self.emergency_flights,
            "aircraft_list": self.aircraft_list,
        }

        return self.state, reward, done, info