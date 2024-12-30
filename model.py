from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from airport_environment import AirTrafficEnv


env = AirTrafficEnv()

env = DummyVecEnv([lambda: env])

model = DQN(
    'MlpPolicy',
     env,
    verbose=1,
    learning_rate=1e-4,  
    buffer_size=100000,   
    learning_starts=500,  
    batch_size=128,       
    gamma=0.99,         
    target_update_interval=500,  
    exploration_fraction=0.1,    
    exploration_final_eps=0.05, 
)

model.learn(total_timesteps=100000)
model.save("air_traffic_modell")



total_reward = 0
state = env.reset()
done = False
while not done:
    action, _ = model.predict(state, deterministic=True)
    state, reward, done, info = env.step(action)
    total_reward += reward

print(f"Total Reward: {total_reward}")


