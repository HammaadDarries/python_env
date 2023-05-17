import numpy as np
from simple_f1tenth_drl.PlannerUtils.TrackLine import TrackLine

import csv
import os, shutil
import matplotlib.pyplot as plt

from simple_f1tenth_drl.PlannerUtils.VehicleStateHistory import VehicleStateHistory

NUMBER_SCANS = 1
NUMBER_BEAMS = 10
MAX_SPEED = 2
MAX_STEER = 0.4
RANGE_FINDER_SCALE = 10
NOISE_FACTOR = 0.75
 

class EndToEndTest: 
    def __init__(self, agent, map_name, test_name):
        self.scan_buffer = np.zeros((NUMBER_SCANS, NUMBER_BEAMS))
        self.state_space = NUMBER_SCANS * NUMBER_BEAMS
        self.action_space = 2
        
        self.agent = agent
        self.vehicle_state_history = VehicleStateHistory(test_name, map_name)
        self.track_line = TrackLine(map_name, False, False)
        self.action_history = []
        self.collisions = 0
        self.data_plots_dir = "Data Plots"
        os.makedirs(self.data_plots_dir, exist_ok=True)  # Create directory for data plots
    
    def plan(self, obs):
        nn_state = self.transform_obs(obs)
        
        if obs['linear_vels_x'][0] < 1: # prevents unstable behavior at low speeds
            action = np.array([0, 2])
            return action

        nn_act = self.agent.act(nn_state)
        action = self.transform_action(nn_act)
        
        self.action_history.append(action)  # Store action in the history
        
        self.vehicle_state_history.add_memory_entry(obs, action)
        
        return action 

    def transform_obs(self, obs):
        """
        Transforms the observation received from the environment into a vector which can be used with a neural network.
    
        Args:
            obs: observation from env

        Returns:
            nn_obs: observation vector for neural network
        """
            
        scan = np.array(obs['scans'][0]) 
        # Add noise to the scan data
        noise = np.random.normal(loc=0, scale=NOISE_FACTOR, size=scan.shape)
        noisy_scan = scan + noise
        scaled_scan = noisy_scan/RANGE_FINDER_SCALE
        scan = np.clip(scaled_scan, 0, 1)

        if self.scan_buffer.all() ==0: # first reading
            for i in range(NUMBER_SCANS):
                self.scan_buffer[i, :] = scan 
        else:
            self.scan_buffer = np.roll(self.scan_buffer, 1, axis=0)
            self.scan_buffer[0, :] = scan

        nn_obs = np.reshape(self.scan_buffer, (NUMBER_BEAMS * NUMBER_SCANS))

        return nn_obs
    
    def collision_rate(self, laps):
        collision_rate = (self.collisions/laps) * 100
        print("Collision Rate = " + str(collision_rate) + "%")
    
    def transform_action(self, nn_action):
        steering_angle = nn_action[0] *   MAX_STEER
        speed = (nn_action[1] + 1) * (MAX_SPEED  / 2 - 0.5) + 1
        speed = min(speed, MAX_SPEED) # cap the speed

        action = np.array([steering_angle, speed])

        return action
    
    
    def done_callback(self, final_obs):
        self.vehicle_state_history.add_memory_entry(final_obs, np.array([0, 0]))
        self.vehicle_state_history.save_history()
        
        progress = self.track_line.calculate_progress_percent([final_obs['poses_x'][0], final_obs['poses_y'][0]]) * 100
        
        if(final_obs['collisions'][0] == True):
            self.collisions = self.collisions + 1
        print(f"Test lap complete --> Time: {final_obs['lap_times'][0]:.2f}, Colission: {bool(final_obs['collisions'][0])}, Lap p: {progress:.1f}%")


    def save_action_history(self, filename):
        raw_data_dir = "Raw Data"
        os.makedirs(raw_data_dir, exist_ok=True)  # Create "Raw Data" directory
        
        filename = os.path.join(raw_data_dir, "action_history.csv")
        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Steering Angle", "Speed"])  # Write header row
            for action in self.action_history:
                writer.writerow(action)  # Write action to CSV file
        print(f"Action history saved to {filename}")

    def generate_plots(self, map, final_obs):
        time_scale = 0.1  # Time scale: 10 actions per 1 second
        time = np.arange(len(self.action_history)) * time_scale  # Time points
        
        map_dir = os.path.join(self.data_plots_dir, map)
        os.makedirs(map_dir, exist_ok=True)  # Create directory for the specific map
        
        # Plot steering angle over time
        steering_angles = [action[0] for action in self.action_history]
        plt.figure()
        plt.plot(time, steering_angles)
        plt.xlabel("Time (s)")
        plt.ylabel("Steering Angle")
        plt.title(f"Steering Angle over Time\nLap Time: {final_obs['lap_times'][0]:.2f} seconds")
        plt.savefig(os.path.join(map_dir, "steering_angle.png"))
        
        # Plot speed over time
        speeds = [action[1] for action in self.action_history]
        plt.figure()
        plt.plot(time, speeds)
        plt.xlabel("Time (s)")
        plt.ylabel("Speed")
        plt.title(f"Speed over Time\nLap Time: {final_obs['lap_times'][0]:.2f} seconds")
        plt.savefig(os.path.join(map_dir, "speed.png"))
        
        # Plot change in steering angle over time
        steering_angle_changes = np.diff(steering_angles)
        plt.figure()
        plt.plot(time[1:], steering_angle_changes)
        plt.xlabel("Time (s)")
        plt.ylabel("Change in Steering Angle")
        plt.title(f"Change in Steering Angle over Time\nLap Time: {final_obs['lap_times'][0]:.2f} seconds")
        plt.savefig(os.path.join(map_dir, "steering_angle_change.png"))
        
        # Plot change in speed over time
        speed_changes = np.diff(speeds)
        plt.figure()
        plt.plot(time[1:], speed_changes)
        plt.xlabel("Time (s)")
        plt.ylabel("Change in Speed")
        plt.title(f"Change in Speed over Time\nLap Time: {final_obs['lap_times'][0]:.2f} seconds")
        plt.savefig(os.path.join(map_dir, "speed_change.png"))
        
        # Plot steering angle against speed
        plt.figure()
        plt.plot(steering_angles, speeds, 'o')
        plt.xlabel("Steering Angle")
        plt.ylabel("Speed")
        plt.title(f"Steering Angle vs. Speed\nLap Time: {final_obs['lap_times'][0]:.2f} seconds")
        plt.savefig(os.path.join(map_dir, "steering_angle_vs_speed.png"))
        
        print(f"Data plots saved to {map_dir}")
