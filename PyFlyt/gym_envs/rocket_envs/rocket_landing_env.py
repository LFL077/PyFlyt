"""Rocket Landing Environment."""
from __future__ import annotations

import os

import numpy as np
import pybullet as p
from gymnasium.spaces import Box

from .rocket_base_env import RocketBaseEnv


class RocketLandingEnv(RocketBaseEnv):
    """Rocket Landing Environment.

    Actions are finlet_x, finlet_y, finlet_roll, booster ignition, throttle, booster gimbal x, booster gimbal y
    The goal is to land the rocket on the landing pad.

    Args:
        sparse_reward (bool): whether to use sparse rewards or not.
        ceiling (float): the absolute ceiling of the flying area.
        max_displacement (float): the maximum horizontal distance the rocket can go.
        max_duration_seconds (float): maximum simulation time of the environment.
        angle_representation (str): can be "euler" or "quaternion".
        agent_hz (int): looprate of the agent to environment interaction..
        render_mode (None | str): can be "human" or None.
        render_resolution (tuple[int, int]): render_resolution.
    """

    def __init__(
        self,
        sparse_reward: bool = False,
        ceiling: float = 500.0,
        max_displacement: float = 200.0,
        max_duration_seconds: float = 30.0,
        angle_representation: str = "quaternion",
        agent_hz: int = 40,
        render_mode: None | str = None,
        render_resolution: tuple[int, int] = (480, 480),
    ):
        """__init__.

        Args:
            sparse_reward (bool): whether to use sparse rewards or not.
            ceiling (float): the absolute ceiling of the flying area.
            max_displacement (float): the maximum horizontal distance the rocket can go.
            max_duration_seconds (float): maximum simulation time of the environment.
            angle_representation (str): can be "euler" or "quaternion".
            agent_hz (int): looprate of the agent to environment interaction..
            render_mode (None | str): can be "human" or None.
            render_resolution (tuple[int, int]): render_resolution.
        """
        super().__init__(
            start_pos=np.array([[0.0, 0.0, ceiling * 0.9]]),
            start_orn=np.array([[0.0, 0.0, 0.0]]),
            ceiling=ceiling,
            max_displacement=max_displacement,
            max_duration_seconds=max_duration_seconds,
            angle_representation=angle_representation,
            agent_hz=agent_hz,
            render_mode=render_mode,
            render_resolution=render_resolution,
        )

        """GYMNASIUM STUFF"""
        # the space is the standard space + pad touch indicator + relative pad location
        max_offset = max(ceiling, max_displacement)
        low = self.combined_space.low
        low = np.concatenate(
            (low, np.array([0.0, -max_offset, -max_offset, -max_offset]))
        )
        high = self.combined_space.high
        high = np.concatenate(
            (high, np.array([1.0, max_offset, max_offset, max_offset]))
        )
        self.observation_space = Box(low=low, high=high, dtype=np.float64)

        # the landing pad
        file_dir = os.path.dirname(os.path.realpath(__file__))
        self.targ_obj_dir = os.path.join(file_dir, "../../models/landing_pad.urdf")

        """CONSTANTS"""
        self.sparse_reward = sparse_reward

    def reset(self, seed=None, options=dict()):
        """Resets the environment.

        Args:
            seed: int
            options: None
        """
        
        '''
        Those are here for the optimisation, we may not need the fitness 
        in the envirionment if using success as fitness
        '''
        self.fitness = 0.0 # fitness function to be used for learning rewards
        self.reward_options = options # included here for reward function optimisation
        
        options = dict(randomize_drop=True, accelerate_drop=True)
        drone_options = dict(starting_fuel_ratio=0.01)

        super().begin_reset(seed, options, drone_options)

        # reset the tracked parameters
        self.landing_pad_contact = 0.0
        self.ang_vel = np.zeros((3,))
        self.lin_vel = np.zeros((3,))
        self.distance = np.zeros((3,))
        self.previous_ang_vel = np.zeros((3,))
        self.previous_lin_vel = np.zeros((3,))
        self.previous_distance = np.zeros((3,))

        # randomly generate the target landing location
        theta = self.np_random.uniform(0.0, 2.0 * np.pi)
        distance = self.np_random.uniform(0.0, 0.05 * self.ceiling)
        self.landing_pad_position = (
            np.array([np.cos(theta), np.sin(theta), 0.1]) * distance
        )
        self.landing_pad_id = self.env.loadURDF(
            self.targ_obj_dir,
            basePosition=self.landing_pad_position,
            useFixedBase=True,
        )

        super().end_reset(seed, options)

        return self.state, self.info

    def compute_state(self):
        """Computes the state of the current timestep.

        This returns the observation.
        - ang_vel (vector of 3 values)
        - ang_pos (vector of 3/4 values)
        - lin_vel (vector of 3 values)
        - lin_pos (vector of 3 values)
        - previous_action (vector of 7 values)
        - auxiliary information (vector of 9 values)
        - landing_pad_contact (vector of 1 value)
        - rotated_distance to landing_pad_position (vector of 3 values)
        """
        # update the previous values to current values
        self.previous_ang_vel = self.ang_vel.copy()
        self.previous_lin_vel = self.lin_vel.copy()
        self.previous_distance = self.distance.copy()

        # update current values
        (
            self.ang_vel,
            self.ang_pos,
            self.lin_vel,
            lin_pos,
            quarternion,
        ) = super().compute_attitude()
        aux_state = super().compute_auxiliary()

        # drone to landing pad
        rotation = np.array(p.getMatrixFromQuaternion(quarternion)).reshape(3, 3)
        self.distance = lin_pos - self.landing_pad_position
        rotated_distance = np.matmul(self.distance, rotation)

        # combine everything
        if self.angle_representation == 0:
            self.state = np.array(
                [
                    *self.ang_vel,
                    *self.ang_pos,
                    *self.lin_vel,
                    *lin_pos,
                    *self.action,
                    *aux_state,
                    self.landing_pad_contact,
                    *rotated_distance,
                ]
            )
        elif self.angle_representation == 1:
            self.state = np.array(
                [
                    *self.ang_vel,
                    *quarternion,
                    *self.lin_vel,
                    *lin_pos,
                    *self.action,
                    *aux_state,
                    self.landing_pad_contact,
                    *rotated_distance,
                ]
            )

    def compute_term_trunc_reward(self):
        """Computes the termination, truncation, and reward of the current timestep."""
        super().compute_base_term_trunc_reward(
            collision_ignore_mask=[self.env.drones[0].Id, self.landing_pad_id]
        )

        # compute reward
        if not self.sparse_reward:
            # progress and distance to pad
            progress_to_pad = float(  # noqa
                np.linalg.norm(self.previous_distance[:2])
                - np.linalg.norm(self.distance[:2])
            )
            offset_to_pad = np.linalg.norm(self.distance[:2]) + 0.1  # noqa

            # deceleration as long as we're still falling
            deceleration_bonus = (  # noqa
                max(
                    (self.lin_vel[-1] < 0.0)
                    * (self.lin_vel[-1] - self.previous_lin_vel[-1]),
                    0.0,
                )
                / self.distance[-1]
            )
            
            '''
            # composite reward together
            
            # self.reward += (
            #     -5.0  # negative offset to discourage staying in the air
            #     + (2.0 / offset_to_pad)  # encourage being near the pad
            #     + (100.0 * progress_to_pad)  # encourage progress to landing pad
            #     -(1.0 * abs(self.ang_vel[-1]))  # minimize spinning
            #     - (1.0 * np.linalg.norm(self.ang_pos[:2]))  # penalize aggressive angles
            #     # + (5.0 * deceleration_bonus)  # reward deceleration when near pad
            # )
            
            # -5.0  # negative offset to discourage staying in the air
            # + (2.0 / offset_to_pad)  # encourage being near the pad
            # + (100.0 * progress_to_pad)  # encourage progress to landing pad
            # -(1.0 * abs(self.ang_vel[-1]))  # minimize spinning
            # - (1.0 * np.linalg.norm(self.ang_pos[:2]))  # penalize aggressive angles
            # # + (5.0 * deceleration_bonus)  # reward deceleration when near pad
            
            # LfL: 18.06.2023
            # variable reward function for optimisation
            # "winning" conditions:
            #     np.linalg.norm(self.previous_ang_vel) < 0.02
            #     and np.linalg.norm(self.previous_lin_vel) < 0.02
            #     and np.linalg.norm(self.ang_pos[:2]) < 0.1
            # # print(self.reward_options)
            
            # angular_velocity = np.linalg.norm(self.ang_vel)
            # angular_position = np.linalg.norm(self.ang_pos)
            # distance_to_pad = np.linalg.norm(self.distance)
            # linear_velocity = np.linalg.norm(self.lin_vel)
            
            # self.reward += (
            #     - self.reward_options[0] # negative offset to discourage staying in the air
            #     + (self.reward_options[1] / offset_to_pad)  # encourage being near the pad
            #     + (self.reward_options[2] * progress_to_pad)  # encourage progress to landing pad
            #     - (self.reward_options[3] * abs(self.ang_vel[-1]))  # minimize spinning
            #     - (self.reward_options[4] * np.linalg.norm(self.ang_pos[:2]))  # penalize aggressive angles
            #     - (self.reward_options[5] * angular_velocity)  # not spinning
            #     - (self.reward_options[6] * angular_position)  # and upright
            #     - (self.reward_options[7] * linear_velocity)   # basically stopped
            #     - (self.reward_options[8] * distance_to_pad)   # we want to be at the pad
            #     )
            '''
            
            # LfL: 25.06.2023
            # variable reward function 2 for optimisation: including all states
 
            ang_vel = self.state[:3]
            ang_pos = self.state[3:7]
            lin_vel = self.state[7:10]
            lin_pos = self.state[10:13]
            # action = self.state[13:20] # we don't need that for the rewards
            aux_state = self.state[20:29]
            # landing_pad_contact_obs = self.state[29:30]
            # rotated_distance = self.state[30:]
            
            
            angular_velocity = np.linalg.norm(ang_vel)
            angular_position = np.linalg.norm(ang_pos[:2])
            linear_velocity = np.linalg.norm(lin_vel)
            distance_to_pad = np.linalg.norm(lin_pos)
            lift_surface_0 = np.linalg.norm(aux_state[0])
            lift_surface_1 = np.linalg.norm(aux_state[1])
            lift_surface_2 = np.linalg.norm(aux_state[2])
            lift_surface_3 = np.linalg.norm(aux_state[3])
            ignition_state = np.linalg.norm(aux_state[4])
            remaining_fuel = np.linalg.norm(aux_state[5])
            current_throttle = np.linalg.norm(aux_state[6])
            gimbal_state_0 = np.linalg.norm(aux_state[7])
            gimbal_state_1 = np.linalg.norm(aux_state[8])
            # landing_pad_contact = np.linalg.norm(landing_pad_contact_obs) # also don't need this, already included
            # distance_to_pad_rotated = np.linalg.norm(rotated_distance) # and this one repeats the other

            angular_velocity = np.linalg.norm(self.ang_vel)
            angular_position = np.linalg.norm(self.ang_pos)
            distance_to_pad = np.linalg.norm(self.distance)
            linear_velocity = np.linalg.norm(self.lin_vel)
            
            self.reward += (
                - self.reward_options[0] 
                - (self.reward_options[1] * angular_velocity) 
                - (self.reward_options[2] * angular_position) 
                - (self.reward_options[3] * linear_velocity) 
                - (self.reward_options[4] * distance_to_pad) 
                - (self.reward_options[5] * lift_surface_0) 
                - (self.reward_options[6] * lift_surface_1) 
                - (self.reward_options[7] * lift_surface_2) 
                - (self.reward_options[8] * lift_surface_3) 
                - (self.reward_options[9] * ignition_state) 
                + (self.reward_options[10] * remaining_fuel) 
                - (self.reward_options[11] * current_throttle) 
                - (self.reward_options[12] * gimbal_state_0) 
                - (self.reward_options[13] * gimbal_state_1) 
                )

            # LfL: 28.06.2023
            # Adding the original function again    
            self.reward += (
                -self.reward_options[14] # negative offset to discourage staying in the air
                + (self.reward_options[15] / offset_to_pad)  # encourage being near the pad
                + (self.reward_options[16] * progress_to_pad)  # encourage progress to landing pad
                -(self.reward_options[17] * abs(self.ang_vel[-1]))  # minimize spinning
                - (self.reward_options[18] * np.linalg.norm(self.ang_pos[:2]))  # penalize aggressive angles
            )


        # check if we touched the landing pad
        if self.env.contact_array[self.env.drones[0].Id, self.landing_pad_id]:
            self.landing_pad_contact = 1.0
            # self.reward += 5000 # optimisation will maximize this, so giving a high value anyway
            self.fitness += 1 # improves fitness if touching pad
        else:
            self.landing_pad_contact = 0.0
            self.reward -= 50000
            return

        # if collision has more than 0.35 rad/s angular velocity, we dead
        # truthfully, if collision has more than 0.55 m/s linear velocity, we dead
        # number taken from here:
        # https://cosmosmagazine.com/space/launch-land-repeat-reusable-rockets-explained/
        # but doing so is kinda impossible for RL, so I've lessened the requirement to 1.0
        if (
            np.linalg.norm(self.previous_ang_vel) > 0.35
            or np.linalg.norm(self.previous_lin_vel) > 1.0
        ):
            self.info["fatal_collision"] = True
            self.termination |= True
            self.reward -= 1000000
            return

        # if our both velocities are less than 0.02 m/s and we upright, we LANDED!
        if (
            np.linalg.norm(self.previous_ang_vel) < 0.02
            and np.linalg.norm(self.previous_lin_vel) < 0.02
            and np.linalg.norm(self.ang_pos[:2]) < 0.1
        ):
            # self.reward += 100000 # just giving a very high completion bonus
            self.info["env_complete"] = True
            self.termination |= True
            self.fitness += 100 # greatly improve fitness if successful landing
            return
