# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Modular strategy classes for quadcopter environment rewards, observations, and resets."""

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from isaaclab.utils.math import (
    subtract_frame_transforms,
    quat_from_euler_xyz,
    euler_xyz_from_quat,
    wrap_to_pi,
    matrix_from_quat,
)

if TYPE_CHECKING:
    from .quadcopter_env import QuadcopterEnv

D2R = np.pi / 180.0
R2D = 180.0 / np.pi


class DefaultQuadcopterStrategy:
    """Default strategy implementation for quadcopter environment."""

    def __init__(self, env: QuadcopterEnv):
        """Initialize the default strategy.

        Args:
            env: The quadcopter environment instance.
        """
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs
        self.cfg = env.cfg

        # Initialize episode sums for logging if in training mode
        if self.cfg.is_train and hasattr(env, "rew"):
            keys = [key.split("_reward_scale")[0] for key in env.rew.keys() if key != "death_cost"]
            self._episode_sums = {
                key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                for key in keys
            }

        # Initialize nominal parameters once
        self.env._K_aero[:, :2] = self.env._k_aero_xy_value
        self.env._K_aero[:, 2] = self.env._k_aero_z_value

        self.env._kp_omega[:, :2] = self.env._kp_omega_rp_value
        self.env._ki_omega[:, :2] = self.env._ki_omega_rp_value
        self.env._kd_omega[:, :2] = self.env._kd_omega_rp_value

        self.env._kp_omega[:, 2] = self.env._kp_omega_y_value
        self.env._ki_omega[:, 2] = self.env._ki_omega_y_value
        self.env._kd_omega[:, 2] = self.env._kd_omega_y_value

        self.env._tau_m[:] = self.env._tau_m_value
        self.env._thrust_to_weight[:] = self.env._twr_value

        # Only used for play/eval logging
        self._gate_pass_times = [[] for _ in range(self.num_envs)]

        # Used for stuck penalty
        self._stuck_counter = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

    # ------------------------------------------------------------------
    # Domain randomization helpers
    # ------------------------------------------------------------------
    def _randomize_twr(self, env_ids: torch.Tensor):
        """Randomize thrust-to-weight ratio.

        PDF range:
            twr in [0.95, 1.05] * nominal_twr
        """
        if env_ids is None or len(env_ids) == 0:
            return

        twr_scale = torch.empty(len(env_ids), device=self.device).uniform_(0.95, 1.05)
        self.env._thrust_to_weight[env_ids] = self.env._twr_value * twr_scale

    def _randomize_aero(self, env_ids: torch.Tensor):
        """Randomize aerodynamic coefficients.

        PDF range:
            k_aero_xy in [0.5, 2.0] * nominal_k_aero_xy
            k_aero_z  in [0.5, 2.0] * nominal_k_aero_z
        """
        if env_ids is None or len(env_ids) == 0:
            return

        kxy_scale = torch.empty(len(env_ids), device=self.device).uniform_(0.5, 2.0)
        kz_scale = torch.empty(len(env_ids), device=self.device).uniform_(0.5, 2.0)

        self.env._K_aero[env_ids, 0] = self.env._k_aero_xy_value * kxy_scale
        self.env._K_aero[env_ids, 1] = self.env._k_aero_xy_value * kxy_scale
        self.env._K_aero[env_ids, 2] = self.env._k_aero_z_value * kz_scale

    def _randomize_pid(self, env_ids: torch.Tensor):
        """Randomize PID gains.

        PDF ranges:
            kp_omega_rp in [0.85, 1.15] * nominal
            ki_omega_rp in [0.85, 1.15] * nominal
            kd_omega_rp in [0.7,  1.3 ] * nominal

            kp_omega_y  in [0.85, 1.15] * nominal
            ki_omega_y  in [0.85, 1.15] * nominal
            kd_omega_y  in [0.7,  1.3 ] * nominal
        """
        if env_ids is None or len(env_ids) == 0:
            return

        kp_rp_scale = torch.empty(len(env_ids), device=self.device).uniform_(0.85, 1.15)
        ki_rp_scale = torch.empty(len(env_ids), device=self.device).uniform_(0.85, 1.15)
        kd_rp_scale = torch.empty(len(env_ids), device=self.device).uniform_(0.7, 1.3)

        kp_y_scale = torch.empty(len(env_ids), device=self.device).uniform_(0.85, 1.15)
        ki_y_scale = torch.empty(len(env_ids), device=self.device).uniform_(0.85, 1.15)
        kd_y_scale = torch.empty(len(env_ids), device=self.device).uniform_(0.7, 1.3)

        # roll/pitch
        self.env._kp_omega[env_ids, 0] = self.env._kp_omega_rp_value * kp_rp_scale
        self.env._kp_omega[env_ids, 1] = self.env._kp_omega_rp_value * kp_rp_scale

        self.env._ki_omega[env_ids, 0] = self.env._ki_omega_rp_value * ki_rp_scale
        self.env._ki_omega[env_ids, 1] = self.env._ki_omega_rp_value * ki_rp_scale

        self.env._kd_omega[env_ids, 0] = self.env._kd_omega_rp_value * kd_rp_scale
        self.env._kd_omega[env_ids, 1] = self.env._kd_omega_rp_value * kd_rp_scale

        # yaw
        self.env._kp_omega[env_ids, 2] = self.env._kp_omega_y_value * kp_y_scale
        self.env._ki_omega[env_ids, 2] = self.env._ki_omega_y_value * ki_y_scale
        self.env._kd_omega[env_ids, 2] = self.env._kd_omega_y_value * kd_y_scale

    def _set_nominal_params(self, env_ids: torch.Tensor):
        """Restore nominal dynamics/controller parameters."""
        if env_ids is None or len(env_ids) == 0:
            return

        self.env._thrust_to_weight[env_ids] = self.env._twr_value

        self.env._K_aero[env_ids, 0] = self.env._k_aero_xy_value
        self.env._K_aero[env_ids, 1] = self.env._k_aero_xy_value
        self.env._K_aero[env_ids, 2] = self.env._k_aero_z_value

        self.env._kp_omega[env_ids, 0] = self.env._kp_omega_rp_value
        self.env._kp_omega[env_ids, 1] = self.env._kp_omega_rp_value
        self.env._kp_omega[env_ids, 2] = self.env._kp_omega_y_value

        self.env._ki_omega[env_ids, 0] = self.env._ki_omega_rp_value
        self.env._ki_omega[env_ids, 1] = self.env._ki_omega_rp_value
        self.env._ki_omega[env_ids, 2] = self.env._ki_omega_y_value

        self.env._kd_omega[env_ids, 0] = self.env._kd_omega_rp_value
        self.env._kd_omega[env_ids, 1] = self.env._kd_omega_rp_value
        self.env._kd_omega[env_ids, 2] = self.env._kd_omega_y_value

    def _compute_shaped_gate_distance(
        self,
        pose_gate: torch.Tensor,
        gate_half_width: float = 0.50,
        gate_half_height: float = 0.50,
        anchor_return_weight: float = 1.0,
    ) -> torch.Tensor:
        """Compute shaped distance to current gate."""
        curr_x = pose_gate[:, 0]

        direct_dist = torch.linalg.norm(pose_gate, dim=1)

        a = gate_half_width
        b = gate_half_height

        anchors = torch.tensor(
            [
                [0.0,  a,  b],
                [0.0,  a, -b],
                [0.0, -a,  b],
                [0.0, -a, -b],
                [0.0,  a, 0.0],
                [0.0, -a, 0.0],
                [0.0, 0.0,  b],
                [0.0, 0.0, -b],
            ],
            device=pose_gate.device,
            dtype=pose_gate.dtype,
        )

        drone_to_anchor = torch.linalg.norm(
            pose_gate.unsqueeze(1) - anchors.unsqueeze(0),
            dim=2,
        )

        anchor_to_center = torch.linalg.norm(anchors, dim=1)
        detour_candidates = drone_to_anchor + anchor_return_weight * anchor_to_center.unsqueeze(0)
        detour_dist = torch.min(detour_candidates, dim=1).values

        shaped_dist = direct_dist.clone()
        shaped_dist[curr_x < 0.0] = detour_dist[curr_x < 0.0]

        return shaped_dist

    def get_rewards(self) -> torch.Tensor:
        """Racing reward."""
        gate_half_width = 0.50
        gate_half_height = 0.50

        # ------------------------------------------------------------------
        # Update pose of drone w.r.t. current target gate
        # ------------------------------------------------------------------
        self.env._pose_drone_wrt_gate, _ = subtract_frame_transforms(
            self.env._waypoints[self.env._idx_wp, :3],
            self.env._waypoints_quat[self.env._idx_wp, :],
            self.env._robot.data.root_link_state_w[:, :3],
        )

        pose_gate = self.env._pose_drone_wrt_gate
        curr_x = pose_gate[:, 0]
        curr_y = pose_gate[:, 1]
        curr_z = pose_gate[:, 2]

        # ------------------------------------------------------------------
        # Dense progress reward based on ORIGINAL Euclidean distance
        # ------------------------------------------------------------------
        curr_dist_to_gate = torch.linalg.norm(pose_gate, dim=1)
        progress = self.env._last_distance_to_goal - curr_dist_to_gate

        _ = self._compute_shaped_gate_distance(
            pose_gate,
            gate_half_width=gate_half_width,
            gate_half_height=gate_half_height,
        )

        # ------------------------------------------------------------------
        # Velocity in gate frame
        # ------------------------------------------------------------------
        drone_lin_vel_w = self.env._robot.data.root_com_lin_vel_w
        gate_rot_mat = matrix_from_quat(self.env._waypoints_quat[self.env._idx_wp])

        drone_lin_vel_gate = torch.bmm(
            gate_rot_mat.transpose(1, 2), drone_lin_vel_w.unsqueeze(-1)
        ).squeeze(-1)

        vel_forward = -drone_lin_vel_gate[:, 0]

        # ------------------------------------------------------------------
        # Gate traversal logic
        # Match controller_simple_policy.py:
        # forward pass:
        #   norm(pose_gate) < gate_side
        #   -0.1 < x < 0
        #   |y| < gate_side / 2
        #   |z| < gate_side / 2
        #
        # reverse pass (mirrored):
        #   norm(pose_gate) < gate_side
        #   0 < x < 0.1
        #   |y| < gate_side / 2
        #   |z| < gate_side / 2
        # ------------------------------------------------------------------
        gate_side = 2.0 * gate_half_width

        inside_gate = (torch.abs(curr_y) < gate_half_width) & (torch.abs(curr_z) < gate_half_height)
        near_gate_center = curr_dist_to_gate < gate_side

        forward_x_band = (curr_x < 0.0) & (curr_x > -0.1)
        reverse_x_band = (curr_x > 0.0) & (curr_x < 0.1)

        gate_passed = near_gate_center & forward_x_band & inside_gate
        reverse_gate_passed = near_gate_center & reverse_x_band & inside_gate

        ids_gate_passed = torch.where(gate_passed)[0]

        if len(ids_gate_passed) > 0:
            curr_time = self.env.episode_length_buf.float() * self.env.cfg.sim.dt * self.env.cfg.decimation

            if not self.cfg.is_train:
                for env_id in ids_gate_passed.tolist():
                    gate_idx_before_advance = int(self.env._idx_wp[env_id].item())
                    next_idx = (gate_idx_before_advance + 1) % self.env._waypoints.shape[0]
                    t = float(curr_time[env_id].item())

                    drone_pos_w = self.env._robot.data.root_link_state_w[env_id, :3]
                    gate_pos_w = self.env._waypoints[gate_idx_before_advance, :3]
                    next_gate_pos_w = self.env._waypoints[next_idx, :3]

                    self._gate_pass_times[env_id].append((gate_idx_before_advance, t))

                    print(
                        f"[GATE PASS DEBUG] "
                        f"env={env_id} "
                        f"time={t:.3f}s "
                        f"passed_gate={gate_idx_before_advance} "
                        f"next_gate={next_idx} "
                        f"drone_pos_w={drone_pos_w.tolist()} "
                        f"gate_pos_w={gate_pos_w.tolist()} "
                        f"next_gate_pos_w={next_gate_pos_w.tolist()} "
                        f"curr_x={float(curr_x[env_id].item()):.4f} "
                        f"curr_y={float(curr_y[env_id].item()):.4f} "
                        f"curr_z={float(curr_z[env_id].item()):.4f} "
                        f"dist={float(curr_dist_to_gate[env_id].item()):.4f}"
                    )

            self.env._idx_wp[ids_gate_passed] = (
                self.env._idx_wp[ids_gate_passed] + 1
            ) % self.env._waypoints.shape[0]

            self.env._n_gates_passed[ids_gate_passed] += 1

            self.env._desired_pos_w[ids_gate_passed, :3] = self.env._waypoints[
                self.env._idx_wp[ids_gate_passed], :3
            ]

            new_pose_gate, _ = subtract_frame_transforms(
                self.env._waypoints[self.env._idx_wp[ids_gate_passed], :3],
                self.env._waypoints_quat[self.env._idx_wp[ids_gate_passed], :],
                self.env._robot.data.root_link_state_w[ids_gate_passed, :3],
            )

            self.env._last_distance_to_goal[ids_gate_passed] = torch.linalg.norm(new_pose_gate, dim=1)
            self.env._prev_x_drone_wrt_gate[ids_gate_passed] = new_pose_gate[:, 0]

            # Passing a gate means not stuck
            self._stuck_counter[ids_gate_passed] = 0

        not_passed = ~gate_passed
        self.env._last_distance_to_goal[not_passed] = curr_dist_to_gate[not_passed]
        self.env._prev_x_drone_wrt_gate[not_passed] = curr_x[not_passed]

        # ------------------------------------------------------------------
        # Additional geometry helpers
        # ------------------------------------------------------------------
        yz_dist = torch.sqrt(curr_y**2 + curr_z**2)
        near_gate = curr_dist_to_gate < 2.0
        roughly_centered = yz_dist < 0.5

        center_reward = (1.0 - torch.tanh(yz_dist / 0.5)) * near_gate.float()
        vel_forward_masked = vel_forward * near_gate.float() * roughly_centered.float()

        # ------------------------------------------------------------------
        # Penalty for being on exit side and moving back toward gate plane
        # ------------------------------------------------------------------
        on_wrong_side = curr_x < 0.0
        inside_gate_loose = (torch.abs(curr_y) < 0.65) & (torch.abs(curr_z) < 0.65)
        near_gate_loose = curr_dist_to_gate < 1.5

        toward_gate_speed = torch.clamp(drone_lin_vel_gate[:, 0], min=0.0)
        center_weight = torch.exp(-((yz_dist / 0.35) ** 2))

        wrong_side_toward_gate_penalty = (
            on_wrong_side.float()
            * near_gate_loose.float()
            * inside_gate_loose.float()
            * center_weight
            * toward_gate_speed
        )

        wrong_side_toward_gate_penalty = wrong_side_toward_gate_penalty * (~gate_passed).float()

        # ------------------------------------------------------------------
        # Simple stuck penalty
        # ------------------------------------------------------------------
        lin_speed = torch.linalg.norm(drone_lin_vel_w, dim=1)

        low_progress = progress < 0.002
        low_speed = lin_speed < 0.35
        not_crucially_moving = (~gate_passed)

        stuck_condition = low_progress & low_speed & not_crucially_moving
        self._stuck_counter = torch.where(
            stuck_condition,
            self._stuck_counter + 1,
            torch.clamp(self._stuck_counter - 2, min=0),
        )

        stuck_penalty = (self._stuck_counter > 15).float()

        # ------------------------------------------------------------------
        # Crash penalty
        # ------------------------------------------------------------------
        contact_forces = self.env._contact_sensor.data.net_forces_w
        crashed = (torch.norm(contact_forces, dim=-1) > 1e-8).squeeze(1).float()

        mask = (self.env.episode_length_buf > 10).float()
        self.env._crashed += crashed.int() * mask.int()

        # ------------------------------------------------------------------
        # Small time penalty
        # ------------------------------------------------------------------
        time_penalty = torch.ones(self.num_envs, device=self.device)

        if self.cfg.is_train:
            rewards = {
                "progress_goal": progress * self.env.rew["progress_goal_reward_scale"],
                "gate_pass": gate_passed.float() * self.env.rew["gate_pass_reward_scale"],
                "crash": crashed * self.env.rew["crash_reward_scale"],
                "time_penalty": time_penalty * self.env.rew["time_penalty_reward_scale"],
            }

            if "reverse_gate_penalty_reward_scale" in self.env.rew:
                rewards["reverse_gate_penalty"] = (
                    reverse_gate_passed.float() * self.env.rew["reverse_gate_penalty_reward_scale"]
                )

            if "vel_forward_reward_scale" in self.env.rew:
                rewards["vel_forward"] = vel_forward_masked * self.env.rew["vel_forward_reward_scale"]

            if "center_gate_reward_scale" in self.env.rew:
                rewards["center_gate"] = center_reward * self.env.rew["center_gate_reward_scale"]

            if "wrong_side_toward_gate_penalty_reward_scale" in self.env.rew:
                rewards["wrong_side_toward_gate_penalty"] = (
                    wrong_side_toward_gate_penalty
                    * self.env.rew["wrong_side_toward_gate_penalty_reward_scale"]
                )

            if "stuck_penalty_reward_scale" in self.env.rew:
                rewards["stuck_penalty"] = (
                    stuck_penalty * self.env.rew["stuck_penalty_reward_scale"]
                )

            reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

            reward = torch.where(
                self.env.reset_terminated,
                torch.ones_like(reward) * self.env.rew["death_cost"],
                reward,
            )

            for key, value in rewards.items():
                if key not in self._episode_sums:
                    self._episode_sums[key] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
                self._episode_sums[key] += value
        else:
            reward = torch.zeros(self.num_envs, device=self.device)

        return reward

    def get_observations(self) -> Dict[str, torch.Tensor]:
        """Get observations including waypoint positions and drone state."""
        curr_idx = self.env._idx_wp % self.env._waypoints.shape[0]
        next_idx = (self.env._idx_wp + 1) % self.env._waypoints.shape[0]

        wp_curr_pos = self.env._waypoints[curr_idx, :3]
        wp_next_pos = self.env._waypoints[next_idx, :3]
        quat_curr = self.env._waypoints_quat[curr_idx]
        quat_next = self.env._waypoints_quat[next_idx]

        rot_curr = matrix_from_quat(quat_curr)
        rot_next = matrix_from_quat(quat_next)

        verts_curr = (
            torch.bmm(self.env._local_square, rot_curr.transpose(1, 2))
            + wp_curr_pos.unsqueeze(1)
            + self.env._terrain.env_origins.unsqueeze(1)
        )
        verts_next = (
            torch.bmm(self.env._local_square, rot_next.transpose(1, 2))
            + wp_next_pos.unsqueeze(1)
            + self.env._terrain.env_origins.unsqueeze(1)
        )

        waypoint_pos_b_curr, _ = subtract_frame_transforms(
            self.env._robot.data.root_link_state_w[:, :3].repeat_interleave(4, dim=0),
            self.env._robot.data.root_link_state_w[:, 3:7].repeat_interleave(4, dim=0),
            verts_curr.view(-1, 3),
        )
        waypoint_pos_b_next, _ = subtract_frame_transforms(
            self.env._robot.data.root_link_state_w[:, :3].repeat_interleave(4, dim=0),
            self.env._robot.data.root_link_state_w[:, 3:7].repeat_interleave(4, dim=0),
            verts_next.view(-1, 3),
        )

        waypoint_pos_b_curr = waypoint_pos_b_curr.view(self.num_envs, 4, 3)
        waypoint_pos_b_next = waypoint_pos_b_next.view(self.num_envs, 4, 3)

        quat_w = self.env._robot.data.root_quat_w
        attitude_mat = matrix_from_quat(quat_w)

        obs = torch.cat(
            [
                self.env._robot.data.root_com_lin_vel_b,                         # 3 dim
                attitude_mat.view(attitude_mat.shape[0], -1),                   # 9 dim
                waypoint_pos_b_curr.view(waypoint_pos_b_curr.shape[0], -1),     # 12 dim
                waypoint_pos_b_next.view(waypoint_pos_b_next.shape[0], -1),     # 12 dim
            ],
            dim=-1,
        )
        observations = {"policy": obs}

        # Update yaw tracking
        rpy = euler_xyz_from_quat(quat_w)
        yaw_w = wrap_to_pi(rpy[2])

        delta_yaw = yaw_w - self.env._previous_yaw
        self.env._previous_yaw = yaw_w
        self.env._yaw_n_laps += torch.where(delta_yaw < -np.pi, 1, 0)
        self.env._yaw_n_laps -= torch.where(delta_yaw > np.pi, 1, 0)

        self.env.unwrapped_yaw = yaw_w + 2 * np.pi * self.env._yaw_n_laps

        self.env._previous_actions = self.env._actions.clone()

        return observations

    def reset_idx(self, env_ids: Optional[torch.Tensor]):
        """Reset specific environments to randomized racing start states."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.env._robot._ALL_INDICES

        if self.cfg.is_train and hasattr(self, "_episode_sums"):
            extras = dict()
            for key in self._episode_sums.keys():
                episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
                extras["Episode_Reward/" + key] = episodic_sum_avg / self.env.max_episode_length_s
                self._episode_sums[key][env_ids] = 0.0

            self.env.extras["log"] = dict()
            self.env.extras["log"].update(extras)

            extras = dict()
            extras["Episode_Termination/died"] = torch.count_nonzero(self.env.reset_terminated[env_ids]).item()
            extras["Episode_Termination/time_out"] = torch.count_nonzero(self.env.reset_time_outs[env_ids]).item()
            self.env.extras["log"].update(extras)
        else:
            for env_id in env_ids.tolist():
                if len(self._gate_pass_times[env_id]) > 0:
                    print(f"[EPISODE SUMMARY] env {env_id} gate pass times: {self._gate_pass_times[env_id]}")

        self.env._robot.reset(env_ids)

        if not self.env._models_paths_initialized:
            num_models_per_env = self.env._waypoints.size(0)
            model_prim_names_in_env = [
                f"{self.env.target_models_prim_base_name}_{i}" for i in range(num_models_per_env)
            ]

            self.env._all_target_models_paths = []
            for env_path in self.env.scene.env_prim_paths:
                paths_for_this_env = [f"{env_path}/{name}" for name in model_prim_names_in_env]
                self.env._all_target_models_paths.append(paths_for_this_env)

            self.env._models_paths_initialized = True

        n_reset = len(env_ids)
        if n_reset == self.num_envs and self.num_envs > 1:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf,
                high=int(self.env.max_episode_length),
            )

        self.env._actions[env_ids] = 0.0
        self.env._previous_actions[env_ids] = 0.0
        self.env._previous_yaw[env_ids] = 0.0
        self.env._motor_speeds[env_ids] = 0.0
        self.env._previous_omega_meas[env_ids] = 0.0
        self.env._previous_omega_err[env_ids] = 0.0
        self.env._omega_err_integral[env_ids] = 0.0
        self._stuck_counter[env_ids] = 0

        if self.cfg.is_train:
            self._randomize_twr(env_ids)
            self._randomize_aero(env_ids)
            self._randomize_pid(env_ids)
        else:
            self._set_nominal_params(env_ids)

        for env_id in env_ids.tolist():
            self._gate_pass_times[env_id] = []

        joint_pos = self.env._robot.data.default_joint_pos[env_ids]
        joint_vel = self.env._robot.data.default_joint_vel[env_ids]
        self.env._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        default_root_state = self.env._robot.data.default_root_state[env_ids].clone()

        if self.cfg.is_train:
            waypoint_indices = torch.zeros(n_reset, device=self.device, dtype=self.env._idx_wp.dtype)

            use_gate23_curriculum = torch.zeros(n_reset, dtype=torch.bool, device=self.device)

            if self.env._waypoints.shape[0] >= 4:
                use_gate23_curriculum = torch.rand(n_reset, device=self.device) < 0.30
                waypoint_indices[use_gate23_curriculum] = 3

            regular_ids_local = torch.where(~use_gate23_curriculum)[0]
            if len(regular_ids_local) > 0:
                wp0 = waypoint_indices[regular_ids_local]
                x0_wp = self.env._waypoints[wp0, 0]
                y0_wp = self.env._waypoints[wp0, 1]
                theta = self.env._waypoints[wp0, -1]

                x_local = torch.empty(len(regular_ids_local), device=self.device).uniform_(-3.0, -0.5)
                y_local = torch.empty(len(regular_ids_local), device=self.device).uniform_(-1.0, 1.0)

                cos_theta = torch.cos(theta)
                sin_theta = torch.sin(theta)

                x_rot = cos_theta * x_local - sin_theta * y_local
                y_rot = sin_theta * x_local + cos_theta * y_local

                initial_x = x0_wp - x_rot
                initial_y = y0_wp - y_rot
                initial_z = torch.empty(len(regular_ids_local), device=self.device).uniform_(0.12, 0.18)

                default_root_state[regular_ids_local, 0] = initial_x
                default_root_state[regular_ids_local, 1] = initial_y
                default_root_state[regular_ids_local, 2] = initial_z

                initial_yaw = torch.atan2(y0_wp - initial_y, x0_wp - initial_x)
                yaw_noise = torch.empty(len(regular_ids_local), device=self.device).uniform_(-0.25, 0.25)

                quat = quat_from_euler_xyz(
                    torch.zeros(len(regular_ids_local), device=self.device),
                    torch.zeros(len(regular_ids_local), device=self.device),
                    initial_yaw + yaw_noise,
                )
                default_root_state[regular_ids_local, 3:7] = quat

                default_root_state[regular_ids_local, 7:10] = 0.15 * torch.randn(
                    (len(regular_ids_local), 3), device=self.device
                )
                default_root_state[regular_ids_local, 10:13] = 0.05 * torch.randn(
                    (len(regular_ids_local), 3), device=self.device
                )

            gate23_ids_local = torch.where(use_gate23_curriculum)[0]
            if len(gate23_ids_local) > 0:
                gate2_idx = torch.full(
                    (len(gate23_ids_local),), 2, dtype=self.env._idx_wp.dtype, device=self.device
                )
                gate3_idx = torch.full(
                    (len(gate23_ids_local),), 3, dtype=self.env._idx_wp.dtype, device=self.device
                )

                gate2_pos = self.env._waypoints[gate2_idx, :3]
                gate2_yaw = self.env._waypoints[gate2_idx, -1]
                gate3_pos = self.env._waypoints[gate3_idx, :3]

                x_local = torch.empty(len(gate23_ids_local), device=self.device).uniform_(0.7, 1.3)
                y_local = torch.empty(len(gate23_ids_local), device=self.device).uniform_(-0.25, 0.25)
                z_local = torch.empty(len(gate23_ids_local), device=self.device).uniform_(0.35, 0.85)

                cos_theta = torch.cos(gate2_yaw)
                sin_theta = torch.sin(gate2_yaw)

                x_rot = cos_theta * x_local - sin_theta * y_local
                y_rot = sin_theta * x_local + cos_theta * y_local

                start_x = gate2_pos[:, 0] - x_rot
                start_y = gate2_pos[:, 1] - y_rot
                start_z = gate2_pos[:, 2] + z_local

                default_root_state[gate23_ids_local, 0] = start_x
                default_root_state[gate23_ids_local, 1] = start_y
                default_root_state[gate23_ids_local, 2] = start_z

                yaw_to_gate3 = torch.atan2(
                    gate3_pos[:, 1] - start_y,
                    gate3_pos[:, 0] - start_x,
                )
                yaw_noise = torch.empty(len(gate23_ids_local), device=self.device).uniform_(-0.20, 0.20)

                quat = quat_from_euler_xyz(
                    torch.zeros(len(gate23_ids_local), device=self.device),
                    torch.zeros(len(gate23_ids_local), device=self.device),
                    yaw_to_gate3 + yaw_noise,
                )
                default_root_state[gate23_ids_local, 3:7] = quat

                default_root_state[gate23_ids_local, 7:10] = 0.10 * torch.randn(
                    (len(gate23_ids_local), 3), device=self.device
                )
                default_root_state[gate23_ids_local, 10:13] = 0.05 * torch.randn(
                    (len(gate23_ids_local), 3), device=self.device
                )

        else:
            x_local = torch.empty(1, device=self.device).uniform_(-3.0, -0.5)
            y_local = torch.empty(1, device=self.device).uniform_(-1.0, 1.0)

            x0_wp = self.env._waypoints[self.env._initial_wp, 0]
            y0_wp = self.env._waypoints[self.env._initial_wp, 1]
            theta = self.env._waypoints[self.env._initial_wp, -1]

            cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
            x_rot = cos_theta * x_local - sin_theta * y_local
            y_rot = sin_theta * x_local + cos_theta * y_local

            x0 = x0_wp - x_rot
            y0 = y0_wp - y_rot

            z0 = torch.empty(1, device=self.device).uniform_(0.12, 0.18)

            yaw0 = torch.atan2(y0_wp - y0, x0_wp - x0)

            default_root_state = self.env._robot.data.default_root_state[0].unsqueeze(0).clone()
            default_root_state[:, 0] = x0
            default_root_state[:, 1] = y0
            default_root_state[:, 2] = z0

            quat = quat_from_euler_xyz(
                torch.zeros(1, device=self.device),
                torch.zeros(1, device=self.device),
                yaw0,
            )
            default_root_state[:, 3:7] = quat
            default_root_state[:, 7:13] = 0.0

            waypoint_indices = self.env._initial_wp

        self.env._idx_wp[env_ids] = waypoint_indices
        self.env._desired_pos_w[env_ids, :3] = self.env._waypoints[waypoint_indices, :3].clone()

        self.env._n_gates_passed[env_ids] = 0
        self.env._yaw_n_laps[env_ids] = 0
        self.env._crashed[env_ids] = 0

        self.env._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)
        self.env._robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)

        self.env._pose_drone_wrt_gate[env_ids], _ = subtract_frame_transforms(
            self.env._waypoints[self.env._idx_wp[env_ids], :3],
            self.env._waypoints_quat[self.env._idx_wp[env_ids], :],
            default_root_state[:, :3],
        )

        self.env._last_distance_to_goal[env_ids] = torch.linalg.norm(
            self.env._pose_drone_wrt_gate[env_ids], dim=1
        )
        self.env._prev_x_drone_wrt_gate[env_ids] = self.env._pose_drone_wrt_gate[env_ids, 0]