# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

import math

import isaaclab.sim as sim_utils
import isaacsim.core.utils.torch as torch_utils  # type: ignore
import torch
from isaaclab.assets.articulation import Articulation
from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.managers import EventManager, RewardManager
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.sensors.camera import TiledCamera
from isaaclab.sim import PhysxCfg, SimulationContext
from isaaclab.utils.buffers import CircularBuffer, DelayBuffer
from isaaclab.utils.math import quat_apply, quat_conjugate
from scipy.spatial.transform import Rotation

from legged_lab.envs.elf3_tang_noarm.walk_cfg import Elf3TangWalkNoarmFlatEnvCfg
from legged_lab.utils.env_utils.scene import SceneCfg
from rsl_rl.env import VecEnv
from rsl_rl.utils import AMPLoaderDisplay


HAND_OFFSET_FROM_ELBOW = 0.3
ROOT_HEIGHT_OFFSET_FOR_DISPLAY = 0.3
TERMINATION_CONTACT_FORCE_THRESHOLD = 500.0
ARM_JOINT_NAMES = [
    "l_shoulder_y_joint",
    "l_shoulder_x_joint",
    "l_shoulder_z_joint",
    "l_elbow_y_joint",
    "l_wrist_x_joint",
    "l_wrist_y_joint",
    "l_wrist_z_joint",
    "r_shoulder_y_joint",
    "r_shoulder_x_joint",
    "r_shoulder_z_joint",
    "r_elbow_y_joint",
    "r_wrist_x_joint",
    "r_wrist_y_joint",
    "r_wrist_z_joint",
]


class Elf3TangNoarmEnv(VecEnv):
    def __init__(self, cfg: Elf3TangWalkNoarmFlatEnvCfg, headless: bool):
        self.cfg = cfg
        self.headless = headless
        self.device = self.cfg.device
        self.physics_dt = self.cfg.sim.dt
        self.step_dt = self.cfg.sim.decimation * self.cfg.sim.dt
        self.num_envs = self.cfg.scene.num_envs
        self.seed(cfg.scene.seed)

        self.sim = self._create_sim_context()
        scene_cfg = SceneCfg(config=cfg.scene, physics_dt=self.physics_dt, step_dt=self.step_dt)
        self.scene = InteractiveScene(scene_cfg)
        self.sim.reset()

        self.robot: Articulation = self.scene["robot"]
        self.contact_sensor: ContactSensor = self.scene.sensors["contact_sensor"]
        self._init_optional_sensors()

        self.command_generator = self._create_command_generator()
        self.reward_manager = RewardManager(self.cfg.reward, self)
        print(self.reward_manager)

        self.init_buffers()
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.event_manager = EventManager(self.cfg.domain_rand.events, self)
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")
        self.reset(env_ids)

        self.amp_loader_display = AMPLoaderDisplay(
            motion_files=self.cfg.amp_motion_files_display, device=self.device, time_between_frames=self.physics_dt
        )
        self.motion_len = self.amp_loader_display.trajectory_num_frames[0]

    def _create_sim_context(self) -> SimulationContext:
        sim_cfg = sim_utils.SimulationCfg(
            device=self.cfg.device,
            dt=self.cfg.sim.dt,
            render_interval=self.cfg.sim.decimation,
            physx=PhysxCfg(gpu_max_rigid_patch_count=self.cfg.sim.physx.gpu_max_rigid_patch_count),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
        )
        return SimulationContext(sim_cfg)

    def _init_optional_sensors(self) -> None:
        if self.cfg.scene.height_scanner.enable_height_scan:
            self.height_scanner: RayCaster = self.scene.sensors["height_scanner"]
        if self.cfg.scene.lidar.enable_lidar:
            self.lidar: RayCaster = self.scene.sensors["lidar"]
        if self.cfg.scene.depth_camera.enable_depth_camera:
            self.depth_camera: TiledCamera = self.scene.sensors["depth_camera"]

    def _create_command_generator(self) -> UniformVelocityCommand:
        command_cfg = UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=self.cfg.commands.resampling_time_range,
            rel_standing_envs=self.cfg.commands.rel_standing_envs,
            rel_heading_envs=self.cfg.commands.rel_heading_envs,
            heading_command=self.cfg.commands.heading_command,
            heading_control_stiffness=self.cfg.commands.heading_control_stiffness,
            debug_vis=self.cfg.commands.debug_vis,
            ranges=self.cfg.commands.ranges,
        )
        return UniformVelocityCommand(cfg=command_cfg, env=self)

    def init_buffers(self):
        self.extras = {}

        self.max_episode_length_s = self.cfg.scene.max_episode_length_s
        self.max_episode_length = math.ceil(self.max_episode_length_s / self.step_dt)
        self.num_dofs = self.robot.data.default_joint_pos.shape[1]
        self.num_actions = self.num_dofs
        self.clip_actions = self.cfg.normalization.clip_actions
        self.clip_obs = self.cfg.normalization.clip_observations

        # 支持标量或逐关节列表的action_scale
        if isinstance(self.cfg.robot.action_scale, (list, tuple)):
            self.action_scale = torch.tensor(self.cfg.robot.action_scale, dtype=torch.float, device=self.device)
        else:
            self.action_scale = self.cfg.robot.action_scale

        self._resolve_scene_entities()
        self._resolve_robot_indices()
        self._init_action_buffer()
        self._init_arm_teleop_buffers()

        self.obs_scales = self.cfg.normalization.obs_scales
        self.add_noise = self.cfg.noise.add_noise

        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.sim_step_counter = 0
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        self.left_arm_local_vec = self._make_repeated_vector([0.0, 0.0, -HAND_OFFSET_FROM_ELBOW])
        self.right_arm_local_vec = self._make_repeated_vector([0.0, 0.0, -HAND_OFFSET_FROM_ELBOW])
        self._init_gait_buffers()
        self.action = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.avg_feet_force_per_step = self._new_feet_metric_buffer()
        self.avg_feet_speed_per_step = self._new_feet_metric_buffer()
        self.init_obs_buffer()

    def _resolve_scene_entities(self) -> None:
        self.robot_cfg = SceneEntityCfg(name="robot")
        self.robot_cfg.resolve(self.scene)

        self.termination_contact_cfg = SceneEntityCfg(
            name="contact_sensor", body_names=self.cfg.robot.terminate_contacts_body_names
        )
        self.termination_contact_cfg.resolve(self.scene)

        self.feet_cfg = SceneEntityCfg(name="contact_sensor", body_names=self.cfg.robot.feet_body_names)
        self.feet_cfg.resolve(self.scene)

    def _resolve_robot_indices(self) -> None:
        self.waist_ids = self._find_joints(["waist_y_joint", "waist_x_joint", "waist_z_joint"])
        self.left_wrist_ids = self._find_joints(["l_wrist_x_joint", "l_wrist_y_joint", "l_wrist_z_joint"])
        self.right_wrist_ids = self._find_joints(["r_wrist_x_joint", "r_wrist_y_joint", "r_wrist_z_joint"])

        self.feet_body_ids, _ = self.robot.find_bodies(
            name_keys=["l_ankle_x_link", "r_ankle_x_link"], preserve_order=True
        )
        self.elbow_body_ids, _ = self.robot.find_bodies(
            name_keys=["l_elbow_y_link", "r_elbow_y_link"], preserve_order=True
        )

        self.left_leg_ids = self._find_joints(
            ["l_hip_y_joint", "l_hip_x_joint", "l_hip_z_joint", "l_knee_y_joint", "l_ankle_y_joint", "l_ankle_x_joint"]
        )
        self.right_leg_ids = self._find_joints(
            ["r_hip_y_joint", "r_hip_x_joint", "r_hip_z_joint", "r_knee_y_joint", "r_ankle_y_joint", "r_ankle_x_joint"]
        )
        self.left_arm_ids = self._find_joints(
            ["l_shoulder_y_joint", "l_shoulder_x_joint", "l_shoulder_z_joint", "l_elbow_y_joint"]
        )
        self.right_arm_ids = self._find_joints(
            ["r_shoulder_y_joint", "r_shoulder_x_joint", "r_shoulder_z_joint", "r_elbow_y_joint"]
        )
        self.all_arms_ids = self._find_joints(ARM_JOINT_NAMES)
        self.num_arm_actions = len(self.all_arms_ids)
        self.ankle_joint_ids = self._find_joints(
            ["l_ankle_y_joint", "r_ankle_y_joint", "l_ankle_x_joint", "r_ankle_x_joint"]
        )

        arm_mask = torch.zeros(self.num_dofs, dtype=torch.bool, device=self.device)
        arm_mask[self.all_arms_ids] = True
        self.policy_joint_ids = torch.arange(self.num_dofs, device=self.device)[~arm_mask]
        self.num_policy_actions = len(self.policy_joint_ids)
        self.num_actions = self.num_policy_actions

    def _find_joints(self, name_keys: list[str]) -> list[int]:
        joint_ids, _ = self.robot.find_joints(name_keys=name_keys, preserve_order=True)
        return joint_ids

    def _init_action_buffer(self) -> None:
        action_delay_cfg = self.cfg.domain_rand.action_delay
        self.action_buffer = DelayBuffer(action_delay_cfg.params["max_delay"], self.num_envs, device=self.device)
        self.action_buffer.compute(
            torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        )

        if action_delay_cfg.enable:
            time_lags = torch.randint(
                low=action_delay_cfg.params["min_delay"],
                high=action_delay_cfg.params["max_delay"] + 1,
                size=(self.num_envs,),
                dtype=torch.int,
                device=self.device,
            )
            self.action_buffer.set_time_lag(time_lags, torch.arange(self.num_envs, device=self.device))

    def _init_arm_teleop_buffers(self) -> None:
        target_ranges = self.cfg.arm_teleop.target_ranges
        self.arm_target_low = torch.tensor(
            [target_ranges[name][0] for name in ARM_JOINT_NAMES], dtype=torch.float, device=self.device
        )
        self.arm_target_high = torch.tensor(
            [target_ranges[name][1] for name in ARM_JOINT_NAMES], dtype=torch.float, device=self.device
        )

        default_arm_pos = self.robot.data.default_joint_pos[:, self.all_arms_ids].clone()
        self.arm_command = default_arm_pos.clone()
        self.arm_target = default_arm_pos.clone()
        self.arm_max_delta = self._sample_arm_max_delta(self.num_envs)
        self.arm_resample_steps = self._sample_arm_resample_steps(self.num_envs)
        self.arm_elapsed_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def _init_gait_buffers(self) -> None:
        self.gait_phase = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.gait_cycle = torch.full(
            (self.num_envs,), self.cfg.gait.gait_cycle, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.phase_ratio = self._make_repeated_vector(
            [self.cfg.gait.gait_air_ratio_l, self.cfg.gait.gait_air_ratio_r]
        )
        self.phase_offset = self._make_repeated_vector(
            [self.cfg.gait.gait_phase_offset_l, self.cfg.gait.gait_phase_offset_r]
        )

    def _make_repeated_vector(self, values: list[float]) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.float, device=self.device).repeat(self.num_envs, 1)

    def _new_feet_metric_buffer(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, len(self.feet_cfg.body_ids), dtype=torch.float, device=self.device)

    def visualize_motion(self, time: float) -> torch.Tensor:
        """Write an AMP motion frame into the simulator and return its AMP observation."""
        visual_motion_frame = self.amp_loader_display.get_full_frame_at_time(0, time)
        dof_pos, dof_vel = self._motion_frame_to_dof_state(visual_motion_frame)

        self.robot.write_joint_position_to_sim(dof_pos)
        self.robot.write_joint_velocity_to_sim(dof_vel)
        self.robot.write_root_state_to_sim(
            self._motion_frame_to_root_state(visual_motion_frame),
            torch.arange(self.num_envs, device=self.device),
        )
        self.sim.render()
        self.sim.step()
        self.scene.update(dt=self.step_dt)

        return self._build_amp_obs(dof_pos, dof_vel)

    def _motion_frame_to_dof_state(self, motion_frame: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dof_pos = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)
        dof_vel = torch.zeros_like(dof_pos)

        dof_pos[:, self.waist_ids] = motion_frame[6:9]
        dof_pos[:, self.left_leg_ids] = motion_frame[9:15]
        dof_pos[:, self.right_leg_ids] = motion_frame[15:21]
        dof_pos[:, self.left_arm_ids] = motion_frame[21:25]
        dof_pos[:, self.left_wrist_ids] = motion_frame[25:28]
        dof_pos[:, self.right_arm_ids] = motion_frame[28:32]
        dof_pos[:, self.right_wrist_ids] = motion_frame[32:35]

        dof_vel[:, self.waist_ids] = motion_frame[41:44]
        dof_vel[:, self.left_leg_ids] = motion_frame[44:50]
        dof_vel[:, self.right_leg_ids] = motion_frame[50:56]
        dof_vel[:, self.left_arm_ids] = motion_frame[56:60]
        dof_vel[:, self.left_wrist_ids] = motion_frame[60:63]
        dof_vel[:, self.right_arm_ids] = motion_frame[63:67]
        dof_vel[:, self.right_wrist_ids] = motion_frame[67:70]

        return dof_pos, dof_vel

    def _motion_frame_to_root_state(self, motion_frame: torch.Tensor) -> torch.Tensor:
        root_pos = motion_frame[:3].clone()
        root_pos[2] += ROOT_HEIGHT_OFFSET_FOR_DISPLAY

        quat_xyzw = Rotation.from_euler("XYZ", motion_frame[3:6].cpu().numpy(), degrees=False).as_quat()
        quat_wxyz = torch.tensor(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=torch.float32, device=self.device
        )

        root_state = torch.zeros((self.num_envs, 13), device=self.device)
        root_state[:, 0:3] = root_pos.unsqueeze(0).repeat(self.num_envs, 1)
        root_state[:, 3:7] = quat_wxyz.unsqueeze(0).repeat(self.num_envs, 1)
        root_state[:, 7:10] = motion_frame[35:38].unsqueeze(0).repeat(self.num_envs, 1)
        return root_state

    def _build_amp_obs(self, dof_pos: torch.Tensor, dof_vel: torch.Tensor) -> torch.Tensor:
        left_hand_pos, right_hand_pos, left_foot_pos, right_foot_pos = self._relative_limb_positions()
        return torch.cat(
            (
                dof_pos[:, self.waist_ids],
                dof_pos[:, self.right_leg_ids],
                dof_pos[:, self.left_leg_ids],
                dof_vel[:, self.waist_ids],
                dof_vel[:, self.right_leg_ids],
                dof_vel[:, self.left_leg_ids],
                left_hand_pos,
                right_hand_pos,
                left_foot_pos,
                right_foot_pos,
            ),
            dim=-1,
        )

    def _relative_limb_positions(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        left_hand_pos = (
            self.robot.data.body_state_w[:, self.elbow_body_ids[0], :3]
            - self.robot.data.root_state_w[:, 0:3]
            + quat_apply(self.robot.data.body_state_w[:, self.elbow_body_ids[0], 3:7], self.left_arm_local_vec)
        )
        right_hand_pos = (
            self.robot.data.body_state_w[:, self.elbow_body_ids[1], :3]
            - self.robot.data.root_state_w[:, 0:3]
            + quat_apply(self.robot.data.body_state_w[:, self.elbow_body_ids[1], 3:7], self.right_arm_local_vec)
        )
        left_hand_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), left_hand_pos)
        right_hand_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), right_hand_pos)
        left_foot_pos = (
            self.robot.data.body_state_w[:, self.feet_body_ids[0], :3] - self.robot.data.root_state_w[:, 0:3]
        )
        right_foot_pos = (
            self.robot.data.body_state_w[:, self.feet_body_ids[1], :3] - self.robot.data.root_state_w[:, 0:3]
        )
        left_foot_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), left_foot_pos)
        right_foot_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), right_foot_pos)
        return left_hand_pos, right_hand_pos, left_foot_pos, right_foot_pos

    def compute_current_observations(self):
        robot = self.robot
        net_contact_forces = self.contact_sensor.data.net_forces_w_history

        ang_vel = robot.data.root_ang_vel_b
        projected_gravity = robot.data.projected_gravity_b
        command = self.command_generator.command
        joint_pos = robot.data.joint_pos - robot.data.default_joint_pos
        joint_vel = robot.data.joint_vel - robot.data.default_joint_vel
        action = self.action_buffer._circular_buffer.buffer[:, -1, :]  # last executed action, not latest command
        root_lin_vel = robot.data.root_lin_vel_b
        # privileged: critic-only; actor cannot measure ground truth velocity on real hardware
        feet_contact = torch.max(torch.norm(net_contact_forces[:, :, self.feet_cfg.body_ids], dim=-1), dim=1)[0] > 0.5
        policy_joint_pos = joint_pos[:, self.policy_joint_ids]
        policy_joint_vel = joint_vel[:, self.policy_joint_ids]
        policy_action = action
        current_actor_obs = torch.cat(
            [
                ang_vel * self.obs_scales.ang_vel,  # 3
                projected_gravity * self.obs_scales.projected_gravity,  # 3
                command * self.obs_scales.commands,  # 3
                policy_joint_pos * self.obs_scales.joint_pos,  # non-arm joints
                policy_joint_vel * self.obs_scales.joint_vel,  # non-arm joints
                policy_action * self.obs_scales.actions,  # non-arm previous action
            ],
            dim=-1,
        )
        # critic appends privileged info (lin_vel + feet_contact) not available on real robot
        current_critic_obs = torch.cat(
            [current_actor_obs, root_lin_vel * self.obs_scales.lin_vel, feet_contact], dim=-1
        )

        return current_actor_obs, current_critic_obs

    def compute_observations(self):
        current_actor_obs, current_critic_obs = self.compute_current_observations()
        if self.add_noise:
            current_actor_obs += (2 * torch.rand_like(current_actor_obs) - 1) * self.noise_scale_vec

        self.actor_obs_buffer.append(current_actor_obs)
        self.critic_obs_buffer.append(current_critic_obs)

        actor_obs = self.actor_obs_buffer.buffer.reshape(self.num_envs, -1)
        critic_obs = self.critic_obs_buffer.buffer.reshape(self.num_envs, -1)
        if self.cfg.scene.height_scanner.enable_height_scan:
            actor_obs, critic_obs = self._append_height_scan(actor_obs, critic_obs)

        if self.cfg.scene.depth_camera.enable_depth_camera:
            actor_obs, critic_obs = self._append_depth_image(actor_obs, critic_obs)

        actor_obs = torch.clip(actor_obs, -self.clip_obs, self.clip_obs)
        critic_obs = torch.clip(critic_obs, -self.clip_obs, self.clip_obs)

        return actor_obs, critic_obs

    def _append_height_scan(
        self, actor_obs: torch.Tensor, critic_obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        height_scan = (
            self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
            - self.height_scanner.data.ray_hits_w[..., 2]
            - self.cfg.normalization.height_scan_offset
        ) * self.obs_scales.height_scan
        critic_obs = torch.cat([critic_obs, height_scan], dim=-1)
        if self.add_noise:
            height_scan += (2 * torch.rand_like(height_scan) - 1) * self.height_scan_noise_vec
        actor_obs = torch.cat([actor_obs, height_scan], dim=-1)
        return actor_obs, critic_obs

    def _append_depth_image(
        self, actor_obs: torch.Tensor, critic_obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        depth_image = self.depth_camera.data.output["distance_to_image_plane"]
        flattened_depth = depth_image.view(self.num_envs, -1)
        actor_obs = torch.cat([actor_obs, flattened_depth], dim=-1)
        critic_obs = torch.cat([critic_obs, flattened_depth], dim=-1)
        return actor_obs, critic_obs

    def reset(self, env_ids):
        if len(env_ids) == 0:
            return

        # Reset buffer
        self.avg_feet_force_per_step[env_ids] = 0.0
        self.avg_feet_speed_per_step[env_ids] = 0.0

        self.extras["log"] = dict()
        if self.cfg.scene.terrain_generator is not None:
            if self.cfg.scene.terrain_generator.curriculum:
                terrain_levels = self.update_terrain_levels(env_ids)
                self.extras["log"].update(terrain_levels)

        self.scene.reset(env_ids)
        if "reset" in self.event_manager.available_modes:
            self.event_manager.apply(
                mode="reset",
                env_ids=env_ids,
                dt=self.step_dt,
                global_env_step_count=self.sim_step_counter // self.cfg.sim.decimation,
            )
        self._reset_arm_teleop(env_ids)

        reward_extras = self.reward_manager.reset(env_ids)
        self.extras["log"].update(reward_extras)
        self.extras["time_outs"] = self.time_out_buf

        self.command_generator.reset(env_ids)
        self.actor_obs_buffer.reset(env_ids)
        self.critic_obs_buffer.reset(env_ids)
        self.action_buffer.reset(env_ids)
        self.episode_length_buf[env_ids] = 0
        self.scene.write_data_to_sim()
        self.sim.forward()

    def _reset_arm_teleop(self, env_ids: torch.Tensor) -> None:
        arm_pos = self.robot.data.default_joint_pos[env_ids][:, self.all_arms_ids]
        arm_vel = torch.zeros_like(arm_pos)
        self.arm_command[env_ids] = arm_pos
        self.arm_target[env_ids] = arm_pos
        self.arm_max_delta[env_ids] = self._sample_arm_max_delta(len(env_ids))
        self.arm_resample_steps[env_ids] = self._sample_arm_resample_steps(len(env_ids))
        self.arm_elapsed_steps[env_ids] = 0
        self.robot.write_joint_state_to_sim(arm_pos, arm_vel, joint_ids=self.all_arms_ids, env_ids=env_ids)
        self.robot.set_joint_position_target(arm_pos, joint_ids=self.all_arms_ids, env_ids=env_ids)

    def _sample_arm_max_delta(self, count: int) -> torch.Tensor:
        low, high = self.cfg.arm_teleop.max_delta_per_step_range
        return torch.empty((count, 1), dtype=torch.float, device=self.device).uniform_(low, high)

    def _sample_arm_resample_steps(self, count: int) -> torch.Tensor:
        low_s, high_s = self.cfg.arm_teleop.resampling_time_range
        low_steps = max(1, math.ceil(low_s / self.step_dt))
        high_steps = max(low_steps, math.ceil(high_s / self.step_dt))
        return torch.randint(low_steps, high_steps + 1, (count,), dtype=torch.long, device=self.device)

    def _update_arm_teleop_command(self) -> torch.Tensor:
        if not self.cfg.arm_teleop.enable:
            return self.robot.data.default_joint_pos[:, self.all_arms_ids]

        self.arm_elapsed_steps += 1
        resample_env_ids = (self.arm_elapsed_steps >= self.arm_resample_steps).nonzero(as_tuple=False).flatten()
        self._resample_arm_teleop_targets(resample_env_ids)

        delta = self.arm_target - self.arm_command
        self.arm_command += torch.clamp(delta, -self.arm_max_delta, self.arm_max_delta)
        return self.arm_command

    def _resample_arm_teleop_targets(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return

        random_unit = torch.rand((len(env_ids), self.num_arm_actions), dtype=torch.float, device=self.device)
        self.arm_target[env_ids] = self.arm_target_low + random_unit * (self.arm_target_high - self.arm_target_low)
        self.arm_max_delta[env_ids] = self._sample_arm_max_delta(len(env_ids))
        self.arm_resample_steps[env_ids] = self._sample_arm_resample_steps(len(env_ids))
        self.arm_elapsed_steps[env_ids] = 0

    def step(self, actions: torch.Tensor):
        self.action = self._build_full_action(actions)
        self._simulate_action(self.action * self.action_scale + self.robot.data.default_joint_pos)
        if not self.headless:
            self.sim.render()

        self.episode_length_buf += 1
        self._calculate_gait_para()

        self.command_generator.compute(self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.reset_buf, self.time_out_buf = self.check_reset()
        reward_buf = self.reward_manager.compute(self.step_dt)
        self.reset_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset(self.reset_env_ids)

        actor_obs, critic_obs = self.compute_observations()
        self.extras["observations"] = {"critic": critic_obs}

        return actor_obs, reward_buf, self.reset_buf, self.extras

    def _build_full_action(self, actions: torch.Tensor) -> torch.Tensor:
        delayed_actions = self.action_buffer.compute(actions)
        policy_actions = torch.clip(delayed_actions, -self.clip_actions, self.clip_actions).to(self.device)

        full_actions = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        full_actions[:, self.policy_joint_ids] = policy_actions
        full_actions[:, self.all_arms_ids] = self._arm_position_to_action(self._update_arm_teleop_command())
        return full_actions

    def _arm_position_to_action(self, arm_pos: torch.Tensor) -> torch.Tensor:
        default_arm_pos = self.robot.data.default_joint_pos[:, self.all_arms_ids]
        if isinstance(self.action_scale, torch.Tensor):
            return (arm_pos - default_arm_pos) / self.action_scale[self.all_arms_ids]
        return (arm_pos - default_arm_pos) / self.action_scale

    def _simulate_action(self, processed_actions: torch.Tensor) -> None:
        self.avg_feet_force_per_step.zero_()
        self.avg_feet_speed_per_step.zero_()

        for _ in range(self.cfg.sim.decimation):
            self.sim_step_counter += 1
            self.robot.set_joint_position_target(processed_actions)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)
            self._accumulate_feet_metrics()

        self.avg_feet_force_per_step /= self.cfg.sim.decimation
        self.avg_feet_speed_per_step /= self.cfg.sim.decimation

    def _accumulate_feet_metrics(self) -> None:
        self.avg_feet_force_per_step += torch.norm(
            self.contact_sensor.data.net_forces_w[:, self.feet_cfg.body_ids, :3], dim=-1
        )
        self.avg_feet_speed_per_step += torch.norm(self.robot.data.body_lin_vel_w[:, self.feet_body_ids, :], dim=-1)

    def check_reset(self):
        net_contact_forces = self.contact_sensor.data.net_forces_w_history

        reset_buf = torch.any(
            torch.max(
                torch.norm(
                    net_contact_forces[:, :, self.termination_contact_cfg.body_ids],
                    dim=-1,
                ),
                dim=1,
            )[0]
            > TERMINATION_CONTACT_FORCE_THRESHOLD,
            dim=1,
        )
        time_out_buf = self.episode_length_buf >= self.max_episode_length
        reset_buf |= time_out_buf
        return reset_buf, time_out_buf

    def init_obs_buffer(self):
        if self.add_noise:
            actor_obs, _ = self.compute_current_observations()
            noise_vec = torch.zeros_like(actor_obs[0])
            noise_scales = self.cfg.noise.noise_scales

            obs_idx = 0
            noise_vec[obs_idx : obs_idx + 3] = noise_scales.ang_vel * self.obs_scales.ang_vel
            obs_idx += 3
            noise_vec[obs_idx : obs_idx + 3] = noise_scales.projected_gravity * self.obs_scales.projected_gravity
            obs_idx += 3
            noise_vec[obs_idx : obs_idx + 3] = 0.0  # commands no noise
            obs_idx += 3
            noise_vec[obs_idx : obs_idx + self.num_policy_actions] = noise_scales.joint_pos * self.obs_scales.joint_pos
            obs_idx += self.num_policy_actions
            noise_vec[obs_idx : obs_idx + self.num_policy_actions] = noise_scales.joint_vel * self.obs_scales.joint_vel
            obs_idx += self.num_policy_actions
            noise_vec[obs_idx : obs_idx + self.num_policy_actions] = 0.0  # previous actions no noise

            self.noise_scale_vec = noise_vec

            if self.cfg.scene.height_scanner.enable_height_scan:
                height_scan = (
                    self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
                    - self.height_scanner.data.ray_hits_w[..., 2]
                    - self.cfg.normalization.height_scan_offset
                )
                height_scan_noise_vec = torch.zeros_like(height_scan[0])
                height_scan_noise_vec[:] = noise_scales.height_scan * self.obs_scales.height_scan
                self.height_scan_noise_vec = height_scan_noise_vec

        self.actor_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.actor_obs_history_length, batch_size=self.num_envs, device=self.device
        )
        self.critic_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.critic_obs_history_length, batch_size=self.num_envs, device=self.device
        )

    def update_terrain_levels(self, env_ids):
        distance = torch.norm(self.robot.data.root_pos_w[env_ids, :2] - self.scene.env_origins[env_ids, :2], dim=1)
        move_up = distance > self.scene.terrain.cfg.terrain_generator.size[0] / 2
        move_down = (
            distance < torch.norm(self.command_generator.command[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5
        )
        move_down *= ~move_up  # prevent simultaneous up/down to avoid oscillation at boundary
        self.scene.terrain.update_env_origins(env_ids, move_up, move_down)
        extras = {}
        extras["Curriculum/terrain_levels"] = torch.mean(self.scene.terrain.terrain_levels.float())
        return extras

    def get_observations(self):
        actor_obs, critic_obs = self.compute_observations()
        self.extras["observations"] = {"critic": critic_obs}
        return actor_obs, self.extras

    def get_amp_motion_mask(self, lin_threshold: float = 0.1, yaw_threshold: float = 0.1) -> torch.Tensor:
        """Return 1.0 for moving commands and 0.0 for near-standing commands."""
        cmd = self.command_generator.command
        is_moving = (torch.norm(cmd[:, :2], dim=-1) >= lin_threshold) | (torch.abs(cmd[:, 2]) >= yaw_threshold)
        return is_moving.float()

    def get_amp_obs_for_expert_trans(self):
        """Return the current AMP observation used by the discriminator."""
        return self._build_amp_obs(self.robot.data.joint_pos, self.robot.data.joint_vel)

    @staticmethod
    def seed(seed: int = -1) -> int:
        try:
            import omni.replicator.core as rep  # type: ignore

            rep.set_global_seed(seed)
        except ModuleNotFoundError:
            pass
        return torch_utils.set_seed(seed)

    def _calculate_gait_para(self) -> None:
        # phase is episode-local time so it resets cleanly at episode boundary (not wall-clock time)
        t = self.episode_length_buf * self.step_dt / self.gait_cycle
        self.gait_phase[:, 0] = (t + self.phase_offset[:, 0]) % 1.0
        self.gait_phase[:, 1] = (t + self.phase_offset[:, 1]) % 1.0
