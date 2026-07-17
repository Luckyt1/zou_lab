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
import torch

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (  # noqa:F401
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlRndCfg,
    RslRlSymmetryCfg,
)

import legged_lab.mdp as mdp
from legged_lab.assets.elf3_lite import ELF3LITE_CFG
from legged_lab.envs.base.base_config import (
    ActionDelayCfg,
    BaseSceneCfg,
    CommandRangesCfg,
    CommandsCfg,
    DomainRandCfg,
    EventCfg,
    HeightScannerCfg,
    NoiseCfg,
    NoiseScalesCfg,
    NormalizationCfg,
    ObsScalesCfg,
    PhysxCfg,
    RobotCfg,
    SimCfg,
)
from legged_lab.terrains import GRAVEL_TERRAINS_CFG, ROUGH_TERRAINS_CFG  # noqa:F401


@configclass
class GaitCfg:
    gait_air_ratio_l: float = 0.38
    gait_air_ratio_r: float = 0.38
    gait_phase_offset_l: float = 0.38
    gait_phase_offset_r: float = 0.88
    gait_cycle: float = 0.85


@configclass
class ArmMotionCurriculumCfg:
    # 手臂随机目标课程从第几个 PPO iteration 开始计时。
    start_iteration: int = 0

    # 课程持续的仿真 rollout 时间；期间手臂随机幅值从 initial_scale 线性增长到 1.0。
    curriculum_duration_s: float = 4800.0

    # 训练刚开始时使用的幅值比例，避免一开始手臂扰动太大导致站不稳。
    initial_scale: float = 0.02

    # 命令速度阈值；cmd_norm 小于该值时更偏向 stand_amplitudes，大于该值时更偏向 move_amplitudes。
    command_threshold: float = 0.6

    # 每隔多少秒重新采样一次手臂目标姿态，目标之间会平滑插值。
    # 远手臂动作如果切换太快，会通过躯干反作用力把腿部顶起来。
    resample_interval_s: float = 0.1

    # 零速度站立时手臂/手腕关节的随机幅值上限，单位 rad。
    # 顺序为：
    # l_shoulder_y, l_shoulder_x, l_shoulder_z, l_elbow_y, l_wrist_x, l_wrist_y, l_wrist_z,
    # r_shoulder_y, r_shoulder_x, r_shoulder_z, r_elbow_y, r_wrist_x, r_wrist_y, r_wrist_z。
    stand_amplitudes: list = [
        0.8,
        0.3,
        0.4,
        0.5,
        0.6,
        0.4,
        0.3,
        0.8,
        0.3,
        0.4,
        0.5,
        0.6,
        0.4,
        0.3,
    ]
    # stand_amplitudes: list = [
    #     0.18,
    #     0.10,
    #     0.18,
    #     0.10,
    #     0.06,
    #     0.06,
    #     0.06,
    #     0.18,
    #     0.10,
    #     0.18,
    #     0.10,
    #     0.06,
    #     0.06,
    #     0.06,
    # ]
    move_amplitudes: list = [
        0.3,
        0.2,
        0.2,
        0.2,
        0.3,
        0.3,
        0.3,
        0.3,
        0.2,
        0.2,
        0.2,
        0.3,
        0.3,
        0.3,
    ]


@configclass
class ResetVelocityCurriculumCfg:
    # reset 初始速度扰动课程持续时间；用于逐步增加每次重置时 base 的随机初速度。
    curriculum_duration_s: float = 4800.0

    # 训练刚开始时的扰动比例；0.0 表示一开始 reset 后 base 初速度完全为 0。
    initial_scale: float = 0.0

    # 课程结束后的扰动比例；1.0 表示使用下面 velocity_ranges 的完整范围。
    final_scale: float = 1.0

    # reset 时给 root state 采样的初始速度范围，单位：
    # x/y/z 为 m/s，roll/pitch/yaw 为 rad/s。
    # 实际范围会乘以当前课程 scale。
    velocity_ranges: dict = {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
        "z": (-0.5, 0.5),
        "roll": (-0.5, 0.5),
        "pitch": (-0.5, 0.5),
        "yaw": (-0.5, 0.5),
    }


@configclass
class PushVelocityCurriculumCfg:
    # 训练过程中外部推扰动课程持续时间；用于逐步增加 interval push 的强度。
    curriculum_duration_s: float = 4800.0

    # 训练刚开始时的推扰动比例；0.05 表示先用完整推力范围的 5%。
    initial_scale: float = 0.05

    # 课程结束后的推扰动比例；1.0 表示使用下面 velocity_ranges 的完整范围。
    final_scale: float = 1.0

    # push_robot 事件给 base 叠加的速度扰动范围，单位：
    # x/y 为 m/s，yaw 为 rad/s。
    # 实际范围会乘以当前课程 scale，并在 episode 中按 interval_range_s 周期触发。
    velocity_ranges: dict = {
        "x": (-0.8, 0.8),
        "y": (-0.8, 0.8),
        "yaw": (-0.2, 0.2),
    }


ELF3_TANG_STRAIGHT_JOINT_POS = dict(ELF3LITE_CFG.init_state.joint_pos)
ELF3_TANG_STRAIGHT_JOINT_POS.update(
    {
        "l_hip_y_joint": -0.30,
        "l_knee_y_joint": 0.60,
        "l_ankle_y_joint": -0.30,
        "r_hip_y_joint": -0.30,
        "r_knee_y_joint": 0.60,
        "r_ankle_y_joint": -0.30,
    }
)

ELF3_TANG_STRAIGHT_CFG = ELF3LITE_CFG.replace(
    init_state=ELF3LITE_CFG.init_state.replace(
        pos=(0.0, 0.0, 1.11),
        joint_pos=ELF3_TANG_STRAIGHT_JOINT_POS,
    )
)


# #run gait
# @configclass
# class GaitCfg:
#     gait_air_ratio_l: float = 0.6
#     gait_air_ratio_r: float = 0.6
#     gait_phase_offset_l: float = 0.6
#     gait_phase_offset_r: float = 0.1
#     gait_cycle: float = 0.5
    

@configclass
class LiteRewardCfg:
    # RewardManager 计算方式：term = func(...) * weight * step_dt。
    # weight > 0 是奖励，weight < 0 是惩罚，weight = 0 表示当前关闭。

    # 存活奖励：只要没有 reset，每步给一个小的正奖励，鼓励延长 episode。
    alive = RewTerm(func=mdp.alive, weight=0.6)

    # XY 线速度跟踪奖励：当前纯站立任务关闭。
    # track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.5, params={"std": 0.30})

    # yaw 角速度跟踪奖励：当前纯站立任务关闭。
    # track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=4.0, params={"std": 0.35})

    # 竖直速度惩罚：惩罚 base 在 Z 方向上下窜动，减少跳动/下沉。
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-8.0)

    # 横滚/俯仰角速度惩罚：惩罚 base 的 roll/pitch 角速度，减少身体晃动。
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-1.8)

    # base 高度区间惩罚：root 高度在 [1.08, 1.14] 内不罚，匹配 -0.15/0.30/-0.15 目标微曲站姿。
    # base_height_range_l2 = RewTerm(
    #     func=mdp.base_height_range_l2,
    #     weight=-6.0,
    #     params={"min_height": 1.08, "max_height": 1.14},
    # )

    # 静止命令下的 base 速度惩罚：命令接近 0 时惩罚水平漂移和 yaw 运动。
    stand_base_velocity_l2 = RewTerm(
        func=mdp.stand_base_velocity_l2,
        weight=-10.0,
        params={"command_threshold": 0.12},
    )

    # 静止 base 稳定奖励：命令接近 0 时，base 水平速度和 yaw 速度越小奖励越高。
    stand_base_stability_exp = RewTerm(
        func=mdp.stand_base_stability_exp,
        weight=4.0,
        params={"std": 0.30, "command_threshold": 0.12},
    )

    # 静止 base 大范围回正惩罚：当前由 stand_com_sway_l2 约束水平位移，关闭以减少重复扣分。
    # stand_base_displacement_l2 = RewTerm(
    #     func=mdp.stand_base_displacement_l2,
    #     weight=-1.5,
    #     params={"command_threshold": 0.12, "max_displacement": 0.7},
    # )

    # 静止重心居中奖励：root/近似重心越接近 episode 起点 XY，奖励越高。
    # stand_com_center_exp = RewTerm(
    #     func=mdp.stand_base_center_exp,
    #     weight=2.0,
    #     params={"std": 0.25, "command_threshold": 0.2},
    # )

    # 静止重心小范围晃动惩罚：更敏感地惩罚 root/近似重心相对 episode 起点的水平位移。
    stand_com_sway_l2 = RewTerm(
        func=mdp.stand_base_displacement_l2,
        weight=-2.0,
        params={"command_threshold": 0.2, "max_displacement": 0.7},
    )

    # 静止 base 朝向回正惩罚：惩罚相对 reset 初始朝向的 yaw 漂移，避免手臂扰动后机器人慢慢自转。
    stand_base_yaw_drift_l2 = RewTerm(
        func=mdp.stand_base_yaw_drift_l2,
        weight=-2.0,
        params={"command_threshold": 0.2, "max_yaw": 0.2},
    )

    # 静止命令下的双脚接触奖励：命令接近 0 时奖励脚踝接触地面，避免单脚/飞脚站立。
    stand_feet_contact = RewTerm(
        func=mdp.stand_feet_contact,
        weight=8.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_x.*"), "command_threshold": 0.12},
    )

    # 足底低载荷惩罚：静止命令下，如果某只脚竖直支撑力太小就扣分，抑制小跳/脚离地。
    feet_force_low = RewTerm(
        func=mdp.feet_force_below_threshold,
        weight=-5.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_x.*"),
            "threshold": 270.0,
            "command_threshold": 0.2,
        },
    )

    # 能耗惩罚：惩罚腰腿关节功率 |torque * velocity|，鼓励省力动作。
    energy = RewTerm(
        func=mdp.energy,
        weight=-1.5e-3,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["waist_.*_joint", ".*_hip_.*_joint", ".*_knee_.*_joint", ".*_ankle_.*_joint"],
            )
        },
    )

    # 关节加速度惩罚：惩罚腰腿关节加速度，减少突变和抖动。
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.0e-6,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["waist_.*_joint", ".*_hip_.*_joint", ".*_knee_.*_joint", ".*_ankle_.*_joint"],
            )
        },
    )

    # 腿部六关节速度惩罚：限制 hip_y/x/z、knee_y、ankle_y/x 快速来回晃动。
    leg_dof_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-4.0e-3,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_hip_y_joint",
                    ".*_hip_x_joint",
                    ".*_hip_z_joint",
                    ".*_knee_y_joint",
                    ".*_ankle_y_joint",
                    ".*_ankle_x_joint",
                ],
            )
        },
    )

    # 动作变化率惩罚：惩罚相邻两步 action 的差，降低控制抖动。
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.1)

    # 动作平滑惩罚：同时惩罚 action 一阶差分、二阶差分和动作幅值。
    action_rate_smooth = RewTerm(func=mdp.action_smoothness, weight=-0.1)

    # 腿部动作变化率惩罚：只惩罚腿部 action 的变化，进一步平滑下肢控制。
    leg_action_rate_l2 = RewTerm(func=mdp.leg_action_rate_l2, weight=-0.08)

    # 踝关节扭矩惩罚：惩罚踝关节输出力矩过大，减少踝部硬顶。
    ankle_torque = RewTerm(func=mdp.ankle_torque, weight=-0.0005)

    # 踝关节动作幅值惩罚：惩罚踝关节 action 过大，避免脚踝大幅摆动。
    ankle_action = RewTerm(func=mdp.ankle_action, weight=-0.001)

    # 髋 roll 动作幅值惩罚：限制左右 hip_x action，避免大幅侧摆补偿。
    hip_roll_action = RewTerm(func=mdp.hip_roll_action, weight=-0.2)

    # 髋 yaw 动作幅值惩罚：当前纯站立任务关闭，避免和上身/手臂扰动恢复动作冲突。
    # hip_yaw_action = RewTerm(func=mdp.hip_yaw_action, weight=-0.05)

    # 非期望接触惩罚：膝/髋/腰/躯干等非脚部接触地面时扣分。
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor", body_names=[".*_knee_y.*", ".*_hip_z.*", ".*_hip_y.*", "waist_z.*", "torso_link"]
            ),
            "threshold": 3.0,
        },
    )

    # torso 姿态惩罚：惩罚 torso 的重力投影水平分量，身体越倾斜扣分越多。
    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2, params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")}, weight=-3.2
    )

    # torso 姿态奖励：当前由 stand_body_orientation_exp 约束，关闭以减少重复姿态奖励。
    # body_orientation_euler = RewTerm(
    #     func=mdp.body_orientation_euler, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*torso.*")}, weight=1.0
    # )

    # 静止 torso 姿态奖励：命令接近 0 时奖励 torso 接近竖直。
    stand_body_orientation_exp = RewTerm(
        func=mdp.stand_body_orientation_exp,
        weight=3.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*torso.*"), "command_threshold": 0.12},
    )
    # body_pitch_l2 = RewTerm(
    #     func=mdp.body_pitch_l2, params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")}, weight=-3.0
    # )

    # base 和双脚中点的前后偏移惩罚：当前原地旋转任务中关闭，避免限制转身时的自然重心调整。
    # base_feet_x_offset_l2 = RewTerm(
    #     func=mdp.base_feet_x_offset_l2,
    #     weight=-1.0,
    #     params={"target_offset": 0.2, "command_threshold": 0.1},
    # )

    # base 水平姿态惩罚：当前由 torso 姿态项约束，关闭以减少重复姿态惩罚。
    # flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)

    # 提前终止惩罚：非超时 reset 时扣大分，主要惩罚摔倒/严重碰撞。
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # 脚底滑动惩罚：脚接触地面时，惩罚脚在水平面上的速度。
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        # weight=-0.75,
        weight=-1.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_x.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_x.*"),
        },
    )

    # # 脚部冲击力惩罚：脚踝竖直接触力超过 threshold 后扣分，限制重踩地。
    # feet_force = RewTerm(
    #     func=mdp.body_force,
    #     weight=-5e-3,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_x.*"),
    #         "threshold": 400,   
    #         # "max_reward": 400,
    #         "max_reward": 400,
    #     },
    # )

    # 双脚过近惩罚：双脚距离小于 threshold 时扣分，避免交叉脚/站距过窄。
    feet_too_near = RewTerm(
        func=mdp.feet_too_near_humanoid,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_x.*"]), "threshold": 0.27},
    )

    # 脚部绊倒惩罚：水平接触力远大于竖直力时扣分，检测脚被卡/被绊。
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*_ankle_x.*"])},
    )

    # 步宽惩罚：Y 方向脚间距偏离期望值时扣分，当前在低侧向命令下生效。
    feet_y_distance = RewTerm(func=mdp.feet_y_distance, weight=-2.0)

    # 足部姿态惩罚：脚接触地面时惩罚脚面 roll/pitch 倾斜，减少脚边角点地。
    feet_orientation_l2 = RewTerm(
        func=mdp.feet_orientation_l2,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor",
                body_names=".*ankle_x_link",
            ),
        },
    )

    # 足部 yaw 姿态奖励：当前注释关闭，不参与训练。
    # feet_orientation_euler = RewTerm(
    #     func=mdp.feet_orientation_euler, params={"asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_x_link.*")}, weight=0.25
    # )

    # 关节软限位惩罚：腰腿关节超过 soft limit 时扣分。
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["waist_.*_joint", ".*_hip_.*_joint", ".*_knee_.*_joint", ".*_ankle_.*_joint"],
            )
        },
    )
    # dof_vel_limits = RewTerm(func=mdp.joint_vel_limits, weight=-0.5, params={"soft_ratio": 0.9})

    # 髋 roll/yaw 姿态偏离惩罚：惩罚 hip_z/hip_x 偏离默认角，限制髋部大幅侧摆/扭转。
    # joint_deviation_hip = RewTerm(
    #     func=mdp.joint_deviation_l1_always,
    #     # weight=-0.15,
    #     weight=-0.25,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             # "robot", joint_names=[".*hip_z.*", ".*hip_x.*", ".*shoulder_y.*", ".*elbow_y.*"],
    #             "robot", joint_names=[".*hip_z.*", ".*hip_x.*",],
    #             # joint_names=[
    #             #     # ".*_hip_y_joint",
    #             #     ".*_hip_x_joint",
    #             #     ".*_hip_z_joint",
    #             # ],
    #         )
    #     },
    # )

    # 行走时的髋/手臂偏离惩罚：当前注释关闭，不参与训练。
    # joint_deviation_hip_walk = RewTerm(
    #     func=mdp.joint_deviation_l2,
    #     weight=-1.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot", joint_names=[".*hip_z.*", ".*hip_x.*", ".*shoulder_y.*", ".*elbow_y.*"],
    #         )
    #     },
    # )

    # #避免一侧罚重一侧轻arm/waist/wrist关节偏差
    # 手臂默认姿态偏离惩罚：当前注释关闭；若开启会和随机手臂扰动目标冲突。
    # joint_deviation_arms_0 = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     # func=mdp.joint_deviation_l1_always,
    #     # weight=-2.0,
    #     weight=-0.15,
    #     # weight=-0.5,
    #     # weight=-0.05,
    #     # weight=-0.02,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_x_joint", ".*_shoulder_z_joint"])},
    #     # params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_x_joint", ".*_shoulder_z_joint", ".*_wrist_.*_joint"])},
    #     # params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_y_joint", ".*_shoulder_x_joint", ".*_shoulder_z_joint", ".*_elbow_y_joint", ".*_wrist_.*_joint"])},
    # )

    # 另一组手臂默认姿态偏离惩罚：当前注释关闭；若开启会和随机手臂扰动目标冲突。
    # joint_deviation_arms_1 = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     # func=mdp.joint_deviation_l1_always,
    #     # weight=-2.0,
    #     weight=-0.15,
    #     # weight=-0.5,
    #     # weight=-0.02,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_y_joint", ".*_elbow_y_joint" , ".*_wrist_.*_joint"])},
    # )

    # #加入腰部关节偏差惩罚 - 覆盖所有腰部关节
    # 腰部偏离惩罚候选项：当前注释关闭，不参与训练。
    # joint_deviation_waists = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     # func=mdp.joint_deviation_l1_always,
    #     # weight=-1.0,
    #     weight=-0.15,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_x_joint"])},

    # )

    # 腰部偏离惩罚候选项：当前注释关闭，不参与训练。
    # joint_deviation_waists1 = RewTerm(
    #     func=mdp.joint_deviation_l1,
    #     weight=-0.15,
    #     # params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_.*_joint"])},  # 覆盖所有腰部关节
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_y_joint","waist_z_joint"])},
    # )

    # 腰腿默认姿态偏离惩罚：轻量约束除膝盖外的腿部关节回到默认角，减少下半身晃动。
    # joint_deviation_legs = RewTerm(
    #     func=mdp.joint_deviation_l1_always,
    #     # func=mdp.joint_deviation_l1_always,
    #     weight=-0.10,
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             joint_names=[
    #                 ".*_hip_x_joint",
    #                 ".*_hip_z_joint",
    #                 ".*_ankle_x_joint",
    #             ],
    #         )
    #     },
    # )

    # 静止命令下的腿部 pitch 目标奖励：鼓励膝/髋/踝接近 0.40/-0.20/-0.20。
    hip_pitch_target_exp = RewTerm(
        func=mdp.stand_joint_target_exp,
        weight=1.0,
        params={
            "std": 0.12,
            "target_angle": -0.20,
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_y_joint"]),
        },
    )
    ankle_pitch_target_exp = RewTerm(
        func=mdp.stand_joint_target_exp,
        weight=1.0,
        params={
            "std": 0.12,
            "target_angle": -0.20,
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_y_joint"]),
        },
    )
    knee_target_exp = RewTerm(
        func=mdp.stand_joint_target_exp,
        weight=1.5,
        params={
            "std": 0.15,
            "target_angle": 0.40,
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_knee_y_joint"]),
        },
    )

    # knee_bend_l2 = RewTerm(
    #     func=mdp.knee_bend_l2,
    #     weight=-18.0,
    #     params={"target_angle": 0.40, "allowed_bend": 0.05, "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_knee_y_joint"])},
    # )

    # 对应的静止偏离惩罚：离 0.40/-0.20/-0.20 越远，惩罚越大。
    knee_target_l1 = RewTerm(
        func=mdp.stand_joint_target_l1,
        weight=-2.5,
        params={
            "target_angle": 0.40,
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_knee_y_joint"]),
        },
    )
    hip_pitch_target_l1 = RewTerm(
        func=mdp.stand_joint_target_l1,
        weight=-2.0,
        params={
            "target_angle": -0.20,
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_y_joint"]),
        },
    )
    ankle_pitch_target_l1 = RewTerm(
        func=mdp.stand_joint_target_l1,
        weight=-2.0,
        params={
            "target_angle": -0.20,
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_y_joint"]),
        },
    )

    # 腰 roll/pitch 默认姿态偏离惩罚：限制 waist_x/waist_y 大幅补偿。
    joint_deviation_waist = RewTerm(
        func=mdp.joint_deviation_l1_always,
        weight=-0.25,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_x_joint", "waist_y_joint"])},
    )

    # 腰 yaw 默认姿态偏离惩罚：限制 waist_z 左右扭动，减少腰 z 轴晃动。
    joint_deviation_waist_yaw = RewTerm(
        func=mdp.joint_deviation_l1_always,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["waist_z_joint"])},
    )

    # 以下步态周期项当前权重为 0，多用于行走任务；当前静止站立任务中关闭。
    # gait_feet_frc_perio = RewTerm(func=mdp.gait_feet_frc_perio, weight=2.0, params={"delta_t": 0.02})# 增加权重,鼓励抬脚用力周期
    # gait_feet_spd_perio = RewTerm(func=mdp.gait_feet_spd_perio, weight=2.0, params={"delta_t": 0.02})
    # gait_feet_frc_support_perio = RewTerm(func=mdp.gait_feet_frc_support_perio, weight=1.2, params={"delta_t": 0.02})

    # gait_feet_frc_perio = RewTerm(func=mdp.gait_feet_frc_perio_smooth, weight=2.0, params={"delta_t": 0.015})
    # 摆动相脚接触力周期奖励：当前权重为 0，不参与训练。
    # gait_feet_frc_perio = RewTerm(func=mdp.gait_feet_frc_perio_smooth, weight=0.0, params={"delta_t": 0.015})

    # 支撑相脚速度周期奖励：当前权重为 0，不参与训练。
    # gait_feet_spd_perio = RewTerm(func=mdp.gait_feet_spd_perio_smooth, weight=0.0, params={"delta_t": 0.015})

    # 摆动相脚接触力惩罚：当前权重为 0，不参与训练。
    # gait_feet_frc_perio_penalize = RewTerm(func=mdp.gait_feet_frc_perio_penalize, weight=0.0, params={"delta_t": 0.015})

    # 步态相位和足部接触匹配奖励：当前权重为 0，不参与训练。
    # gait_phase_contact = RewTerm(
    #     func=mdp.gait_phase_contact_smooth,
    #     weight=0.0,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["l_ankle_x.*", "r_ankle_x.*"]),
    #         "delta_t": 0.03,
    #         "command_threshold": 0.1,
    #     },
    # )

    # 摆动脚高度惩罚：当前权重为 0，不参与训练。
    # gait_swing_height = RewTerm(
    #     func=mdp.gait_swing_height_l2,
    #     weight=0.0,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=["l_ankle_x_link", "r_ankle_x_link"]),
    #         "target_height": 0.06,
    #         "delta_t": 0.03,
    #         "command_threshold": 0.1,
    #     },
    # )

    # 双脚离地惩罚：检测脚踝都没有接触地面时扣分，避免飞起。
    fly = RewTerm(
        func=mdp.fly,
        weight=-18.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_x_link.*"), "threshold": 1.0},
    )

    # #zero stand 
    # 静止站姿惩罚：小命令下约束腰、髋 roll/yaw 和踝 roll 回到默认姿态；
    # 腿部 pitch 由独立的 0.40/-0.20/-0.20 目标项约束，避免与默认姿态产生冲突。
    stand_still = RewTerm(
        func=mdp.stand_still,
        weight=-5.0,
        params={
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "waist_.*_joint",
                    ".*_hip_x_joint",
                    ".*_hip_z_joint",
                    ".*_ankle_x_joint",
                ],
            ),
        },
    )

    # 静止腰 yaw 额外锁定：waist_z 默认角为 0，站立时单独加重惩罚，避免腰 z 轴歪得太多。
    stand_still_waist_yaw = RewTerm(
        func=mdp.stand_still,
        weight=-1.0,
        params={
            "command_threshold": 0.4,
            "asset_cfg": SceneEntityCfg("robot", joint_names=["waist_z_joint"]),
        },
    )

    # 双足腾空时间奖励：当前注释关闭，不参与训练。
    # feet_air_time = RewTerm(
    #     func=mdp.feet_air_time_positive_biped,
    #     weight=0.15,
    #     params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_x.*"), "threshold": 0.4},
    # )
    
    # ==================== DWAQ Core Rewards (from DreamWaQ) ====================
    # Survival bonus - 使用与 HumanoidDreamWaq 一致的权重
    # HumanoidDreamWaq 使用 alive = 0.15，较小的值避免偷懒站立
    # alive = RewTerm(func=mdp.alive, weight=0.15)
    
    # ==================== 偷懒惩罚 (DWAQ 专用) ====================
    # 核心问题: 机器人学会"收到移动命令但站着不动"的偷懒策略
    # 解决方案: 直接惩罚"被命令移动但实际静止"的行为
    # - cmd_threshold=0.2: 命令速度 > 0.2 m/s 时视为"需要移动"
    # - vel_threshold=0.1: 实际速度 < 0.1 m/s 时视为"静止"
    # - weight=-2.0: 每步惩罚 -2.0，与 termination_penalty=-200 形成对比
    #   (站着20秒 = 1000步 × 2.0 = -2000，远比摔倒惩罚高)
    # idle_penalty = RewTerm(
    #     func=mdp.idle_when_commanded,
    #     weight=-2.0,
    #     # weight=-4.0,
    #     params={"cmd_threshold": 0.2, "vel_threshold": 0.1},
    # )
    
    #Swing foot height control - 控制抬腿高度
    # feet_swing_height = RewTerm(
    #     func=mdp.feet_swing_height,
    #     # weight=-0.2,
    #     weight=-0.4,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*ankle_x_link.*"),
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_x_link.*"),
    #         "target_height": 0.08,
    #     },
    # )
    
    # Gait phase matching for bipedal walking - 学习正确的两足步态
    # 奖励机器人在正确的相位进行触地/摆动
    # - stance phase (phase < 0.55): 脚应该接触地面
    # - swing phase (phase >= 0.55): 脚应该在空中
    # 注意: body_names 顺序必须是 [左脚, 右脚]，与 leg_phase 顺序一致
    # gait_phase_contact = RewTerm(
    #     func=mdp.gait_phase_contact,
    #     # weight=0.2,
    #     weight=1.0,
    #     params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=["l_ankle_x.*", "r_ankle_x.*"]), "stance_threshold": 0.55},
    # )
    
    


@configclass
class Elf3WalkTangFlatEnvCfg:
    amp_motion_files_display = [
                                # "legged_lab/envs/elf3/datasets/motion_visualization/stand.txt",
                                "legged_lab/envs/elf3/datasets/motion_visualization/stand_back.txt",
                                # "legged_lab/envs/elf3/datasets/motion_visualization/walk_run.txt",
                                # "legged_lab/envs/elf3/datasets/motion_visualization/walk_around.txt",
                                "legged_lab/envs/elf3/datasets/motion_visualization/walk_left.txt",
                                "legged_lab/envs/elf3/datasets/motion_visualization/walk_right.txt",
                                # "legged_lab/envs/elf3/datasets/motion_visualization/walk.txt",
                                ]
    device: str = "cuda:0"
    scene: BaseSceneCfg = BaseSceneCfg(
        max_episode_length_s=20.0,
        num_envs=4096,
        env_spacing=2.5,
        robot=ELF3_TANG_STRAIGHT_CFG,
        #崎岖地形
        terrain_type="generator",
        terrain_generator=GRAVEL_TERRAINS_CFG,
        #平地
        # terrain_type="plane",
        # terrain_generator=None,
        max_init_terrain_level=1,
        height_scanner=HeightScannerCfg(
            enable_height_scan=False,
            prim_body_name="torso_link",
            resolution=0.1,
            size=(1.6, 1.0),
            debug_vis=False,
            drift_range=(0.0, 0.0),  # (0.3, 0.3)
        ),
    )
    robot: RobotCfg = RobotCfg(
        actor_obs_history_length=10,
        critic_obs_history_length=10,
        # critic_obs_history_length=1,
        # Policy action order is defined in elf3_env.py as:
        # waist_y, waist_x, waist_z,
        # l_hip_y, l_hip_x, l_hip_z, l_knee_y, l_ankle_y, l_ankle_x,
        # r_hip_y, r_hip_x, r_hip_z, r_knee_y, r_ankle_y, r_ankle_x.
        # Keep this list 15-D to avoid silently slicing a 29-D full-body scale table with a different joint order.
        action_scale=[
            0.231, 0.154, 0.213,
            0.213, 0.213, 0.231, 0.213, 0.373, 0.230,
            0.213, 0.213, 0.231, 0.213, 0.373, 0.230,
        ],
        terminate_contacts_body_names=[".*_hip_z.*", "waist_z.*", "torso_link"],
        feet_body_names=[".*_ankle_x.*"],
    )
    reward = LiteRewardCfg()
    gait = GaitCfg()
    arm_motion = ArmMotionCurriculumCfg()
    reset_velocity_curriculum = ResetVelocityCurriculumCfg()
    push_velocity_curriculum = PushVelocityCurriculumCfg()
    normalization: NormalizationCfg = NormalizationCfg(
        obs_scales=ObsScalesCfg(
            lin_vel=1.0,
            ang_vel=1.0,
            projected_gravity=1.0,
            commands=1.0,
            joint_pos=1.0,
            joint_vel=1.0,
            actions=1.0,
            height_scan=1.0,
        ),
        clip_observations=100.0,
        clip_actions=100.0,
        height_scan_offset=0.5,
    )
    commands: CommandsCfg = CommandsCfg(
        resampling_time_range=(10.0, 10.0),
        # resampling_time_range=(10.0, 20.0),
        # rel_standing_envs=0.3,
        # rel_standing_envs=0.2,
        rel_standing_envs=1.0,
        rel_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=CommandRangesCfg(
            lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0), heading=(0.0, 0.0)
            # lin_vel_x=(-0.6, 3.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.57, 1.57), heading=(-math.pi, math.pi)#run
        ),
    )
    noise: NoiseCfg = NoiseCfg(
        add_noise=True,
        noise_scales=NoiseScalesCfg(
            lin_vel=0.2,
            ang_vel=0.2,
            projected_gravity=0.05,
            joint_pos=0.01,
            joint_vel=1.5,
            height_scan=0.1,
        ),
    )
    domain_rand: DomainRandCfg = DomainRandCfg(
        events=EventCfg(
            physics_material=EventTerm(
                func=mdp.randomize_rigid_body_material,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                    "static_friction_range": (0.6, 1.2),
                    "dynamic_friction_range": (0.4, 1.0),
                    "restitution_range": (0.0, 0.005),
                    "num_buckets": 64,
                },
            ),
            add_base_mass=EventTerm(
                func=mdp.randomize_rigid_body_mass,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                    "mass_distribution_params": (-0.5, 0.5),
                    "operation": "add",
                },
            ),
            reset_base=EventTerm(
                func=mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (-0.10, 0.10)},
                    "velocity_range": {
                        "x": (0.0, 0.0),
                        "y": (0.0, 0.0),
                        "z": (0.0, 0.0),
                        "roll": (0.0, 0.0),
                        "pitch": (0.0, 0.0),
                        "yaw": (0.0, 0.0),
                    },
                },
            ),
            reset_robot_joints=EventTerm(
                func=mdp.reset_joints_by_offset,
                mode="reset",
                params={
                    "position_range": (-0.05, 0.05),
                    "velocity_range": (0.0, 0.0),
                },
            ),
            push_robot=EventTerm(
                func=mdp.push_by_setting_velocity,
                mode="interval",
                # interval_range_s=(10.0, 15.0),
                # interval_range_s=(4.0, 8.0),
                interval_range_s=(8.0, 12.0),
                # params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
                params={"velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-0.07, 0.07)}},
            ),
        ),
        action_delay=ActionDelayCfg(enable=False, params={"max_delay": 5, "min_delay": 0}),
    )
    sim: SimCfg = SimCfg(dt=0.005, decimation=4, physx=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15))


@configclass
class Elf3WalkTangAgentCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    # num_steps_per_env = 32
    max_iterations = 200000
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.003,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        # learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        symmetry_cfg=None,  # RslRlSymmetryCfg()
        # symmetry_cfg=RslRlSymmetryCfg(),  # RslRlSymmetryCfg()
        rnd_cfg=None,  # RslRlRndCfg() 
    )
    clip_actions = None
    save_interval = 100
    runner_class_name = "OnPolicyRunner"
    experiment_name = "walk_tang"
    run_name = ""
    logger = "tensorboard"
    neptune_project = "walk_tang"
    wandb_project = "walk_tang"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"

 
