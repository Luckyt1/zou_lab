import argparse
import ast
import os
import time
from pathlib import Path
from typing import Any


np = None
mujoco = None
mujoco_viewer = None


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

DEFAULT_JOINT_POS = {
    "waist_y_joint": 0.0,
    "waist_x_joint": 0.0,
    "waist_z_joint": 0.0,
    "l_hip_y_joint": -0.05,
    "l_hip_x_joint": 0.0,
    "l_hip_z_joint": 0.0,
    "l_knee_y_joint": 0.10,
    "l_ankle_y_joint": -0.05,
    "l_ankle_x_joint": 0.0,
    "r_hip_y_joint": -0.05,
    "r_hip_x_joint": 0.0,
    "r_hip_z_joint": 0.0,
    "r_knee_y_joint": 0.10,
    "r_ankle_y_joint": -0.05,
    "r_ankle_x_joint": 0.0,
    "l_shoulder_y_joint": 0.2,
    "l_shoulder_x_joint": 0.2,
    "l_shoulder_z_joint": 0.0,
    "l_elbow_y_joint": 0.6,
    "l_wrist_x_joint": 0.0,
    "l_wrist_y_joint": 0.0,
    "l_wrist_z_joint": 0.0,
    "r_shoulder_y_joint": 0.2,
    "r_shoulder_x_joint": -0.2,
    "r_shoulder_z_joint": 0.0,
    "r_elbow_y_joint": 0.6,
    "r_wrist_x_joint": 0.0,
    "r_wrist_y_joint": 0.0,
    "r_wrist_z_joint": 0.0,
}

JOINT_KP = {
    "waist_y_joint": 108.448,
    "waist_x_joint": 162.672,
    "waist_z_joint": 176.421,
    "l_hip_y_joint": 176.421,
    "l_hip_x_joint": 176.421,
    "l_hip_z_joint": 54.224,
    "l_knee_y_joint": 176.421,
    "l_ankle_y_joint": 33.493,
    "l_ankle_x_joint": 21.771,
    "r_hip_y_joint": 176.421,
    "r_hip_x_joint": 176.421,
    "r_hip_z_joint": 54.224,
    "r_knee_y_joint": 176.421,
    "r_ankle_y_joint": 33.493,
    "r_ankle_x_joint": 21.771,
    "l_shoulder_y_joint": 54.224,
    "l_shoulder_x_joint": 54.224,
    "l_shoulder_z_joint": 16.747,
    "l_elbow_y_joint": 54.224,
    "l_wrist_x_joint": 16.747,
    "l_wrist_y_joint": 16.747,
    "l_wrist_z_joint": 16.747,
    "r_shoulder_y_joint": 54.224,
    "r_shoulder_x_joint": 54.224,
    "r_shoulder_z_joint": 16.747,
    "r_elbow_y_joint": 54.224,
    "r_wrist_x_joint": 16.747,
    "r_wrist_y_joint": 16.747,
    "r_wrist_z_joint": 16.747,
}

JOINT_KD = {
    "waist_y_joint": 6.904,
    "waist_x_joint": 10.356,
    "waist_z_joint": 11.231,
    "l_hip_y_joint": 11.231,
    "l_hip_x_joint": 11.231,
    "l_hip_z_joint": 3.452,
    "l_knee_y_joint": 11.231,
    "l_ankle_y_joint": 2.132,
    "l_ankle_x_joint": 1.386,
    "r_hip_y_joint": 11.231,
    "r_hip_x_joint": 11.231,
    "r_hip_z_joint": 3.452,
    "r_knee_y_joint": 11.231,
    "r_ankle_y_joint": 2.132,
    "r_ankle_x_joint": 1.386,
    "l_shoulder_y_joint": 3.452,
    "l_shoulder_x_joint": 3.452,
    "l_shoulder_z_joint": 1.066,
    "l_elbow_y_joint": 3.452,
    "l_wrist_x_joint": 1.066,
    "l_wrist_y_joint": 1.066,
    "l_wrist_z_joint": 1.066,
    "r_shoulder_y_joint": 3.452,
    "r_shoulder_x_joint": 3.452,
    "r_shoulder_z_joint": 1.066,
    "r_elbow_y_joint": 3.452,
    "r_wrist_x_joint": 1.066,
    "r_wrist_y_joint": 1.066,
    "r_wrist_z_joint": 1.066,
}


def read_stand_amplitudes(cfg_path: Path) -> Any:
    tree = ast.parse(cfg_path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ArmMotionCurriculumCfg":
            for stmt in node.body:
                target = getattr(stmt, "target", None)
                if isinstance(target, ast.Name) and target.id == "stand_amplitudes":
                    values = ast.literal_eval(stmt.value)
                    if len(values) != len(ARM_JOINT_NAMES):
                        raise ValueError(f"stand_amplitudes must have {len(ARM_JOINT_NAMES)} values, got {len(values)}")
                    return np.asarray(values, dtype=np.float64)
    raise ValueError(f"Could not find ArmMotionCurriculumCfg.stand_amplitudes in {cfg_path}")


def joint_qpos_addr(model: Any, joint_name: str) -> int:
    return int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)])


def joint_qvel_addr(model: Any, joint_name: str) -> int:
    return int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)])


def actuator_id(model: Any, actuator_name: str) -> int:
    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name))


def smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def sample_arm_target(
    rng: Any,
    default_qpos: dict[str, float],
    amplitudes: Any,
    scale: float,
) -> dict[str, float]:
    offsets = rng.uniform(-1.0, 1.0, size=len(ARM_JOINT_NAMES)) * amplitudes * scale
    return {name: default_qpos[name] + offset for name, offset in zip(ARM_JOINT_NAMES, offsets)}


def apply_default_pose(model: Any, data: Any, default_qpos: dict[str, float]) -> None:
    data.qpos[:7] = np.array([0.0, 0.0, 1.10, 1.0, 0.0, 0.0, 0.0])
    data.qvel[:] = 0.0
    for joint_name, joint_pos in default_qpos.items():
        data.qpos[joint_qpos_addr(model, joint_name)] = joint_pos
    mujoco.mj_forward(model, data)


def run(args: argparse.Namespace) -> None:
    global np, mujoco, mujoco_viewer
    try:
        import numpy as _np
        import mujoco as _mujoco
        import mujoco_viewer as _mujoco_viewer
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "MuJoCo visualization requires Python packages 'numpy', 'mujoco', and 'mujoco_viewer'. "
            "Run this script in the same environment you use for the existing sim2sim scripts."
        ) from exc
    np = _np
    mujoco = _mujoco
    mujoco_viewer = _mujoco_viewer

    amplitudes = read_stand_amplitudes(Path(args.cfg))
    model = mujoco.MjModel.from_xml_path(args.model)
    model.opt.timestep = args.dt
    data = mujoco.MjData(model)
    viewer = mujoco_viewer.MujocoViewer(model, data)
    viewer._render_every_frame = False

    rng = np.random.default_rng(args.seed)
    default_qpos = dict(DEFAULT_JOINT_POS)
    apply_default_pose(model, data, default_qpos)

    joint_qpos_ids = {name: joint_qpos_addr(model, name) for name in DEFAULT_JOINT_POS}
    joint_qvel_ids = {name: joint_qvel_addr(model, name) for name in DEFAULT_JOINT_POS}
    actuator_ids = {name: actuator_id(model, name) for name in DEFAULT_JOINT_POS}

    start_target = {name: default_qpos[name] for name in ARM_JOINT_NAMES}
    goal_target = sample_arm_target(rng, default_qpos, amplitudes, args.scale)
    transition_start = data.time
    next_print_time = 0.0

    print("[INFO] Visualizing ArmMotionCurriculumCfg.stand_amplitudes")
    print("[INFO] Joint order:")
    for name, amp in zip(ARM_JOINT_NAMES, amplitudes):
        print(f"  {name:20s} amplitude={amp:.3f} rad, scaled={amp * args.scale:.3f} rad")

    try:
        while viewer.is_alive and data.time < args.duration:
            if data.time - transition_start >= args.resample_interval:
                start_target = goal_target
                goal_target = sample_arm_target(rng, default_qpos, amplitudes, args.scale)
                transition_start = data.time

            alpha = smoothstep((data.time - transition_start) / max(args.resample_interval, 1e-6))
            target_qpos = dict(default_qpos)
            for joint_name in ARM_JOINT_NAMES:
                target_qpos[joint_name] = (1.0 - alpha) * start_target[joint_name] + alpha * goal_target[joint_name]

            tau = np.zeros(model.nu)
            for joint_name, target in target_qpos.items():
                q = data.qpos[joint_qpos_ids[joint_name]]
                dq = data.qvel[joint_qvel_ids[joint_name]]
                tau[actuator_ids[joint_name]] = (target - q) * JOINT_KP[joint_name] - dq * JOINT_KD[joint_name]
            data.ctrl[:] = tau

            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.render()

            if args.print_targets and data.time >= next_print_time:
                summary = ", ".join(f"{name}={goal_target[name]:+.2f}" for name in ARM_JOINT_NAMES[:4])
                print(f"[t={data.time:6.2f}s] next left arm target: {summary}, ...")
                next_print_time = data.time + args.resample_interval

            sleep_time = args.dt - (time.time() - step_start)
            if sleep_time > 0.0:
                time.sleep(sleep_time)
    finally:
        viewer.close()


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Visualize elf3_tang arm randomization in MuJoCo.")
    parser.add_argument(
        "--model",
        default=str(root / "legged_lab/assets/elf3_lite/xml/elf3.xml"),
        help="Path to the MuJoCo XML model.",
    )
    parser.add_argument(
        "--cfg",
        default=str(root / "legged_lab/envs/elf3_tang/walk_cfg.py"),
        help="Path to elf3_tang walk_cfg.py. The script reads stand_amplitudes from this file.",
    )
    parser.add_argument("--duration", type=float, default=60.0, help="Visualization duration in seconds.")
    parser.add_argument("--dt", type=float, default=0.005, help="MuJoCo simulation timestep.")
    parser.add_argument("--resample-interval", type=float, default=1.0, help="Seconds between new arm targets.")
    parser.add_argument("--scale", type=float, default=1.0, help="Multiplier applied to stand_amplitudes.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for arm target sampling.")
    parser.add_argument("--print-targets", action="store_true", help="Print sampled arm targets to the terminal.")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
