"""
check_sensors_node.py — verify the robot's sensor conventions before any run.

    ros2 run nao_rl_bridge check_sensors --ros-args -p export_dir:=<exported/NAME>

Prints, twice a second, what the policy would see: roll/pitch, projected
gravity (correct and, for legacy policies, as the PPO baseline saw it),
gyro, the kinematic torso height and stance foot, and how far each leg joint
is from the policy's home pose. Nothing is sent to the robot.

Manual checks (simulation conventions, see docs/deployment-procedure.md):
  1. Robot upright and still: g ≈ (0, 0, -1), gyro ≈ 0, height ≈ 0.30-0.33 m.
  2. Tilt the torso FORWARD: pitch > 0 and g_x > 0.
  3. Tilt to the robot's LEFT: roll < 0 and g_y > 0.
  4. Rotate it to its left (yaw) by hand: gyro_z > 0 while turning.
  5. Every leg joint is reported (no 'missing').
"""
from __future__ import annotations

import math

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState

from ..core.estimators import StateEstimator
from ..core.exported import load_export
from ..core.kinematics import LegKinematics
from ..core.robot_config import load_robot_config
from ..core.rotations import GRAVITY_DIR, rotate_inverse_legacy, rotate_inverse_wxyz
from .policy_node import SensorCache, default_robot_config


class CheckSensors(Node):
    def __init__(self):
        super().__init__("nao_check_sensors")
        self.declare_parameter("export_dir", "")
        self.declare_parameter("robot_config", str(default_robot_config()))
        ex = load_export(self.get_parameter("export_dir").value)
        cfg = load_robot_config(self.get_parameter("robot_config").value)
        self.spec = ex.spec
        self.legacy = bool(ex.spec.observation and
                           ex.spec.observation.quaternion_convention == "legacy_xyzw")
        spawn = math.radians(ex.spec.observation.spawn_yaw_deg) if ex.spec.observation else math.pi
        self.est = StateEstimator(LegKinematics(ex.urdf_path), ex.spec.leg_order, spawn,
                                  cfg["estimator"]["velocity_filter_alpha"],
                                  cfg["estimator"]["height_offset_m"])
        self.sensors = SensorCache()
        self.create_subscription(JointState, cfg["topics"]["joint_states"], self.sensors.on_joints, 10)
        self.create_subscription(Imu, cfg["topics"]["imu"], self.sensors.on_imu, 10)
        self.create_timer(0.5, self.report)
        print(__doc__)

    def report(self):
        s = self.sensors.sample()
        if s is None:
            print("waiting for /joint_states and the IMU ...")
            return
        missing = [n for n in self.spec.leg_order if n not in s.joints]
        if missing:
            print(f"missing leg joints in /joint_states: {missing}")
            return
        st = self.est.update(s)
        g = rotate_inverse_wxyz(st.quat_world, GRAVITY_DIR)
        dev = np.degrees(st.leg_q - self.spec.home_pose_rad)
        worst = int(np.argmax(np.abs(dev)))
        line = (f"roll {math.degrees(st.roll):+6.1f}  pitch {math.degrees(st.pitch):+6.1f}  "
                f"g ({g[0]:+.2f} {g[1]:+.2f} {g[2]:+.2f})  "
                f"gyro ({st.gyro_body[0]:+.2f} {st.gyro_body[1]:+.2f} {st.gyro_body[2]:+.2f})  "
                f"h {st.base_height:.3f} m [{st.stance_foot}]  "
                f"max |q-home| {abs(dev[worst]):.1f}° ({self.spec.leg_order[worst]})")
        if self.legacy:
            gl = rotate_inverse_legacy(st.quat_world, GRAVITY_DIR)
            line += f"  legacy g ({gl[0]:+.2f} {gl[1]:+.2f} {gl[2]:+.2f})"
        print(line)


def main(args=None):
    rclpy.init(args=args)
    node = CheckSensors()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
