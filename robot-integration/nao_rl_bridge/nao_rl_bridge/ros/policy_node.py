"""
policy_node.py — ROS 2 node that runs an exported walking policy on the NAO.

    ros2 run nao_rl_bridge policy_node --ros-args \
        -p export_dir:=<robot-integration/exported/NAME> -p dry_run:=true \
        -p delta_deg:=0.0 -p repetition:=0

Subscribes (naoqi_driver2):  /joint_states (sensor_msgs/JointState),
                             /imu/torso    (sensor_msgs/Imu)
Publishes:                   /joint_angles (naoqi_bridge_msgs/JointAnglesWithSpeed)
                                 -- only when dry_run is false
                             /walk/command (sensor_msgs/JointState)  every command, also in dry run
                             /walk/observation, /walk/action (std_msgs/Float32MultiArray)
                             /walk/status  (std_msgs/String)
Services (std_srvs/Trigger): /walk/start        start one repetition
                             /walk/finish_line  the robot crossed the finish line (success)
                             /walk/stop         abort the run (robot freezes)
                             /walk/rest         NAOqi rest (crouch, stiffness off), when idle
Topic:                       /walk/emergency_stop (std_msgs/Empty) = abort + freeze

dry_run (default TRUE): nothing is sent to the motors and NAOqi is not
touched; sensors are read and every command is computed, logged and
published on /walk/command.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import Empty, Float32MultiArray, String
from std_srvs.srv import Trigger

from ..core.estimators import SensorSample
from ..core.exported import load_export
from ..core.robot_config import load_robot_config
from ..core.rotations import rpy_from_quat
from ..robot.naoqi_backend import DryRunBackend, NaoqiBackend
from ..session import WalkSession


def default_robot_config() -> Path:
    try:
        from ament_index_python.packages import get_package_share_directory
        return Path(get_package_share_directory("nao_rl_bridge")) / "config" / "robot.yaml"
    except Exception:
        return Path(__file__).resolve().parents[2] / "config" / "robot.yaml"


class SensorCache:
    """Latest joint states and IMU, stamped with the local receive time."""

    def __init__(self):
        self._lock = threading.Lock()
        self.joints, self.joint_stamp = {}, None
        self.imu, self.imu_stamp = None, None

    def on_joints(self, msg: JointState):
        with self._lock:
            self.joints = dict(zip(msg.name, msg.position))
            self.joint_stamp = time.monotonic()

    def on_imu(self, msg: Imu):
        o, w = msg.orientation, msg.angular_velocity
        roll, pitch, yaw = rpy_from_quat((o.w, o.x, o.y, o.z))     # ROS is (x, y, z, w)
        with self._lock:
            self.imu = (roll, pitch, yaw, (w.x, w.y, w.z))
            self.imu_stamp = time.monotonic()

    def sample(self):
        with self._lock:
            if self.imu is None or self.joint_stamp is None:
                return None
            roll, pitch, yaw, gyro = self.imu
            return SensorSample(joints=dict(self.joints), roll=roll, pitch=pitch, yaw=yaw,
                                gyro=gyro, stamp=min(self.joint_stamp, self.imu_stamp))


class PolicyNode(Node):
    def __init__(self):
        super().__init__("nao_policy_node")
        self.declare_parameter("export_dir", "")
        self.declare_parameter("robot_config", str(default_robot_config()))
        self.declare_parameter("dry_run", True)
        self.declare_parameter("robot_ip", "")
        self.declare_parameter("label", "")
        any_type = ParameterDescriptor(dynamic_typing=True)   # launch args: "2" vs "2.0"
        self.declare_parameter("delta_deg", 0.0, any_type)
        self.declare_parameter("repetition", 0, any_type)
        gp = lambda n: self.get_parameter(n).value                           # noqa: E731

        if not gp("export_dir"):
            raise RuntimeError("parameter export_dir is required")
        self.cfg = load_robot_config(gp("robot_config"))
        self.exported = load_export(gp("export_dir"))
        self.dry_run = bool(gp("dry_run"))
        ip = gp("robot_ip") or self.cfg["robot"]["ip"]
        self.get_logger().info(
            f"policy '{self.exported.name}' ({self.exported.spec.algorithm}), "
            f"{'DRY RUN (motors untouched)' if self.dry_run else f'LIVE on {ip}'}")

        self.sensors = SensorCache()
        t = self.cfg["topics"]
        self.create_subscription(JointState, t["joint_states"], self.sensors.on_joints, 10)
        self.create_subscription(Imu, t["imu"], self.sensors.on_imu, 10)
        self.pub_cmd_preview = self.create_publisher(JointState, "/walk/command", 10)
        self.pub_obs = self.create_publisher(Float32MultiArray, "/walk/observation", 10)
        self.pub_act = self.create_publisher(Float32MultiArray, "/walk/action", 10)
        self.pub_status = self.create_publisher(String, "/walk/status", 10)
        self.pub_angles = None
        if self.dry_run:
            self.backend = DryRunBackend(self.get_logger())
        else:
            from naoqi_bridge_msgs.msg import JointAnglesWithSpeed
            self._JointAngles = JointAnglesWithSpeed
            self.pub_angles = self.create_publisher(JointAnglesWithSpeed, t["joint_angles"], 10)
            self.backend = NaoqiBackend(ip, int(self.cfg["robot"]["port"]), self.get_logger())

        self.create_service(Trigger, "/walk/start", self.srv_start)
        self.create_service(Trigger, "/walk/finish_line", self.srv_finish)
        self.create_service(Trigger, "/walk/stop", self.srv_stop)
        self.create_service(Trigger, "/walk/rest", self.srv_rest)
        self.create_subscription(Empty, "/walk/emergency_stop", self.on_estop, 10)
        self.session = None
        self.thread = None
        self.status("idle")

    # ── output ──
    def status(self, text: str):
        self.pub_status.publish(String(data=text))
        self.get_logger().info(f"status: {text}")

    def send_command(self, names, angles, speed):
        stamp = self.get_clock().now().to_msg()
        preview = JointState(name=list(names), position=[float(a) for a in angles])
        preview.header.stamp = stamp
        self.pub_cmd_preview.publish(preview)
        if self.pub_angles is not None:
            m = self._JointAngles()
            m.header.stamp = stamp
            m.joint_names = list(names)
            m.joint_angles = [float(a) for a in angles]
            m.speed = float(speed)
            m.relative = 0
            self.pub_angles.publish(m)

    def publish_debug(self, res):
        if res.step.obs is not None:
            self.pub_obs.publish(Float32MultiArray(data=[float(x) for x in res.step.obs]))
        if res.step.action is not None:
            self.pub_act.publish(Float32MultiArray(data=[float(x) for x in res.step.action]))

    # ── services ──
    def _busy(self):
        return self.thread is not None and self.thread.is_alive()

    def srv_start(self, req, resp):
        if self._busy():
            resp.success, resp.message = False, "a run is already in progress"
            return resp
        gp = lambda n: self.get_parameter(n).value                           # noqa: E731
        self.session = WalkSession(
            self.exported, self.cfg, self.backend, self.sensors.sample, self.send_command,
            self.publish_debug, label=gp("label"), delta_deg=float(gp("delta_deg")),
            repetition=int(gp("repetition")), logger=self.get_logger())
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        resp.success = True
        resp.message = (f"started {self.session.label} δ={gp('delta_deg')}° "
                        f"rep={gp('repetition')} seed={self.session.seed}")
        return resp

    def _run(self):
        self.status("running")
        s = self.session.run()
        self.status(f"done: {s.outcome} ({s.reason}) walk_time={s.walk_time_s:.2f}s "
                    f"log={self.session.run_dir}")

    def srv_finish(self, req, resp):
        if not self._busy():
            resp.success, resp.message = False, "no run in progress"
            return resp
        self.session.request_finish()
        resp.success, resp.message = True, "finish line marked"
        return resp

    def srv_stop(self, req, resp):
        if self._busy():
            self.session.request_stop("operator stop")
        resp.success, resp.message = True, "stop requested"
        return resp

    def srv_rest(self, req, resp):
        if self._busy():
            resp.success, resp.message = False, "stop the run first"
            return resp
        self.backend.rest()
        resp.success, resp.message = True, "rest"
        return resp

    def on_estop(self, _msg):
        self.get_logger().warning("EMERGENCY STOP")
        if self._busy():
            self.session.request_stop("emergency stop")
            self.backend.emergency_stop()          # also interrupts a blocking interpolation
        else:
            self.backend.freeze(list(self.exported.spec.leg_order) + list(self.exported.spec.fixed_pose_rad))

    def shutdown(self):
        if self._busy():
            self.session.request_stop("node shutdown")
            self.thread.join(timeout=5.0)
        self.backend.close()


def main(args=None):
    rclpy.init(args=args)
    node = PolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
