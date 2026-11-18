"""
naoqi_backend.py — what naoqi_driver2 does not expose: stiffness, postures,
blocking interpolations, rest.

The 30 Hz joint stream goes through ROS 2 (`/joint_angles`, executed by the
driver with ALMotion.setAngles). The calls here happen only before and after
a run, through a direct `qi` session to the same robot.

`DryRunBackend` has the same interface and only logs, so the whole node can
be exercised with the motors untouched.
"""
from __future__ import annotations

import logging
import time
from typing import Dict, List, Sequence

log = logging.getLogger("nao_rl_bridge.backend")


class DryRunBackend:
    """Logs every call; never touches the robot."""

    dry_run = True

    def __init__(self, logger=None):
        self.log = logger or log
        self.calls: List[tuple] = []

    def _call(self, name, *args):
        self.calls.append((name, args))
        self.log.info(f"[dry-run] {name}{args if args else ''}")

    def prepare(self, disable_autonomous_life: bool, fall_manager_enabled: bool,
                smart_stiffness_enabled: bool):
        self._call("prepare", disable_autonomous_life, fall_manager_enabled, smart_stiffness_enabled)

    def wake_up(self):
        self._call("wake_up")

    def set_stiffness(self, stiffness: Dict[str, float], ramp_s: float):
        self._call("set_stiffness", {k: round(v, 3) for k, v in stiffness.items()}, ramp_s)

    def move_to(self, names: Sequence[str], angles: Sequence[float], duration_s: float):
        self._call("move_to", len(names), duration_s)

    def freeze(self, names: Sequence[str]):
        self._call("freeze", len(names))

    def rest(self):
        self._call("rest")

    def emergency_stop(self):
        self._call("emergency_stop")

    def close(self):
        pass


class NaoqiBackend:
    """Direct NAOqi session (`pip install qi`)."""

    dry_run = False

    def __init__(self, ip: str, port: int = 9559, logger=None):
        try:
            import qi
        except ImportError as e:
            raise RuntimeError("the NAOqi Python SDK is not installed (pip install qi)") from e
        self.log = logger or log
        self.session = qi.Session()
        self.session.connect(f"tcp://{ip}:{port}")
        self.motion = self.session.service("ALMotion")
        self.posture = self.session.service("ALRobotPosture")
        try:
            self.life = self.session.service("ALAutonomousLife")
        except RuntimeError:
            self.life = None
        self.log.info(f"connected to NAOqi at {ip}:{port}")

    def prepare(self, disable_autonomous_life: bool, fall_manager_enabled: bool,
                smart_stiffness_enabled: bool):
        if disable_autonomous_life and self.life is not None and self.life.getState() != "disabled":
            self.life.setState("disabled")              # NAOqi puts the robot in rest here
        try:
            self.motion.setFallManagerEnabled(bool(fall_manager_enabled))
        except RuntimeError as e:   # disabling needs the robot's web-page setting
            self.log.warning(f"could not set fall manager to {fall_manager_enabled}: {e}")
        self.motion.setSmartStiffnessEnabled(bool(smart_stiffness_enabled))

    def wake_up(self):
        """Stiffness on and NAOqi's own safe transition to StandInit."""
        self.motion.wakeUp()
        self.posture.goToPosture("StandInit", 0.5)

    def set_stiffness(self, stiffness: Dict[str, float], ramp_s: float):
        names = list(stiffness)
        values = [float(stiffness[n]) for n in names]
        self.motion.stiffnessInterpolation(names, values, [max(ramp_s, 0.05)] * len(names))

    def move_to(self, names: Sequence[str], angles: Sequence[float], duration_s: float):
        """Blocking interpolation to absolute angles [rad]."""
        self.motion.angleInterpolation(list(names), [[float(a)] for a in angles],
                                       [[float(duration_s)]] * len(names), True)

    def freeze(self, names: Sequence[str]):
        """Hold the current measured pose."""
        current = self.motion.getAngles(list(names), True)
        self.motion.setAngles(list(names), current, 0.2)
        time.sleep(0.2)

    def rest(self):
        """NAOqi rest: crouch and stiffness off."""
        self.motion.rest()

    def emergency_stop(self):
        """Kill every running motion task (also a blocking angleInterpolation
        in another thread); stiffness stays on, so the robot holds still."""
        self.motion.killAll()

    def close(self):
        try:
            self.session.close()
        except Exception:
            pass
