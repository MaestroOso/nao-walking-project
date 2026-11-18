"""
operator_cli.py — single-key control of the policy node during the protocol.

    ros2 run nao_rl_bridge walk_operator

    s  start the repetition          f  finish line crossed (success)
    x  stop (abort, robot freezes)   SPACE / e  EMERGENCY STOP
    r  rest (crouch, stiffness off)  q  quit
"""
from __future__ import annotations

import select
import sys
import termios
import tty

import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty, String
from std_srvs.srv import Trigger

KEYS = {"s": "/walk/start", "f": "/walk/finish_line", "x": "/walk/stop", "r": "/walk/rest"}


class Operator(Node):
    def __init__(self):
        super().__init__("walk_operator")
        self.clients_ = {k: self.create_client(Trigger, srv) for k, srv in KEYS.items()}
        self.estop = self.create_publisher(Empty, "/walk/emergency_stop", 10)
        self.create_subscription(String, "/walk/status", lambda m: print(f"\r[status] {m.data}"), 10)

    def call(self, key):
        cli = self.clients_[key]
        if not cli.wait_for_service(timeout_sec=1.0):
            print(f"\r{KEYS[key]} not available (is the policy node running?)")
            return
        fut = cli.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
        r = fut.result()
        print(f"\r{KEYS[key]}: {'OK' if r and r.success else 'FAILED'} {r.message if r else ''}")


def main(args=None):
    rclpy.init(args=args)
    node = Operator()
    print(__doc__)
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            if select.select([sys.stdin], [], [], 0)[0]:
                k = sys.stdin.read(1).lower()
                if k in (" ", "e"):
                    node.estop.publish(Empty())
                    print("\rEMERGENCY STOP sent")
                elif k in KEYS:
                    node.call(k)
                elif k == "q":
                    break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
