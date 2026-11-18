from glob import glob

from setuptools import find_packages, setup

package_name = "nao_rl_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", glob("config/*.yaml")),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools", "numpy", "PyYAML"],
    zip_safe=True,
    description="Runs Isaac Sim-trained walking policies on the real NAO through naoqi_driver2.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "policy_node = nao_rl_bridge.ros.policy_node:main",
            "check_sensors = nao_rl_bridge.ros.check_sensors_node:main",
            "walk_operator = nao_rl_bridge.ros.operator_cli:main",
        ],
    },
)
