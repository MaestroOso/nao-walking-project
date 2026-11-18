# Deployment procedure (session with the robot)

Checklist for running the report's protocol (§5.1.1) on the real NAO. An
operator stays next to the robot during every live run, ready to catch it.

## 0. Before the session (no robot)

1. The solution folder has a valid `deployment.yaml`:
   `python3 robot-integration/tools/validate_deployment.py --require-artifacts`
2. Export it (PPO/SAC in the Isaac/SB3 env; PGRL with any Python):
   ```bash
   setup/isaac_env/bin/python robot-integration/export/export_policy.py walking-solutions/<algo>/<variant>
   ```
   The export aborts if the NumPy policy does not reproduce SB3. Keep the
   printed verification line.
3. Optional, recommended once per solution: record a simulated episode with
   the exported policy and compare the robot-side reconstruction:
   ```bash
   setup/isaac_env/bin/python robot-integration/export/record_sim_traces.py \
       walking-solutions/<algo>/<variant> --export robot-integration/exported/<name>
   python3 robot-integration/tools/check_sim_traces.py robot-integration/exported/traces_<name>.npz
   ```
   Observation components must be `OK`.
4. Copy the export folder to the laptop if it is a different machine.

## 1. Set up

1. Flat, hard floor; mark a start line and a finish line 1 m apart
   (`protocol.distance_m`). Robot charged, joints cool.
2. Ethernet cable laptop ↔ robot; note the robot's IP; set it in
   `robot.yaml` (`robot.ip`) or pass `nao_ip:=`.
3. Optional: to let `prepare.fall_manager_enabled: false` take effect, the
   fall manager must be allowed to be disabled in the robot's web page.
   The default keeps it enabled.
4. Start the driver (or pass `start_driver:=true` to the launch file):
   `ros2 launch naoqi_driver naoqi_driver.launch.py nao_ip:=<ip>`

## 2. Sensor check (robot standing, stiffness on, nothing commanded)

```bash
ros2 run nao_rl_bridge check_sensors --ros-args -p export_dir:=<export>
```

| Action | Expected |
|---|---|
| Robot upright, still | g ≈ (0, 0, −1), gyro ≈ 0, torso height 0.30–0.33 m |
| Tilt torso forward | pitch > 0, g_x > 0 |
| Tilt to the robot's left | roll < 0, g_y > 0 |
| Turn it to its left by hand | gyro_z > 0 while turning |
| — | no "missing leg joints" |

If a sign is wrong, stop: the observation would not match training. Fix
the convention (IMU frame / driver version) before any live run.

## 3. Dry run

```bash
ros2 launch nao_rl_bridge deploy.launch.py export_dir:=<export> nao_ip:=<ip>   # dry_run defaults to true
ros2 run nao_rl_bridge walk_operator                                           # second terminal: press s
```

Nothing reaches the motors. Hold the robot in the home pose by hand or
leave it standing; check in the log that the loop runs at 30 Hz
(`loop_overruns: 0` in `run.json`), that `/walk/command` looks sensible
(`ros2 topic echo /walk/command`), and that a run folder appears in
`~/nao_robot_runs/`.

## 4. Live runs

```bash
ros2 launch nao_rl_bridge deploy.launch.py export_dir:=<export> nao_ip:=<ip> \
    dry_run:=false label:=<PPO_robot> delta_deg:=<δ> repetition:=<k> [record_bag:=true]
ros2 run nao_rl_bridge walk_operator
```

Operator keys: `s` start, `f` finish line crossed, `x` stop (freeze),
`SPACE`/`e` emergency stop, `r` rest, `q` quit.

One repetition:
1. Robot at the start line. `s`: it wakes up, goes to StandInit, then
   interpolates to the warm-up pose (3 s), settles, holds the warm-up
   (2 s), then walks.
2. When its torso crosses the finish line press `f` (success). A fall
   (tilt > 40° or torso below 0.20 m) and the 50 s limit end the run
   automatically. After a success or timeout the robot returns to the home
   pose; after a fall it goes to rest.
3. Measure the forward distance and lateral offset (positive to the
   robot's left) and record them:
   ```bash
   python3 robot-integration/tools/record_measurement.py ~/nao_robot_runs/<run> --distance <m> --lateral <m>
   ```
4. An `aborted` run (operator stop, e-stop, sensor/timing fault, pose
   check) is not a repetition: repeat it with the same `repetition:=k`.

Protocol: for each policy, δ ∈ {0, 0.5, 1, 2, 4}° × repetitions k = 0…9
(restart the launch with the new `delta_deg`/`repetition`; repetition k
uses seed `base_seed + k`, the same pose as repetition k in simulation).
Start every new policy at δ = 0 with the robot in a harness or held.

## 5. Results

```bash
python3 robot-integration/tools/summarize_robot_runs.py --label <PPO_robot> --out robot_ppo.csv
```

Prints completed / TC / F_S / time / lateral per δ and writes the
repetitions in the format of
`experiments/test_evaluation_script/external_results_template.csv`, so they
can be merged with the simulation results. Runs still using the odometry
estimate as distance (no tape measurement) are listed.
