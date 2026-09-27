"""Offline direct-multiple-shooting minimum-time reference optimization."""

from dataclasses import asdict
import numpy as np
import jax
import jax.numpy as jnp

from a2rl_drone_training.config import PlannerConfig
from .geometry import frame_boxes, validate_openings, fingerprint
from .reference import Reference
from .dynamics import acceleration_quaternion


def initial_guess(course, model, nodes):
    from scipy.interpolate import BPoly
    from scipy.spatial.transform import Rotation

    # Explicit outgoing/incoming normal tangents generate turnaround arcs for
    # coincident stacked openings, rather than a vertical center-to-center line.
    start = course.nominal_racing_line[0].copy()
    if np.dot(start - course.centers[0], course.normals[0]) >= -0.2:
        start = course.centers[0] - 1.5 * course.normals[0]
    points = np.vstack((start, course.centers))
    distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
    stacked = np.r_[False, course.logical_gate_ids[1:] == course.logical_gate_ids[:-1]]
    durations = np.maximum(distances / 2.0, 1.0) + stacked * 2.0
    events = np.r_[0, np.cumsum(durations)]
    tangents = np.vstack((course.normals[0], course.normals)) * 1.5
    tangents[0] = 0.0
    seed_times, seed_points, seed_tangents = [events[0]], [points[0]], [tangents[0]]
    for segment in range(len(durations)):
        if stacked[segment]:
            # Explicit exit and re-approach waypoints on the shared far side.
            outgoing = points[segment] + 1.5 * course.normals[segment - 1]
            incoming = points[segment + 1] - 1.5 * course.normals[segment]
            vertical = incoming - outgoing
            vertical = vertical / max(np.linalg.norm(vertical), 1e-6) * 1.0
            for fraction, point in ((1 / 3, outgoing), (2 / 3, incoming)):
                seed_times.append(events[segment] + fraction * durations[segment])
                seed_points.append(point)
                seed_tangents.append(vertical)
        seed_times.append(events[segment + 1])
        seed_points.append(points[segment + 1])
        seed_tangents.append(tangents[segment + 1])
    # Continuous acceleration and jerk keep attitude/rate seeds consistent at
    # gates; cubic position splines alone introduce instantaneous attitude jumps.
    spline = BPoly.from_derivatives(
        seed_times,
        [[p, v, np.zeros(3), np.zeros(3)] for p, v in zip(seed_points, seed_tangents)],
    )
    time = np.concatenate(
        [
            np.linspace(events[i], events[i + 1], nodes, endpoint=False)
            for i in range(len(durations))
        ]
        + [events[-1:]]
    )
    x = np.zeros((len(time), 17))
    x[:, :3], x[:, 3:6] = spline(time), spline(time, 1)
    # Constant heading avoids artificial yaw flips at stacked turnarounds. The
    # optimizer remains free to choose yaw. Feedforward seeding includes drag,
    # required body torque, and the inverse rotor response.
    yaw = np.full(len(time), np.arctan2(course.normals[0, 1], course.normals[0, 0]))
    p = model.parameters
    acceleration = spline(time, 2)
    drag_acc = (np.asarray(p["drag_matrix"]) @ x[:, 3:6].T).T / p["mass"]
    thrust_acc = acceleration - drag_acc
    x[:, 6:10] = np.asarray(
        jax.vmap(acceleration_quaternion)(jnp.asarray(thrust_acc), jnp.asarray(yaw))
    )
    rotations = Rotation.from_quat(x[:, 6:10])
    rates = (rotations[:-1].inv() * rotations[1:]).as_rotvec() / np.diff(time)[:, None]
    x[:, 10:13] = np.vstack((rates, rates[-1]))
    rates_dot = np.gradient(x[:, 10:13], time, axis=0)
    torque = rates_dot @ p["J"].T + np.cross(x[:, 10:13], x[:, 10:13] @ p["J"].T)
    curve, moment = p["rpm2thrust"], p["rpm2torque"]
    ratio = (moment[1] * model.rpm_hover + moment[2] * model.rpm_hover**2) / (
        curve[1] * model.rpm_hover + curve[2] * model.rpm_hover**2
    )
    allocation = np.vstack(
        (np.ones(4), p["mixing_matrix"] * np.array([p["L"], p["L"], ratio])[:, None])
    )
    collective = p["mass"] * np.linalg.norm(thrust_acc - p["gravity_vec"], axis=1)
    forces = np.linalg.solve(allocation, np.column_stack((collective, torque)).T).T
    rpm = (
        -curve[1]
        + np.sqrt(np.maximum(curve[1] ** 2 - 4 * curve[2] * (curve[0] - forces), 0))
    ) / (2 * curve[2])
    rpm = np.clip(rpm, model.rpm_min, model.rpm_max)
    rpm_dot = np.gradient(rpm, time, axis=0)
    coef = p["rotor_dyn_coef"]
    linear = np.where(rpm_dot >= 0, coef[0], coef[2])
    quadratic = np.where(rpm_dot >= 0, coef[1], coef[3])
    command = np.where(
        quadratic > 1e-12,
        (
            -linear
            + np.sqrt(
                np.maximum(
                    linear**2
                    + 4 * quadratic * (rpm_dot + linear * rpm + quadratic * rpm**2),
                    0,
                )
            )
        )
        / np.maximum(2 * quadratic, 1e-12),
        rpm + rpm_dot / np.maximum(linear, 1e-12),
    )
    actions = np.asarray(model.action(jnp.asarray(command)))
    x[:, 13:] = rpm / model.rpm_hover
    x[0, 6:10] = np.array([0, 0, np.sin(yaw[0] / 2), np.cos(yaw[0] / 2)])
    x[0, 10:13] = 0
    x[0, 13:] = 1
    return time, x, actions, np.arange(1, course.num_gates + 1) * nodes


def validate_reference(ref, course, model, config, radius):
    """Dense per-interval dynamics replay; failure never produces a valid artifact."""
    boxes = frame_boxes(course, config.frame_depth_m)
    margin = radius + config.tracking_margin_m
    bad = []
    maximum_defect = 0.0
    minimum_clearance = float("inf")
    # Validate each interval from its shooting state: open-loop integration over
    # the entire lap would conflate numerical defects with unstable flight.
    durations = np.diff(ref.time)
    count = max(1, int(np.ceil(np.max(durations) * config.validation_hz)))
    trajectories = model.rollout_intervals(
        jnp.asarray(ref.states[:-1], dtype=jnp.float32),
        jnp.asarray(ref.actions[:-1], dtype=jnp.float32),
        jnp.asarray(durations, dtype=jnp.float32),
        count,
    )
    all_clearance = np.asarray(boxes.clearance(trajectories[:, :, :3], margin))
    all_collision = np.asarray(
        boxes.swept_collision(trajectories[:, :-1, :3], trajectories[:, 1:, :3], margin)
    ).any(axis=1)
    trajectories = np.asarray(trajectories)
    minimum_clearance = float(np.min(all_clearance))
    for i, trajectory in enumerate(trajectories):
        bounds = np.any(
            trajectory[:, :3] < course.bounds_min + margin - 1e-4
        ) or np.any(trajectory[:, :3] > course.bounds_max - margin + 1e-4)
        error = trajectory[-1] - ref.states[i + 1]
        if trajectory[-1, 6:10] @ ref.states[i + 1, 6:10] < 0:
            error[6:10] = trajectory[-1, 6:10] + ref.states[i + 1, 6:10]
        defect = float(np.max(np.abs(error)))
        maximum_defect = max(maximum_defect, defect)
        gate = int(np.searchsorted(ref.gate_indices, i, side="right"))
        plane = (trajectory[:, :3] - course.centers[gate]) @ course.normals[gate]
        # A dense state can approach the event from a slightly different side
        # because the shooting model uses a coarser integrator. Only tolerate
        # endpoint discrepancies within the declared dynamics tolerance.
        early_crossing = np.any(plane[:-1] > config.dynamics_tolerance)
        limits = (
            np.any(
                np.linalg.norm(trajectory[:, 3:6], axis=-1)
                > config.max_speed_m_s + config.dynamics_tolerance
            )
            or np.any(
                np.abs(trajectory[:, 10:13])
                > config.max_body_rate_rad_s + config.dynamics_tolerance
            )
            or np.any(trajectory[:, 13:] < model.rpm_min / model.rpm_hover - 1e-4)
            or np.any(trajectory[:, 13:] > model.rpm_max / model.rpm_hover + 1e-4)
        )
        if (
            all_collision[i]
            or bounds
            or early_crossing
            or limits
            or defect > config.dynamics_tolerance
            or not np.isfinite(trajectory).all()
        ):
            bad.append(i)
    for gate, index in enumerate(ref.gate_indices):
        p = ref.states[index, :3] - course.centers[gate]
        up = np.cross(course.normals[gate], course.right_axes[gate])
        if (
            abs(p @ course.normals[gate]) > 1e-3
            or abs(p @ course.right_axes[gate])
            > course.widths[gate] / 2 - margin + 1e-4
            or abs(p @ up) > course.heights[gate] / 2 - margin + 1e-4
            or ref.states[index, 3:6] @ course.normals[gate] <= 0
        ):
            bad.append(int(index) - 1)
    return {
        "validated": not bad,
        "invalid_intervals": sorted(set(bad)),
        "maximum_dynamics_defect": maximum_defect,
        "minimum_clearance_m": (
            minimum_clearance if np.isfinite(minimum_clearance) else None
        ),
    }


def optimize_reference(course, model, config=PlannerConfig(), radius=0.15):
    import casadi as ca

    validate_openings(course, radius, config.tracking_margin_m)
    if (
        config.nodes_per_segment < 2
        or config.integration_substeps < 1
        or config.validation_hz < round(1 / model.physics_dt)
    ):
        raise ValueError(
            "Invalid planner discretization; validation must run at least at the physics rate"
        )
    boxes = frame_boxes(course, config.frame_depth_m)
    margin = radius + config.tracking_margin_m
    previous = None
    for refinement in range(config.max_refinements + 1):
        nodes = config.nodes_per_segment * 2**refinement
        time, guess, commands, gates = initial_guess(course, model, nodes)
        if previous is not None:
            boundaries = np.r_[0, previous.time[previous.gate_indices]]
            time = np.concatenate(
                [
                    np.linspace(boundaries[i], boundaries[i + 1], nodes, endpoint=False)
                    for i in range(course.num_gates)
                ]
                + [boundaries[-1:]]
            )
            sampled, commands, _ = previous.sample(jnp.asarray(time))
            guess, commands = np.asarray(sampled), np.asarray(commands)
        n = len(time) - 1
        print(
            f"Building trajectory problem: {n} intervals, refinement {refinement}",
            flush=True,
        )
        opt = ca.Opti()
        x = opt.variable(17, n + 1)
        u = opt.variable(4, n)
        duration = opt.variable(course.num_gates)
        opt.subject_to(opt.bounded(0.1, duration, 30.0))
        opt.subject_to(opt.bounded(-1, u, 1))
        opt.subject_to(x[:, 0] == guess[0])
        opt.subject_to(
            opt.bounded(
                model.rpm_min / model.rpm_hover,
                x[13:17, :],
                model.rpm_max / model.rpm_hover,
            )
        )
        opt.subject_to(ca.sum1(x[3:6, :] ** 2) <= config.max_speed_m_s**2)
        opt.subject_to(
            opt.bounded(
                -config.max_body_rate_rad_s, x[10:13, :], config.max_body_rate_rad_s
            )
        )
        step = model.symbolic_step(config.integration_substeps)
        for k in range(n):
            opt.subject_to(
                x[:, k + 1] == step(x[:, k], u[:, k], duration[k // nodes] / nodes)
            )
        for k in range(n + 1):
            opt.subject_to(
                opt.bounded(
                    course.bounds_min + margin, x[:3, k], course.bounds_max - margin
                )
            )
            # Choose a separating face from the safe route seed. This defines
            # a conservative clearance corridor and avoids a nonsmooth choice
            # between six obstacle faces inside the nonlinear flight solve.
            planes = {}
            for center, basis, half in zip(boxes.centers, boxes.axes, boxes.half_sizes):
                local = basis @ (guess[k, :3] - center)
                face = int(np.argmax(np.abs(local) - half - margin))
                sign = 1.0 if local[face] >= 0 else -1.0
                normal = sign * basis[face]
                key = tuple(float(v) for v in normal)
                lower = float(normal @ center + half[face]) + margin + 0.001
                planes[key] = max(planes.get(key, -np.inf), lower)
            for normal, lower in planes.items():
                opt.subject_to(ca.dot(ca.DM(normal), x[:3, k]) >= lower)
        for gate, k in enumerate(gates):
            relative = x[:3, int(k)] - course.centers[gate]
            normal, right = course.normals[gate], course.right_axes[gate]
            up = np.cross(normal, right)
            opt.subject_to(ca.dot(relative, normal) == 0)
            opt.subject_to(
                opt.bounded(
                    -course.widths[gate] / 2 + margin,
                    ca.dot(relative, right),
                    course.widths[gate] / 2 - margin,
                )
            )
            opt.subject_to(
                opt.bounded(
                    -course.heights[gate] / 2 + margin,
                    ca.dot(relative, up),
                    course.heights[gate] / 2 - margin,
                )
            )
            opt.subject_to(ca.dot(x[3:6, int(k)], normal) >= 0.3)
            for j in range(gate * nodes, int(k)):
                # The previous opening of a stacked gate is on this same
                # plane. Permit departure from it before requiring an approach
                # from the negative side of the next opening.
                limit = 0.0 if j == gate * nodes else -1e-5
                opt.subject_to(ca.dot(x[:3, j] - course.centers[gate], normal) <= limit)
        minimum_time_objective = ca.sum1(duration) + 1e-4 * ca.sumsqr(
            u[:, 1:] - u[:, :-1]
        )
        opt.minimize(minimum_time_objective)
        opt.set_initial(x, guess.T)
        opt.set_initial(u, commands[:-1].T)
        opt.set_initial(duration, np.diff(np.r_[0, time[gates]]))
        solver_options = {
            "max_iter": config.max_iterations,
            "tol": 1e-6,
            "acceptable_tol": 1e-3,
            "acceptable_constr_viol_tol": 1e-5,
            "acceptable_obj_change_tol": 1e-5,
            "acceptable_iter": 10,
            "print_level": 0,
            "hessian_approximation": "limited-memory",
        }
        opt.solver("ipopt", {"expand": True}, solver_options)
        try:
            if course.num_gates > 2 and previous is None:
                print("Finding a dynamically feasible warm start...", flush=True)
                opt.minimize(
                    ca.sumsqr(x[:3, :] - guess[:, :3].T)
                    + 0.1 * ca.sumsqr(duration - np.diff(np.r_[0, time[gates]]))
                    + 0.01 * ca.sumsqr(u[:, 1:] - u[:, :-1])
                )
                opt.solver(
                    "ipopt",
                    {"expand": True},
                    {**solver_options, "max_iter": min(300, config.max_iterations)},
                )
                # This stage initializes the final optimization; it need not
                # minimize its arbitrary tracking objective to convergence.
                # Only the final solve plus dense validation can emit a plan.
                feasible = opt.solve_limited()
                violation = (
                    opt.stats().get("iterations", {}).get("inf_pr", [np.inf])[-1]
                )
                if violation > 0.01 or not np.isfinite(feasible.value(opt.x)).all():
                    raise RuntimeError(
                        f"Warm-start constraint violation is {violation:.3g}"
                    )
                print(f"Warm-start constraint violation: {violation:.3g}", flush=True)
                opt.set_initial(opt.x, feasible.value(opt.x))
                opt.minimize(minimum_time_objective)
                opt.solver("ipopt", {"expand": True}, solver_options)
            print("Solving minimum-time trajectory...", flush=True)
            solution = opt.solve_limited()
        except RuntimeError as exc:
            stats = opt.stats()
            iterations = stats.get("iterations", {})
            residuals = iterations.get("inf_pr", [float("nan")])
            raise RuntimeError(
                f"Trajectory optimizer failed ({stats.get('return_status')}, final constraint violation {residuals[-1]:.3g}); no reference was accepted. Inspect gate feasibility or increase planner iterations/nodes."
            ) from exc
        durations = solution.value(duration).reshape(-1)
        time = np.r_[0, np.cumsum(np.repeat(durations / nodes, nodes))]
        states = solution.value(x).T
        actions = solution.value(u).T
        actions = np.vstack((actions, actions[-1]))
        metadata = {
            "schema": 1,
            "course_fingerprint": fingerprint(course),
            "model_fingerprint": model.fingerprint,
            "planner_fingerprint": fingerprint(config),
            "planner": asdict(config),
            "vehicle_radius_m": radius,
            "solver": "ipopt",
            "solver_status": opt.stats().get("return_status"),
            "solver_converged": bool(opt.stats().get("success", False)),
            "solver_iterations": int(opt.stats().get("iter_count", 0)),
            "refinement": refinement,
            "validated": False,
        }
        ref = Reference(time, states, actions, gates, metadata)
        print("Validating at physics rate...", flush=True)
        if not metadata["solver_converged"]:
            print(
                "Optimizer reached its iteration/time limit. The candidate must still pass every dense feasibility check; it is not a converged optimum.",
                flush=True,
            )
        result = validate_reference(ref, course, model, config, radius)
        previous = ref
        if result["validated"]:
            ref.metadata.update(result)
            return ref
        print(
            f'Refining plan: {len(result["invalid_intervals"])} invalid intervals; maximum defect {result["maximum_dynamics_defect"]:.4g}',
            flush=True,
        )
    raise RuntimeError(
        f"Dense feasibility validation failed after refinement: {result}"
    )
