"""Versioned policy action semantics shared by training and evaluation."""

ACTION_SPACE = "motor_rpm_hover_centered_v1"
ACTION_NAMES = ("motor_1", "motor_2", "motor_3", "motor_4")


def validate_checkpoint_actions(
    payload: dict,
    controller: str = "direct_motor",
    reference_fingerprint: str | None = None,
) -> None:
    expected = ACTION_SPACE
    if controller == "residual_mpc":
        from a2rl_drone_training.hierarchical.environment import RESIDUAL_ACTION_SPACE

        expected = RESIDUAL_ACTION_SPACE
    if (
        payload.get("controller", "direct_motor") != controller
        or payload.get("action_space") != expected
    ):
        raise ValueError(
            "Checkpoint action space is incompatible with the selected controller. "
            "Attitude-control checkpoints cannot be resumed or evaluated as motor "
            "policies. Start a fresh run in a new checkpoint directory."
        )
    if controller == "residual_mpc" and (
        payload.get("observation_schema") != "residual_reference_v1"
        or payload.get("reference_fingerprint") != reference_fingerprint
    ):
        raise ValueError(
            "Checkpoint reference or observation schema is incompatible with residual MPC"
        )
