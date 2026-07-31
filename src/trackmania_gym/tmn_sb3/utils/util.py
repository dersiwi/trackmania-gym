import os
from stable_baselines3.common.base_class import BaseAlgorithm

def save_model(model : BaseAlgorithm, run_dir : str) -> None:
    """Saves given model in given run-directory. Also saves replay buffer if the model has one."""
    steps_trained = model.num_timesteps

    savepoint_dir = os.path.join(run_dir, "savepoint")
    os.makedirs(savepoint_dir, exist_ok=True)

    model_save_path = model_save_path or os.path.join(savepoint_dir, "model.zip")
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

        # Save the replay buffer if it exists
    if hasattr(model, 'save_replay_buffer'):
        replay_buffer_save_path = replay_buffer_save_path or os.path.join(savepoint_dir, "replay_buffer.pkl")
        model.save_replay_buffer(replay_buffer_save_path)
        print(f"Replay buffer saved to {replay_buffer_save_path}")

    return steps_trained, model_save_path, replay_buffer_save_path
