
from stable_baselines3.common.base_class import BaseAlgorithm

def print_model_params(model : BaseAlgorithm):
    for name, param in model.policy.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.shape}")


from tminterface.structs import CheckpointData, SimStateData, CheckpointTime
from bytefield import ArrayField


def print_sim_state(ssD : SimStateData) -> None:
    print("\n=== Simulation State Snapshot ===")
    print(f"    Time:                {ssD.time} ms")
    print(f"    Position:            {ssD.position}")
    print(f"    Velocity:            {ssD.velocity}")
    print(f"    Rotation (YawPitchRoll): {ssD.yaw_pitch_roll}")
    print(f"    Display Speed:       {ssD.display_speed} units")
    
    print("\nInput State:")
    print(f"    Accelerate:          {ssD.input_accelerate}")
    print(f"    Brake:               {ssD.input_brake}")
    print(f"    Left:                {ssD.input_left}")
    print(f"    Right:               {ssD.input_right}")
    print(f"    Steer (analog):      {ssD.input_steer}")
    print(f"    Gas (analog):        {ssD.input_gas}")
    
    print("\nRace Progress:")
    print(f"    Race Time:           {ssD.race_time} ms")
    print(f"    Rewind Time:         {ssD.rewind_time} ms")
    print(f"    Num Respawns:        {ssD.num_respawns}")

    cpData : CheckpointData = ssD.cp_data
    cp_times_structs : ArrayField[CheckpointTime] = cpData.cp_times  # This is likely a list of structs

    cp_times = [cp_times_structs[i].time for i in range(cpData.cp_times_length)]
    print(f"   Checkpoint Times:    {cp_times}")
    print(f"    Checkpoints Passed:  {cpData.cp_states_length}")