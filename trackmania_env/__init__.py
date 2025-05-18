from gymnasium.envs.registration import register
register(
    id = "TMNF_Single_Agent_ENV_v0",
    entry_point= "trackmania_env.envs:TMNF_Single_Agent_Env"
)