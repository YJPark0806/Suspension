from gym.envs.registration import register

register(
    id="VehicleSuspension-v0",
    entry_point="scripts.VehicleEnv:VehicleEnv",  # VehicleEnv 클라스 지정
)
