from gym.envs.registration import register

register(
    id='parking_lot-v0',
    entry_point='gym_parking_lot.envs:ParkingLotEnv',
)