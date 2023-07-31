from gym.envs.registration import register


register(
    id='GoalGrid-v0',
    entry_point='envs.continuous_grid:GoalContinuousGrid',
)
