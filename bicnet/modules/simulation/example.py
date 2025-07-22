from robosuite.robots import register_robot_class
from robosuite.models.robots import Panda
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
import mujoco


@register_robot_class("WheeledRobot")
class MobilePanda(Panda):
    @property
    def default_base(self):
        return "OmronMobileBase"

    @property
    def default_arms(self):
        return {"right": "Panda"}

# Create environment
env = suite.make(
    env_name="Lift",
    robots="MobilePanda",
    controller_configs=load_composite_controller_config(controller="BASIC"),
    has_renderer=True,
    has_offscreen_renderer=False,
    render_camera="agentview",
    use_camera_obs=False,
    control_freq=20,
)

# Run the simulation, and visualize it
env.reset()
mujoco.viewer.launch(env.sim.model._model, env.sim.data._data)
