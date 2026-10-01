from comfy_api.latest import ComfyExtension

from .nodes import TurboBaseTrajectoryMixer


class TrajectoryMixerExtension(ComfyExtension):
    async def get_node_list(self):
        return [TurboBaseTrajectoryMixer]


async def comfy_entrypoint():
    return TrajectoryMixerExtension()
