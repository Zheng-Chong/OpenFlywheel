from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.g1d_place_anything import PlaceAnythingG1dCommon


class PutAnythingInG1dCommon(PlaceAnythingG1dCommon):
    """G1-D: put the named object (target, X) into the named container (receptacle, Y) and let go: X's centre over
    its opening, below its rim; Y upright and where it stood (g1d_place_anything with a container)."""

    KIND = "container"


class g1d_put_anything_in(PutAnythingInG1dCommon, TaskEnv):
    pass
