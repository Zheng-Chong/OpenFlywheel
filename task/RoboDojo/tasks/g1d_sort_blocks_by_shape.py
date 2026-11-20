from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.g1d_sort_blocks_by_color import SortBlocksG1dCommon


class g1d_sort_blocks_by_shape(SortBlocksG1dCommon, TaskEnv):
    """G1-D: sort the painted blocks into the two trays by the block shape on each tray's card."""

    BY = "shape"
