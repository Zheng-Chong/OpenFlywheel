from env.environment.task_env import TaskEnv
from task.RoboDojo.tasks.g1d_pour_beans_half_cup import G1dPourBeansHalfCupCommon

# how the platform's 倒半杯 instructions name the beans
RED = {"zh": "红豆", "en": "red beans"}


class G1dPourBeansHalfCupRigidCommon(G1dPourBeansHalfCupCommon):
    """倒半杯 with rigid beans: g1d_pour_beans_half_cup, its beans one convex rigid body each (GranularObject) instead
    of PBD spheres, of the layout's kind (red, mung or black beans, each grain its own size; layouts:
    prepare_pour_half_layouts.py --rigid). The same success check, the cup's capacity counted in the kind's grains.
    """

    def gen_instruction(self, env_idx):
        """倒半杯's instructions, the beans named as the layout's kind."""
        layout = self.scene_manager.layout_manager.saved_layouts[env_idx] or {}
        beans = next(iter(layout.get("Fluid", {}).values()), [{}])[0].get("bean") or RED
        out = []
        for s in super().gen_instruction(env_idx):
            lang = "zh" if RED["zh"] in s else "en"
            out.append(s.replace(RED[lang], beans[lang]))
        return out


class g1d_pour_beans_half_cup_rigid(G1dPourBeansHalfCupRigidCommon, TaskEnv):
    pass
