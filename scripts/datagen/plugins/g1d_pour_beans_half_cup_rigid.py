"""scripts/datagen plugin for g1d_pour_beans_half_cup_rigid: g1d_pour_beans_half_cup's (倒半杯) with rigid beans of the
layout's kind; the target's capacity counted in their grains (the Episode's `grain`, kept in the quality record).
"""

from scripts.datagen.plugins.g1d_pour_beans_half_cup import (  # noqa: F401
    ACCEPTANCE,
    HOLD,
    STEP_LIM,
    TAIL,
    Episode,
    audit,
)

TASK = "g1d_pour_beans_half_cup_rigid"
