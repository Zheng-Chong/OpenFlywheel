from env.robot_manager.robot_config._g1d_dexhands import get_config

NATIVE_CONTACT_LINKS = {
    side: [side + "_" + finger + "_elastomer" for finger in ("thumb", "index", "middle", "ring", "pinky")]
    for side in ("left", "right")
}


def get_robot_config():
    return get_config("g1d_sharpa_wave")
