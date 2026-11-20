from env.robot_manager.robot_config._g1d_dexhands import get_config

NATIVE_CONTACT_LINKS = {
    side: [prefix + finger + "_distal" for finger in ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky")]
    for side, prefix in (("left", "l_"), ("right", "r_"))
}


def get_robot_config():
    return get_config("g1d_wuji_hand2_1")
