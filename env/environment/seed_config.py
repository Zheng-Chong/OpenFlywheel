from collections.abc import Sequence, Sequence as SequenceABC
from typing import Any

from env.seeding import seed_everywhere


class SeedConfigMixin:
    """Global or per-env seeds of an env (sim config `seed`, or the seed list of a reset); every backend's BaseEnv."""

    def update_seed(self, seed: Any | None = None):
        global_seed, _ = self._process_seed_input(seed)
        if global_seed is not None:
            seed_everywhere(global_seed)  # Set seed for all envs

    def _configure_seed_config(self, seed_cfg: Any):
        if seed_cfg is None:
            self._global_seed = None
            self._seed_list = None
            self.env_seeds = None
            return
        if self._is_sequence_seed(seed_cfg):
            seeds = [int(s) for s in seed_cfg]
            self._set_env_seeds_full(seeds)
        else:
            self._set_global_seed(int(seed_cfg))

    def get_env_seed(self, env_id: int) -> int | None:
        if self._seed_list is not None:
            if env_id >= len(self._seed_list):
                raise IndexError(f"Requested env_id {env_id} exceeds configured seeds (len={len(self._seed_list)}).")
            return self._seed_list[env_id]
        return self._global_seed

    def _set_env_seed(self, env_id: int):
        seed = self.get_env_seed(env_id)
        if seed is not None:
            seed_everywhere(seed)

    @staticmethod
    def _is_sequence_seed(seed: Any) -> bool:
        return isinstance(seed, SequenceABC) and not isinstance(seed, str | bytes)

    def _set_global_seed(self, seed_value: int):
        self._global_seed = int(seed_value)
        if self.env_seeds is not None:
            self.env_seeds = [self._global_seed for _ in self.env_seeds]
            self._seed_list = list(self.env_seeds)
        else:
            self._seed_list = None

    def _set_env_seeds_full(self, seeds: Sequence[int]):
        num_envs = int(self.config.scene.num_envs)
        seed_list = [int(s) for s in seeds]
        if len(seed_list) != num_envs:
            raise ValueError(f"Seed list length {len(seed_list)} does not match num_envs {num_envs}.")
        self.env_seeds = list(seed_list)
        self._seed_list = list(seed_list)
        self._global_seed = seed_list[0] if seed_list else None

    def _process_seed_input(self, seed: Any) -> tuple[int | None, list[int] | None]:
        if seed is None:
            return None, self.env_seeds if self.env_seeds is not None else None

        if self._is_sequence_seed(seed):
            self._set_env_seeds_full(seed)
            return (
                self._global_seed,
                list(self.env_seeds) if self.env_seeds is not None else None,
            )

        self._set_global_seed(int(seed))
        return self._global_seed, None
