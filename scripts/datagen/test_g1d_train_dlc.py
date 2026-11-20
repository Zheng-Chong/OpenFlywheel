"""The worker recipe round-trips without submitting a paid DLC job."""

from types import SimpleNamespace

from scripts.datagen import g1d_train_dlc as train


def test_worker_recipe_preserves_datasets_and_overrides(monkeypatch):
    args = SimpleNamespace(
        name="beans", init="/mnt/data/init model", openwam="/mnt/data/code", steps=60, save_steps=30,
        lr=0.0001, grad_accum=2, warmup_ratio=0.05, lr_min_ratio=0.01, prompt_cache=1024, workers=4,
        dataset=["/mnt/data/first set.tar", "/mnt/data/second"], extra=["training.test=true"],
    )  # fmt: skip
    captured = []
    monkeypatch.setattr(train, "cmd_worker", captured.append)
    train.main(["worker", *train.worker_argv(args, "/mnt/data/job", 8)])
    actual = captured[0]
    assert actual.dataset == args.dataset
    assert actual.batch_size == 8 and actual.grad_accum == 2
    assert actual.init == args.init and actual.extra == args.extra
    recipe = train.train_overrides(actual, train.Path("/root/local"))
    assert "training.batch_size=8" in recipe
    assert "training.gradient_accumulation_steps=2" in recipe
    assert recipe[-1] == "training.test=true"
    assert train.dataset_name(args.dataset[0]) == "first set"


def test_init_from_one_checkpoint_of_a_run_dir(tmp_path):
    run = tmp_path / "out" / "2026-11-05_21-49-09"
    (run / "tokenizer" / "google").mkdir(parents=True)
    (run / "tokenizer" / "google" / "spiece.model").write_bytes(b"t")
    for name in ("checkpoint_step_5000.safetensors", "checkpoint_step_6000.safetensors", "config.yaml"):
        (run / name).write_bytes(b"x")
    ckpt = str(run / "checkpoint_step_6000.safetensors")
    assert not train.init_found(ckpt)
    (run / "normalization_stats.npy").write_bytes(b"n")
    assert train.init_found(ckpt) and not train.init_found(str(run / "checkpoint_step_7000.safetensors"))
    train.fetch_init(ckpt, tmp_path / "init")
    got = sorted(str(p.relative_to(tmp_path / "init")) for p in (tmp_path / "init").rglob("*") if p.is_file())
    assert got == [
        "checkpoint_step_6000.safetensors",
        "config.yaml",
        "normalization_stats.npy",
        "tokenizer/google/spiece.model",
    ]


def test_dlc_defaults_keep_legacy_callers_compatible():
    args = SimpleNamespace(cpus=16, memory="64Gi", shm="8Gi", image="image", workspace="w", resource_id="r",
                           max_minutes=60, data_source="d")  # fmt: skip
    body = train.dlc_job("test", "echo test", 2, args)
    assert body["Priority"] == 9
    assert body["JobSpecs"][0]["ResourceConfig"]["GPU"] == "8"
    args.gpus_per_node, args.priority = 4, 3
    body = train.dlc_job("test", "echo test", 2, args)
    assert body["Priority"] == 3
    assert body["JobSpecs"][0]["ResourceConfig"]["GPU"] == "4"
