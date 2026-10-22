"""Shards partition the items exactly (no overlap, nothing lost) and use separate folders."""
import pytest

from src import pipeline
from src.config import Config, DataCfg
from src.data import items_path


@pytest.mark.skipif(not items_path("dev").exists(), reason="run prepare-data first")
def test_shards_partition_items():
    def cfg(shard, n):
        return Config(run_name="x", backend="hf", model_id="m", mode="nothink",
                      data=DataCfg(splits=["dev"], hops=[3, 4], limit_per_split_hop=25, shard=shard, num_shards=n))
    full = [it["id"] for it in pipeline.run_items(cfg(0, 1))]
    parts = [[it["id"] for it in pipeline.run_items(cfg(s, 4))] for s in range(4)]
    assert sorted(sum(parts, [])) == sorted(full)                 # nothing lost
    assert len(set(sum(parts, []))) == len(full)                   # no overlap
    assert len({cfg(s, 4).run_dir for s in range(4)} | {cfg(0, 1).run_dir}) == 5   # separate folders
