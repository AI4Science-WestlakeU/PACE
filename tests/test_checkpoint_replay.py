import hashlib
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch

from scripts.evaluate_checkpoint import evaluate
from src.methods.mfm.networks import VelocityNet

ROOT = Path(__file__).resolve().parents[1]


def test_stage2_replay_needs_no_historical_stage1_checkpoint(tmp_path, monkeypatch):
    """A warm-start config must remain usable for velocity-only inference."""
    monkeypatch.delenv('PACE_STAGE1_CHECKPOINT', raising=False)
    with initialize_config_dir(config_dir=str(ROOT / 'configs_hydra'), version_base='1.3'):
        cfg = compose(config_name='train', overrides=[
            'experiment=oceans_positions_pace_ode', 'samples_per_timepoint=12',
        ])
    cfg.load_stage1_ckpt = '${oc.env:PACE_STAGE1_CHECKPOINT}'
    config_path = tmp_path / 'historical.yaml'
    OmegaConf.save(cfg, config_path)
    model = VelocityNet(dim=cfg.dim, hidden_dims=list(cfg.hidden_dims_flow),
                        activation=cfg.activation_flow, batch_norm=False)
    checkpoint_path = tmp_path / 'velocity.ckpt'
    torch.save({'epoch': 0, 'state_dict': {
        'flow_net.' + key: value for key, value in model.state_dict().items()
    }}, checkpoint_path)

    record = evaluate(config_path, checkpoint_path, ot_max_iter=1000000)

    assert record['checkpoint_sha256'] == hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    assert {row['label'] for row in record['metrics']} == {1, 3, 5, 7}
    assert all(row['n_prediction'] == row['n_reference'] == 12 for row in record['metrics'])
    assert all(0 <= row['w1'] <= row['w2'] + 1e-6 for row in record['metrics'])
    assert not any('numItermax' in warning for warning in record['warnings'])
    # Reading for inference must not rewrite the historical training recipe.
    assert '${oc.env:PACE_STAGE1_CHECKPOINT}' in config_path.read_text()
