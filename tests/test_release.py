from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from src.methods.pace.interpolant import compute_interpolant_and_velocity
from src.methods.pace.flow_matcher import PACEFlowMatcher
from src.dataloaders.balanced_timepoint_data import BalancedTimepointDataModule
ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted(p.relative_to(ROOT/'configs_hydra/experiment').with_suffix('').as_posix() for p in (ROOT/'configs_hydra/experiment').rglob('*.yaml'))

@pytest.mark.parametrize('experiment', CONFIGS)
def test_experiment_resolves_and_data_exists(experiment):
    with initialize_config_dir(config_dir=str(ROOT/'configs_hydra'), version_base='1.3'):
        cfg = compose(config_name='train', overrides=[f'experiment={experiment}'])
    assert cfg.method.name in {'pace','mfm'}
    assert (ROOT/cfg.data_path).is_file()
    assert not set(cfg.train_timepoint_labels) & set(cfg.test_timepoint_labels)

class CurvedBridge(torch.nn.Module):
    def forward(self, x0, x1, s):
        return s.square().expand_as(x0) + x1 - x0

def test_bridge_endpoints_and_autodiff_velocity():
    model = CurvedBridge()
    x0,x1 = torch.randn(8,3),torch.randn(8,3)
    for s,end in [(0.,x0),(1.,x1)]:
        x,_ = compute_interpolant_and_velocity(model,x0,x1,torch.full((8,1),s))
        torch.testing.assert_close(x,end)
    s=torch.full((8,1),.37)
    _,v=compute_interpolant_and_velocity(model,x0,x1,s)
    xp,_=compute_interpolant_and_velocity(model,x0,x1,s+1e-3)
    xm,_=compute_interpolant_and_velocity(model,x0,x1,s-1e-3)
    torch.testing.assert_close(v,(xp-xm)/2e-3,rtol=1e-3,atol=2e-4)

def test_velocity_converts_local_to_physical_time():
    x0,x1=torch.randn(8,2),torch.randn(8,2)
    _,_,u=PACEFlowMatcher(sigma=0).sample_location_and_conditional_flow(x0,x1,.2,.6,t=torch.full((8,),.4))
    torch.testing.assert_close(u,(x1-x0)/.4)

class SnapshotData(BalancedTimepointDataModule):
    def _load_timepoint_frames(self):
        return {0:np.array([[0.,0.],[2.,2.]]), 1:np.full((2,2),1e6), 2:np.array([[4.,4.],[6.,6.]])}

def test_holdout_does_not_affect_standardization():
    dm=SnapshotData(SimpleNamespace(data_path='unused',dim=2,whiten=True,split_ratios=[1,0],train_timepoint_labels=[0,2],test_timepoint_labels=[1]))
    dm.setup()
    np.testing.assert_allclose(dm.scaler.mean_,[3.,3.])
    assert set(dm.train_frames)=={0,2}
    assert set(dm.test_frames)=={1}
