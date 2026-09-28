"""Load locally trained PACE/MFM checkpoints and integrate their velocity field.

Checkpoint loading uses Python pickle for Lightning metadata: load only files
from your own training runs or a trusted source.
"""
from pathlib import Path
import torch
from omegaconf import OmegaConf
from src.methods.mfm.networks import VelocityNet
from src.methods.mfm.ema import EMA
from src.methods.mfm.eval_metrics import _run_ode_rollout


def load_run(run_dir, device='cpu'):
    run_dir=Path(run_dir)
    cfg=OmegaConf.load(run_dir/'resolved_config.yaml')
    folder=run_dir/'checkpoints/stage2_flow'
    checkpoints=list(folder.glob('*.ckpt'))
    if not checkpoints:
        raise FileNotFoundError(f'No Stage 2 checkpoint in {folder}; train this experiment first.')
    checkpoint=folder/'last.ckpt'
    if not checkpoint.exists():
        checkpoint=max(checkpoints,key=lambda p:p.stat().st_mtime)
    model=VelocityNet(dim=cfg.dim,hidden_dims=list(cfg.hidden_dims_flow),activation=cfg.activation_flow,batch_norm=False)
    if cfg.get('ema_decay') is not None:
        model=EMA(model,decay=cfg.ema_decay)
    raw=torch.load(checkpoint,map_location='cpu',weights_only=False)['state_dict']
    state={k.removeprefix('flow_net.'):v for k,v in raw.items() if k.startswith('flow_net.')}
    model.load_state_dict(state,strict=True)
    model.to(device).eval()
    return model,cfg,checkpoint


def rollout(model, source, t_start=0.0, t_end=1.0, steps=101, device='cpu'):
    """Euler integration on the same normalized time scale used in training.

    Returns [steps, cells, dimensions]. Input must use the training coordinates
    (including the training-only StandardScaler for biological datasets).
    """
    if steps < 2 or t_end <= t_start:
        raise ValueError('Need steps >= 2 and t_end > t_start.')
    return _run_ode_rollout(model, torch.as_tensor(source,dtype=torch.float32),
                          t_start,t_end,steps,device).detach().cpu()
