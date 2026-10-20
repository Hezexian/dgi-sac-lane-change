# Training stability (Section 4.4) and generalisation (Section 4.5)

This folder holds the code of the TD-target-clipping and mixed-traffic experiments exactly as it was run. It is kept separate from the `dgi_sac` package so that these results can be traced to the code that produced them; for new work, use the package in the repository root.

| Folder | Paper section | Main files |
|---|---|---|
| `td_clipping/` | 4.4.2 continuations from saved checkpoints; 4.4.3 training from scratch with TD-target clipping | `cf_continue.py`, `run_training.py`, `stabilizers.py` |
| `mixed_traffic/` | 4.5.1 evaluation at other traffic densities and with aggressive drivers; 4.5.2 mixed-traffic training | `mix_evaluate.py`, `run_training.py`, `traffic_mix.py` |

Both folders cover DGI-SAC (`dgi`), SAC (`sac`) and the capacity-matched MLP (`capacity`). These methods sample the replay buffer uniformly; `replay.py` also contains sampling branches from earlier development that these entry points do not use.

## Installation

Use Python 3.12 and a separate virtual environment. From this folder:

```bash
python -m pip install -r requirements.txt
```

This installs highway-env 1.10.2 and the package versions recorded for the released runs. GPU runs require a CUDA-enabled build of PyTorch 2.8.0.

## Usage

Run the commands from inside the corresponding folder. Each script lists its arguments with `--help`.

**TD-target clipping (`td_clipping/`)**

```bash
python run_training.py --variant dgi --stabilizer qclip --seed 26093036 --budget 500000 --run-dir runs/dgi_qclip_seed036 --run-id dgi_qclip_seed036
python cf_continue.py --help
```

`run_training.py --stabilizer qclip` trains from scratch with the TD target clipped (Section 4.4.3). `cf_continue.py` continues a saved training run from one of its checkpoints, either unchanged (`--stabilizer none`) or clipped (`--stabilizer qclip`); `--rng-reseed-offset 0` restores the checkpoint's random-number state and a positive value starts from a new random seed (Section 4.4.2).

**Mixed traffic (`mixed_traffic/`)**

```bash
python run_training.py --variant dgi --traffic-mix mix --seed 26093001 --budget 500000 --run-dir runs/dgi_mix_seed001 --run-id dgi_mix_seed001
python mix_evaluate.py --help
```

`--traffic-mix mix` samples the traffic density and driver behaviour for each training episode (Section 4.5.2). `mix_evaluate.py` evaluates a checkpoint at the traffic density set in the YAML file passed with `--config`; `--hetero-traffic aggressive` uses the aggressive drivers of Section 4.5.1.
