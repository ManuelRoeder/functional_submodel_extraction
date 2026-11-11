# FedHOPE: source code

Source of the method in "Function-Aware Submodel Extraction for
Model-Heterogeneous Federated Learning" (ESANN 2027 submission). FedHOPE scores
every channel of the global model by its Hilbert-Schmidt norm, Eq. (3), sorts
each layer by this score once, and gives every client the top
$\lceil\beta_n K_i\rceil$ channels of each layer.

The package contains only the FedHOPE code. The federated training loop,
client, data pipeline and model are those of the public FedRolex code base and
are not included.

## Files

| File | Content | Paper |
|---|---|---|
| `hope_rank.py` | `channel_rankings`: channel scores, residual-stream groups, one order per group | Eq. (3); Alg. 1, lines 1-3 |
| `fedhope/hope/kernel.py` | `relu_second_moment`: the term $\kappa_{i,k}$ | Eq. (3) |
| `resnet_server_fedhope.py` | server classes that extract each client's submodel from the orders | Alg. 1, line 7 |
| `check_rearrangement.py` | numerical check of the bounds in Sec. 3.1 | Eq. (4), Prop. 1, Corollary (Eq. (5)), Remark |
| `fedhope/__init__.py`, `fedhope/hope/__init__.py` | package stubs | |

## Numerical check of Sec. 3.1

`python check_rearrangement.py` (standard library only, about one second):

- Eq. (4): the population-average deviation bound equals
  $\sum_r(1-\omega_i(r))\,c_{i,\pi_i(r)}$ for every order (Claim 2).
- Prop. 1: over all permutations, the order of decreasing score attains the
  minimum in every one of 400 random layers and capacity profiles (Claim 3),
  and $\omega_i(r)$ is non-increasing in $r$.
- Corollary and Remark: the expected excess of a uniformly random order
  equals Eq. (5), and it is zero when all scores are equal or every client
  keeps all channels (Claim 4).

In the script, $\omega_i(r)$ is `n`, the client weights $p_n$ are `q`, and
the capacity levels $\beta_n$ are `rates`.

## How the parts map to Algorithm 1

- **Line 1, score.** For every prunable convolution, `hope_rank._prunable_units`
  finds the BN that normalizes its output (in the pre-activation ResNet, the
  next BN on the forward path) and the modules that read that BN's output.
  `channel_rankings(model, "hope")` computes
  $c_{i,k}=\|v_{i,k}\|\sqrt{\kappa_{i,k}}$, with $\kappa_{i,k}$ from
  `relu_second_moment(b, |gamma|)` and $v_{i,k}$ the concatenated kernels of
  those modules (`_v_out`).
- **Line 2, residual streams.** The stem and every block-final convolution that
  write into the same residual stream form one group; a projection shortcut
  starts a new stream. The scores of a group are summed, so all its members
  receive one order.
- **Line 3, orders.** Each group is sorted by decreasing (summed) score.
- **Line 7, extraction.** `ResnetServerImportance.split_model` gives client $m$
  the first $\lceil\beta_m K_i\rceil$ entries of each order. The submodels are
  nested, as in HeteroFL.
- **Line 10, selective averaging.** Inherited unchanged from FedRolex's
  `ResnetServerRoll.combine` (not included).

In the paper's runs the global model is loaded from the pretrained checkpoint
before the server is built, and the server options keep their defaults
(`rank_warmup=0`, `rank_refresh=0`, `alloc="uniform"`). The server therefore
ranks once, in `__init__`, and keeps the orders for all rounds. The warm-up,
re-ranking and per-layer width options (the latter needs `layer_alloc.py`, not
included) are not used in the paper.

The baselines of Table 1 use the same code: criteria `slim` (BN scale), `l1`
(filter norm) and `randfix` (random-fixed), served by `ResnetServerSlim`,
`ResnetServerL1` and `ResnetServerRandFix`. HeteroFL and FedRolex are
upstream's `ResnetServerStatic` and `ResnetServerRoll`.

## Usage

```python
import sys
sys.path.insert(0, "fedhope_code")
from hope_rank import channel_rankings

orders = channel_rankings(global_model, criterion="hope")
# {conv weight key: permutation of its output channels, best first}
```

`global_model` is FedRolex's pre-activation ResNet-18 (`models/resnet.py`);
`_prunable_units` relies on its module names (`conv1`, `layer1`-`layer4`,
`n1`, `n2`, `n4`, `shortcut`, `linear`).

To use the server classes, place `resnet_server_fedhope.py`, `hope_rank.py`
and `fedhope/` next to FedRolex's `resnet_server.py` and construct
`ResnetServerHope` with the same arguments as the upstream servers.

## Requirements

- Scores and orders: Python 3.10 or later, PyTorch 2.0 or later.
- Server classes: additionally a FedRolex checkout
  (https://github.com/AIoT-MLSys-Lab/FedRolex, commit `8135109`, Apache
  License 2.0) with its dependencies, including `numpy` and `ray`.
