import torch

try:
    from fedhope.hope.kernel import relu_second_moment
except ImportError as exc:
    raise ImportError(
        "hope_rank needs the fedhope package so the criterion stays identical to "
        "the reference implementation. Install it with:\n"
        "    pip install -e /path/to/FedHOPE/code"
    ) from exc

# hope: Eq. (3); slim, l1, randfix: baselines of Table 1
CRITERIA = ("hope", "cap", "slim", "l1", "randfix")


# Eq. (3) and Algorithm 1, line 2
def _prunable_units(model):
    layers = [model.layer1, model.layer2, model.layer3, model.layer4]
    blocks = [(li, bi, b) for li, layer in enumerate(layers)
              for bi, b in enumerate(layer)]

    def _successor(li, bi):
        layer = layers[li]
        if bi + 1 < len(layer):
            nxt = layer[bi + 1]
        elif li + 1 < len(layers):
            nxt = layers[li + 1][0]
        else:
            return model.n4, [model.linear]
        consumers = [nxt.conv1]
        if hasattr(nxt, "shortcut"):
            consumers.append(nxt.shortcut)
        return nxt.n1, consumers

    first = blocks[0][2]
    stem_consumers = [first.conv1] + ([first.shortcut] if hasattr(first, "shortcut") else [])
    trunk = 0
    yield "conv1.weight", model.conv1, first.n1, stem_consumers, f"trunk{trunk}"

    for li, bi, block in blocks:
        if not hasattr(block, "conv2"):
            raise RuntimeError(
                "trunk grouping only handles the two-convolution Block; a "
                "Bottleneck writes the trunk from conv3, which this harness's "
                "split_model does not treat as prunable anyway."
            )
        if hasattr(block, "shortcut"):
            trunk += 1
        prefix = f"layer{li + 1}.{bi}"
        yield f"{prefix}.conv1.weight", block.conv1, block.n2, [block.conv2], f"{prefix}.conv1"
        bn, consumers = _successor(li, bi)
        yield f"{prefix}.conv2.weight", block.conv2, bn, consumers, f"trunk{trunk}"


# Eq. (3): v_{i,k}
def _v_out(consumers, n_channels):
    parts = []
    for mod in consumers:
        w = mod.weight.detach()
        if w.dim() == 4:
            parts.append(w.permute(1, 0, 2, 3).reshape(w.shape[1], -1))
        else:
            parts.append(w.t().reshape(w.shape[1], -1))
    parts = [p for p in parts if p.shape[0] == n_channels]
    if not parts:
        return None
    return torch.cat(parts, dim=1)


# Algorithm 1, lines 1-3
@torch.no_grad()
def channel_rankings(model, criterion="hope", seed=0):
    if criterion not in CRITERIA:
        raise ValueError(f"unknown criterion {criterion!r}, expected one of {CRITERIA}")

    scores: dict[str, "torch.Tensor"] = {}
    groups: dict[str, list[str]] = {}
    for layer_i, (key, conv, bn, consumers, group) in enumerate(_prunable_units(model)):
        w = conv.weight.detach()
        n = w.shape[0]
        groups.setdefault(group, []).append(key)

        if criterion == "randfix":
            g = torch.Generator().manual_seed(int(seed) * 1_000_003 + layer_i)
            scores[key] = torch.rand(n, generator=g)
            continue

        if criterion == "l1":
            score = w.reshape(n, -1).abs().sum(dim=1)
        elif criterion == "slim":
            score = _bn_gamma(bn, n).abs()
        else:
            # Algorithm 1, line 1; Eq. (3)
            v = _v_out(consumers, n)
            if v is None:
                raise RuntimeError(
                    f"could not build v_out for {key}: no consumer reads {n} channels. "
                    "The pre-activation mapping in _prunable_units does not fit this model."
                )
            gamma = _bn_gamma(bn, n)
            beta = _bn_beta(bn, n)
            kappa = relu_second_moment(beta, gamma.abs()).clamp_min(0.0)
            score = v.norm(dim=1) * torch.sqrt(kappa)

        scores[key] = score.detach().float().cpu()

    # Algorithm 1, lines 2-3
    rankings = {}
    for members in groups.values():
        total = scores[members[0]].clone()
        for k in members[1:]:
            if scores[k].shape != total.shape:
                raise RuntimeError(
                    f"trunk group {members} has mismatched widths "
                    f"{[tuple(scores[m].shape) for m in members]}; the grouping "
                    "in _prunable_units does not fit this architecture"
                )
            total += scores[k]
        order = torch.argsort(total, descending=True)
        for k in members:
            rankings[k] = order.clone()
    return rankings


def _bn_gamma(bn, n):
    if getattr(bn, "weight", None) is None:
        return torch.ones(n, device=_dev(bn), dtype=torch.float32)
    return bn.weight.detach().float()


def _bn_beta(bn, n):
    if getattr(bn, "bias", None) is None:
        return torch.zeros(n, device=_dev(bn), dtype=torch.float32)
    return bn.bias.detach().float()


def _dev(module):
    for p in module.parameters(recurse=False):
        return p.device
    return torch.device("cpu")
