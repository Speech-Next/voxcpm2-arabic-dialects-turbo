"""Dense vs 2:4 sparse W8A8 FFN GEMMs at base_lm's shapes on this GPU (a script, not collected by pytest).

Every measurement is a CUDA graph of 28 calls (one per base_lm layer) cycling over 4 weight sets, so the weights come
from DRAM as in the model; median of the replays, per call. gate_up = the SwiGLU GEMM + the quantization of h (what
the FFN runs); down = the residual GEMM.

    python tests/gpu/bench_sparse24.py --rows 1,8,64,200,1300 [--configs] [--json out.json]
"""

import argparse
import json
import statistics

import torch

HIDDEN, INTER, LAYERS, SETS = 2048, 6144, 28, 4


def build(seed):
    from voxcpm_kernels import fused_layer as F
    from voxcpm_kernels import sparse24 as S
    from voxcpm_kernels.int8_linear import W8A8Linear

    g = torch.Generator(device="cuda").manual_seed(seed)
    down = W8A8Linear(S.prune_24(torch.randn(HIDDEN, INTER, device="cuda", generator=g)), None,
                      torch.rand(INTER, device="cuda", generator=g) + 0.5)
    gate_up = F.swiglu_weights(W8A8Linear(S.prune_24(torch.randn(2 * INTER, HIDDEN, device="cuda", generator=g))), down)
    return gate_up, down, S.Sparse24Linear(gate_up, S.swiglu_rows(2 * INTER, "cuda")), S.Sparse24Linear(down)


def time_graph(fn, reps=30):
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    for _ in range(5):
        graph.replay()
    times = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        graph.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1000 / LAYERS)
    return statistics.median(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="1,8,16,32,64,128,200,256,512,1024,1300,2030")
    ap.add_argument("--configs", action="store_true", help="every tile configuration, not only the default pick")
    ap.add_argument("--json")
    args = ap.parse_args()
    from voxcpm_kernels import fused_layer as F
    from voxcpm_kernels import sparse24 as S
    from voxcpm_kernels.int8_linear import quantize_rows

    sets = [build(i) for i in range(SETS)]
    out = []
    print("| rows | GEMM | dense us | sparse us (config) | speed-up |", flush=True)
    print("|---:|---|---:|---:|---:|", flush=True)
    for m in [int(r) for r in args.rows.split(",")]:
        xq = quantize_rows(torch.randn(m, HIDDEN, device="cuda", dtype=torch.bfloat16))
        hq = quantize_rows(torch.randn(m, INTER, device="cuda", dtype=torch.bfloat16))
        res = torch.randn(m, HIDDEN, device="cuda", dtype=torch.bfloat16)
        for name in ("gate_up", "down"):
            cfgs = [c for c in sorted(S.CONFIGS) if name == "gate_up" or c not in S.SWIGLU_ONLY] \
                if args.configs else [None]
            if name == "gate_up":
                dense = lambda: [F.swiglu_gemm_quant(*xq, sets[i % SETS][0], torch.bfloat16) for i in range(LAYERS)]
                sparse = lambda c: [S.swiglu_gemm_quant(*xq, sets[i % SETS][2], torch.bfloat16, cfg=c)
                                    for i in range(LAYERS)]
            else:
                dense = lambda: [F._gemm(*hq, sets[i % SETS][1], torch.bfloat16, residual=res) for i in range(LAYERS)]
                sparse = lambda c: [S.gemm_res(*hq, sets[i % SETS][3], torch.bfloat16, res, cfg=c)
                                    for i in range(LAYERS)]
            # dense and sparse alternate (power-capped clocks drift)
            td, ts = [], {c: [] for c in cfgs}
            for _ in range(2):
                td.append(time_graph(dense))
                for c in cfgs:
                    ts[c].append(time_graph(lambda: sparse(c)))
            d = min(td)
            per = {c: min(v) for c, v in ts.items()}
            best = min(per, key=per.get)
            pick = S._cfg(m, 2 * INTER if name == "gate_up" else HIDDEN, "swiglu" if name == "gate_up" else "res")
            shown = per[best] if args.configs else per[None]
            label = best if args.configs else pick
            print(f"| {m} | {name} | {d:.1f} | {shown:.1f} ({label}) | {d / shown:.2f}x |", flush=True)
            out.append({"rows": m, "gemm": name, "dense_us": d, "sparse_us": {str(k): v for k, v in per.items()},
                        "default_cfg": pick})
    if args.json:
        json.dump(out, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
