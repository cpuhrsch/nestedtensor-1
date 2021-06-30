import torch
import numpy as np
import time
import random
import nestedtensor
from classy_vision.models import build_model


@torch.inference_mode()
def benchmark_torch_function(iters, f, *args, **kwargs):
    f(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    else:
        t0 = time.time()
    for _ in range(iters):
        f(*args, **kwargs)
    if torch.cuda.is_available():
        end_event.record()
        torch.cuda.synchronize()
        return start_event.elapsed_time(end_event)
    else:
        return (time.time() - t0)


@torch.inference_mode()
def run_benchmark(iters, shapes, model, model_name, bsz, include_loop):
    ts = []
    for s in shapes:
        inp = torch.randn(*s, dtype=torch.half).cuda()
        ts.append(inp)
    ts_nt = nestedtensor.nested_tensor([t.squeeze(0) for t in ts], device=torch.device('cuda'), dtype=torch.half)
    ts_padded = ts_nt.to_padded_tensor()

    def _loop():
        model_outputs = []
        for inp in ts:
            model_outputs.append(model(inp))
        return model_outputs

    def _padded():
        return model(ts_padded)

    # Test
    outputs_nt = model(ts_nt)
    # import time; time.sleep(1)
    # outputs_nt = model(ts_nt)
    # import sys; sys.exit(1)
    model_outputs = _loop()
    for mo, ntmo in zip(model_outputs, outputs_nt.unbind()):
        # Using float16 tolerances from torch/testing/_core.yp
        assert torch.allclose(mo.squeeze(0), ntmo, rtol=1e-3, atol=1e-3)

    if include_loop:
        loop_time = benchmark_torch_function(iters, _loop)
    else:
        padded_time = benchmark_torch_function(iters, _padded)
    nt_time = benchmark_torch_function(iters, lambda: model(ts_nt))

    shapes_2_array = np.array([s[2] for s in shapes])
    shapes_3_array = np.array([s[3] for s in shapes])
    print(f"model_name: {model_name.rjust(18)},", end='')
    print(f" bsz: {bsz:3.0f},", end='')
    print(f" mean±std shapes[2]: {shapes_2_array.mean():.2f}±{shapes_2_array.std():.2f},", end='')
    print(f" mean±std shapes[3]: {shapes_3_array.mean():.2f}±{shapes_3_array.std():.2f},", end='')
    print(f" padded_size: {tuple(ts_padded.size())},", end='')
    print(f" nt: {nt_time / iters:7.2f}ms,", end='')
    if include_loop:
        print(f" loop: {loop_time / iters:7.2f}ms,", end='')
        print(f" speedup: {loop_time / nt_time:.2f}x")
    else:
        print(f" padded: {padded_time / iters:7.2f}ms,", end='')
        print(f" speedup: {padded_time / nt_time:.2f}x")


def benchmark(model_name, bsz, min_size, include_loop):
    iters = 20

    model = build_model({"name": model_name})
    model = model.cuda().half().eval()
    random.seed(123)
    shapes = []
    for i in range(bsz):
        H = random.randint(100, 288)
        W = 288
        if i % 2:
            H, W = W, H
        shapes.append((1, 3, H, W))
    print(shapes)
    run_benchmark(iters, shapes, model, model_name, bsz, include_loop)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='ClassyVision benchmark')
    parser.add_argument('model',
                        choices=["resnext101_32x4d", "regnet_y_128gf"],
                        help='Which model to benchmark.')
    parser.add_argument('bsz', metavar='bsz', type=int,
                        help='Batch size to consider')
    parser.add_argument('min_size', metavar='min_size', type=int,
                        help='Minimum image size to sample.')
    parser.add_argument('--include_loop', action='store_true',
                        help='Include outer loop as baseline.')
    args = parser.parse_args()
    benchmark(args.model, args.bsz, args.min_size, args.include_loop)
