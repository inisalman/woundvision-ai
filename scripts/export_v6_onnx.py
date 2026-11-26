"""Export V6 Two-Stage models (Stage 1 ResNet18 + Stage 2 ResNet34) to ONNX and verify parity."""

from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageOps
import torch
import segmentation_models_pytorch as smp
import onnx
import onnxruntime as ort

PROJECT_DIR = Path(__file__).resolve().parent.parent
STAGE1_CKPT = PROJECT_DIR / "runs" / "v6_stage1_resnet18" / "best_stage1.pth"
STAGE2_CKPT = PROJECT_DIR / "runs" / "v6_stage2_resnet34_384" / "best_stage2.pth"

STAGE1_ONNX = PROJECT_DIR / "models" / "woundvision_v6_stage1_wound.onnx"
STAGE2_ONNX = PROJECT_DIR / "models" / "woundvision_v6_stage2_tissue.onnx"

STAGE1_SIZE = 320
STAGE2_SIZE = 320
NUM_CLASSES = 6
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


def export_model(model: torch.nn.Module, output_path: Path, size: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, size, size, dtype=torch.float32)
    torch.onnx.export(
        model,
        (dummy,),
        str(output_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        dynamo=False,
    )
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    onnx.save_model(onnx_model, str(output_path), save_as_external_data=False)
    mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Exported {output_path.name}: {mb:.2f} MB")


def verify_model(pt_model: torch.nn.Module, onnx_path: Path, size: int, batch_size: int = 1) -> None:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])

    rng = np.random.default_rng(42)
    x = rng.standard_normal((batch_size, 3, size, size)).astype(np.float32)
    with torch.inference_mode():
        pt_out = pt_model(torch.from_numpy(x)).numpy()

    t0 = time.perf_counter()
    ort_out = sess.run(["logits"], {"input": x})[0]
    ms = (time.perf_counter() - t0) * 1000.0
    diff = float(np.max(np.abs(pt_out - ort_out)))
    print(f"  Parity [{onnx_path.name} b={batch_size} @ {size}px]: max_diff={diff:.6f}, cpu_time={ms:.1f}ms")
    assert diff < 1e-3, f"Parity failed for {onnx_path.name}: {diff}"


def main() -> None:
    s1_ck = torch.load(STAGE1_CKPT, map_location="cpu", weights_only=True)
    s1 = smp.Unet(encoder_name=s1_ck["encoder"], encoder_weights=None, in_channels=3, classes=1, activation=None).eval()
    s1.load_state_dict(s1_ck["model_state"], strict=True)
    export_model(s1, STAGE1_ONNX, STAGE1_SIZE)
    verify_model(s1, STAGE1_ONNX, STAGE1_SIZE, batch_size=2)  # H-Flip TTA batch=2

    s2_ck = torch.load(STAGE2_CKPT, map_location="cpu", weights_only=True)
    s2 = smp.Unet(encoder_name=s2_ck["encoder"], encoder_weights=None, in_channels=3, classes=NUM_CLASSES, activation=None).eval()
    s2.load_state_dict(s2_ck["model_state"], strict=True)
    export_model(s2, STAGE2_ONNX, STAGE2_SIZE)
    verify_model(s2, STAGE2_ONNX, STAGE2_SIZE, batch_size=1)


if __name__ == "__main__":
    main()
