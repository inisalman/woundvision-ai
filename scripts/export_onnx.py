"""Export WoundVision AI V5-Augmented (U-Net ResNet34) to ONNX and verify parity."""

from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
import segmentation_models_pytorch as smp
import onnx
import onnxruntime as ort

PROJECT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = PROJECT_DIR / "runs" / "notebook_v5_augmented" / "best_model.pth"
OUTPUT_ONNX_PATH = PROJECT_DIR / "models" / "woundvision_v5_augmented.onnx"
VAL_LIST_PATH = PROJECT_DIR / "data" / "LUTSeg" / "val.txt"
IMAGES_DIR = PROJECT_DIR / "data" / "LUTSeg" / "Images"

IMAGE_SIZE = 320
NUM_CLASSES = 6
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


def load_pytorch_model() -> torch.nn.Module:
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=NUM_CLASSES,
        activation=None,
    )
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model


def export_onnx(model: torch.nn.Module, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32)

    torch.onnx.export(
        model,
        (dummy_input,),
        str(output_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        dynamo=False,
    )

    # Validate ONNX graph and ensure single-file storage
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    onnx.save_model(onnx_model, str(output_path), save_as_external_data=False)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Exported ONNX model to: {output_path} ({size_mb:.2f} MB)")


def preprocess_np(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))[None, ...]
    return (arr - MEAN) / STD


def verify_parity(pt_model: torch.nn.Module, onnx_path: Path) -> None:
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 2
    sess_opts.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(onnx_path), sess_options=sess_opts, providers=["CPUExecutionProvider"]
    )

    # Gather real validation images if present, else use random inputs
    sample_arrays = []
    if VAL_LIST_PATH.exists():
        lutseg_dir = VAL_LIST_PATH.parent
        lines = [line.strip() for line in VAL_LIST_PATH.read_text().splitlines() if line.strip()][:5]
        for line in lines:
            rel_img = line.split()[0]
            candidate = lutseg_dir / rel_img
            if candidate.exists():
                with Image.open(candidate) as img:
                    sample_arrays.append((candidate.name, preprocess_np(img)))

    if not sample_arrays:
        rng = np.random.default_rng(42)
        sample_arrays = [("random", rng.standard_normal((1, 3, IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32))]

    max_logit_diff = 0.0
    pixel_agreements = []
    ort_times = []

    for label, inp_np in sample_arrays:
        with torch.inference_mode():
            pt_logits = pt_model(torch.from_numpy(inp_np)).numpy()

        t0 = time.perf_counter()
        ort_logits = session.run(["logits"], {"input": inp_np})[0]
        ort_times.append((time.perf_counter() - t0) * 1000.0)

        diff = float(np.max(np.abs(pt_logits - ort_logits)))
        max_logit_diff = max(max_logit_diff, diff)

        pt_mask = pt_logits.argmax(axis=1)
        ort_mask = ort_logits.argmax(axis=1)
        agreement = float((pt_mask == ort_mask).mean() * 100.0)
        pixel_agreements.append(agreement)
        print(f"  [{label}] max_logit_diff={diff:.6f}, pixel_agreement={agreement:.4f}%, ort_time={ort_times[-1]:.1f}ms")

    print(
        f"Parity summary: max_logit_diff={max_logit_diff:.6f}, "
        f"mean_pixel_agreement={np.mean(pixel_agreements):.4f}%, "
        f"mean_ort_cpu_latency={np.mean(ort_times):.1f}ms"
    )
    assert max_logit_diff < 1e-3, f"ONNX parity check failed! max_logit_diff={max_logit_diff}"


if __name__ == "__main__":
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    model = load_pytorch_model()
    export_onnx(model, OUTPUT_ONNX_PATH)
    verify_parity(model, OUTPUT_ONNX_PATH)
