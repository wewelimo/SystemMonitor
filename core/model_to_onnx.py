import glob
import os

import onnxruntime
import torch

from core.networks.model import Model
from core.utils.frame_grabber import get_monitor_attributes_test


if __name__ == '__main__':
    checkpoint = "./lightning_logs/version_11/checkpoints"
    checkpoint = glob.glob(os.path.join(checkpoint, "*.ckpt"))[0]

    model = Model.load_from_checkpoint(checkpoint, strict=True)

    # TO ONNX
    filepath = "model.onnx"
    input_sample = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
    model.to_onnx(filepath, input_sample, export_params=True)
    ort_session = onnxruntime.InferenceSession(filepath)
    input_name = ort_session.get_inputs()[0].name
