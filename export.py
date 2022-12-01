import torch
import yaml
import os

from load_model import load_model


if __name__ == "__main__":
    model_dir  = "./saved_model/mvtec/MemSeg-bottle"
    onnx_path  = "./saved_model/mvtec/MemSeg-bottle/best_model.onnx"
    # load config
    cfg = yaml.load(open(os.path.join(model_dir, "config.yaml"),'r'), Loader=yaml.FullLoader)
    # load model
    model = load_model(cfg, model_dir)
    x = torch.ones(1, 3, *cfg["DATASET"]["resize"])

    torch.onnx.export(
        model,
        x,
        onnx_path,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
    )


