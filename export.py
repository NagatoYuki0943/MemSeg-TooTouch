import torch
from torch import nn, Tensor
import onnx
import yaml
import os

from load_model import load_model


def export_onnx(model: nn.Module, x: Tensor, onnx_path : str, simple: bool = True):
    """export model to onnx format.

    Args:
        model (nn.Module):       导出的模型
        x (Tensor):              输入的参考值
        onnx_path (str):         保存onnx路径
        simple (bool, optional): 是否使用onnxsim简化onnx模型. Defaults to True.
    """
    model.cpu()
    x = x.cpu()

    torch.onnx.export(
        model,
        x,
        onnx_path,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
    )

    # onnxsim简化模型
    if simple:
        from onnxsim import simplify
        model_ = onnx.load(onnx_path)
        model_simp, check = simplify(model_)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, onnx_path)
        print("simplify onnx success!")
    print("export onnx success!")


def export_openvino(onnx_path: str, openvino_path: str):
    """export onnx to openvino

    Args:
        onnx_path (str):     onnx路径
        openvino_path (str): openvino路径
    """
    import subprocess
    optimize_command = ["mo", "--input_model", onnx_path, "--output_dir", openvino_path]
    subprocess.run(optimize_command, check=True)  # nosec
    print("export openvino success!")


def export(savedir: str, onnx: bool = True, openvino: bool = False):
    """export to torchscirpt, onnx, openvino...

    Args:
        savedir (str): 模型文件夹路径
        onnx (bool, optional):     导出onnx. Defaults to True.
        openvino (bool, optional): 导出export_openvino. Defaults to False.
    """
    # cfg
    with open(os.path.join(savedir, "config.yaml"), 'r', encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # load model
    model = load_model(cfg, savedir)

    # input
    x = torch.ones(1, 3, *cfg["DATASET"]["resize"])

    # onnx
    if onnx:
        onnx_path = os.path.join(savedir, "best_model.onnx")
        export_onnx(model, x, onnx_path)

        # openvino
        if openvino:
            openvino_path = os.path.join(savedir, "openvino")
            export_openvino(onnx_path, openvino_path)


if __name__ == "__main__":
    savedir  = "./saved_model/mvtec/MemSeg-bottle"

    # export onnx
    export(savedir, True, True)
