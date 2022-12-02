import onnxruntime as ort
import time
import os
from statistics import mean
import numpy as np
import cv2
from infer.infer import Inference
from infer.read_utils import *


class OrtInference(Inference):
    def __init__(self, json_path: str, onnx_path: str, mode: str="cpu") -> None:
        """
        Args:
            json_path (str):  配置文件路径
            onnx_path (str):  onnx_path
            mode (str, optional): cpu cuda tensorrt. Defaults to cpu.
        """
        super().__init__()
        # 超参数
        self.config = get_json(json_path)
        self.infer_size = self.config['infer_size']
        # 载入模型
        self.model = self.get_model(onnx_path, mode)
        # 预热模型
        self.warm_up()


    def get_model(self, onnx_path: str, mode: str="cpu") -> ort.InferenceSession:
        """获取onnxruntime模型

        Args:
            onnx_path (str):    模型路径
            mode (str, optional): cpu cuda tensorrt. Defaults to cpu.

        Returns:
            ort.InferenceSession: 模型session
        """
        mode = mode.lower()
        assert mode in ["cpu", "cuda", "tensorrt"], "onnxruntime only support cpu, cuda and tensorrt inference."
        print(f"inference with {mode} !")

        so = ort.SessionOptions()
        so.log_severity_level = 3
        providers = {
            "cpu":  ['CPUExecutionProvider'],
            # https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html
            "cuda": [
                    ('CUDAExecutionProvider', {
                        'device_id': 0,
                        'arena_extend_strategy': 'kNextPowerOfTwo',
                        'gpu_mem_limit': 2 * 1024 * 1024 * 1024, # 2GB
                        'cudnn_conv_algo_search': 'EXHAUSTIVE',
                        'do_copy_in_default_stream': True,
                    }),
                    'CPUExecutionProvider',
                ],
            # tensorrt
            # https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html
            # it is recommended you also register CUDAExecutionProvider to allow Onnx Runtime to assign nodes to CUDA execution provider that TensorRT does not support.
            # set providers to ['TensorrtExecutionProvider', 'CUDAExecutionProvider'] with TensorrtExecutionProvider having the higher priority.
            "tensorrt": [
                    ('TensorrtExecutionProvider', {
                        'device_id': 0,
                        'trt_max_workspace_size': 2 * 1024 * 1024 * 1024, # 2GB
                        'trt_fp16_enable': False,
                    }),
                    ('CUDAExecutionProvider', {
                        'device_id': 0,
                        'arena_extend_strategy': 'kNextPowerOfTwo',
                        'gpu_mem_limit': 2 * 1024 * 1024 * 1024, # 2GB
                        'cudnn_conv_algo_search': 'EXHAUSTIVE',
                        'do_copy_in_default_stream': True,
                    })
                ]
        }[mode]
        return ort.InferenceSession(onnx_path, sess_options=so, providers=providers)


    def warm_up(self):
        """预热模型
        """
        x = np.zeros((1, 3, *self.infer_size), dtype=np.float32)
        self.model.run(None, {self.model.get_inputs()[0].name: x})


    def infer(self, image: np.ndarray) -> np.ndarray:
        """推理单张图片

        Args:
            image (np.ndarray): 图片

        Returns:
            np.ndarray: hotmap
        """
        # 1.保存图片高宽
        image_height, image_width = image.shape[0], image.shape[1]

        # 2.图片预处理
        # 推理时使用的图片大小
        infer_height, infer_width = self.infer_size
        transform = get_transform(infer_height, infer_width, tensor=False)
        x = transform(image=image)
        x = np.expand_dims(x['image'], axis=0)
        # x = np.ones((1, 3, 224, 224))
        x = x.astype(dtype=np.float32)

        inputs = self.model.get_inputs()
        input_name1 = inputs[0].name
        predictions = self.model.run(None, {input_name1: x})    # 返回值为list

        # 4.预测后处理
        y = softmax(predictions[0], axis=1) # [1, 2, H, W]
        y = y[0][1]                         # [H, W] 取出1,代表有问题的层
        y = y.squeeze()                     # [H, W]

        # 5.还原到原图尺寸
        y = cv2.resize(y, (image_width, image_height))

        return y


def single(json_path: str, onnx_path: str, image_path: str, save_path: str, mode: str="cpu") -> None:
    """预测单张图片

    Args:
        json_path (str):      配置文件路径
        onnx_path (str):      onnx_path
        image_path (str):     图片路径
        save_path (str):      保存图片路径
        mode (str, optional): cpu cuda tensorrt. Defaults to cpu.
    """
    # 1.获取推理器
    inference = OrtInference(json_path, onnx_path, mode)

    # 2.打开图片
    image = load_image(image_path)

    # 3.推理
    start = time.time()
    anomaly_map = inference.infer(image)    # [900, 900]

    # 4.生成mask,mask边缘,热力图叠加原图
    mask, mask_outline, superimposed_map = gen_images(image, anomaly_map)
    end = time.time()

    print("infer time:", end - start)

    # 5.保存图片
    save_image(save_path, image, mask, mask_outline, superimposed_map)


def multi(json_path: str, onnx_path: str, image_dir: str, save_dir: str, mode: str="cpu") -> None:
    """预测多张图片

    Args:
        json_path (str):          配置文件路径
        onnx_path (str):          onnx_path
        image_dir (str):          图片文件夹
        save_dir (str, optional): 保存图片路径,没有就不保存. Defaults to None.
        mode (str, optional):     cpu cuda tensorrt. Defaults to cpu.
    """
    # 0.检查保存路径
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"mkdir {save_dir}")
    else:
        print("保存路径为None,不会保存图片")

    # 1.获取推理器
    inference = OrtInference(json_path, onnx_path, mode)

    # 2.获取文件夹中图片
    imgs = os.listdir(image_dir)
    imgs = [img for img in imgs if img.endswith(("jpg", "jpeg", "png", "bmp"))]

    infer_times: list[float] = []
    # 批量推理
    for img in imgs:
        # 3.拼接图片路径
        image_path = os.path.join(image_dir, img);

        # 4.打开图片
        image = load_image(image_path)

        # 5.推理
        start = time.time()
        anomaly_map = inference.infer(image)    # [900, 900]

        # 6.生成mask,mask边缘,热力图叠加原图
        mask, mask_outline, superimposed_map = gen_images(image, anomaly_map)
        end = time.time()

        infer_times.append(end - start)
        print("infer time:", end - start)

        if save_dir is not None:
            # 7.保存图片
            save_path = os.path.join(save_dir, img)
            save_image(save_path, image, mask, mask_outline, superimposed_map)

    print("avg infer time: ", mean(infer_times))


if __name__ == "__main__":
    image_path = "./datasets/MVTec/bottle/test/broken_large/000.png"
    image_dir  = "./datasets/MVTec/bottle/test/broken_large"
    json_path  = "./saved_model/mvtec/MemSeg-bottle/config.json"
    onnx_path  = "./saved_model/mvtec/MemSeg-bottle/best_model.onnx"
    save_path  = "./saved_model/mvtec/MemSeg-bottle/onnxruntime_output.jpg"
    save_dir   = "./saved_model/mvtec/MemSeg-bottle/onnx_result"
    single(json_path, onnx_path, image_path, save_path, mode="cuda")
    # multi(json_path, onnx_path, image_dir, save_dir, mode="cuda")

    image_path = "./datasets/custom/test/bad/001.jpg"
    image_dir  = "./datasets/custom/test/bad"
    yaml_path  = "./saved_model/1/MemSeg-custom-256/config.json"
    json_path  = "./saved_model/1/MemSeg-custom-256/best_model.onnx"
    save_path  = "./saved_model/1/MemSeg-custom-256/onnxruntime_output.jpg"
    save_dir   = "./saved_model/1/MemSeg-custom-256/onnx_result"
    # single(json_path, onnx_path, image_path, save_path, mode="cuda")
    # multi(json_path, onnx_path, image_dir, save_dir, mode="cuda")
