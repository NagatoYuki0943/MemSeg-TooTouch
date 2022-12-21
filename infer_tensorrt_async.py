import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit # 初始化cuda
import time
import os
from statistics import mean
import numpy as np
import cv2
from infer.infer import Inference
from infer.read_utils import *


class TrtInference(Inference):
    def __init__(self, json_path: str, trt_path: str) -> None:
        """
        Args:
            json_path (str):  配置文件路径
            trt_path (str):   trt_path
            mode (str, optional): cpu cuda tensorrt. Defaults to cpu.
        """
        super().__init__()
        # 超参数
        self.config = get_json(json_path)
        self.infer_size = self.config['infer_size']
        # 载入模型
        self.get_model(trt_path)
        # transform
        infer_height, infer_width = self.infer_size # 推理时使用的图片大小
        self.transform = get_transform(infer_height, infer_width, "numpy")
        # 预热模型
        self.warm_up()


    def get_model(self, trt_path: str):
        """获取tensorrt模型

        Args:
            trt_path (str):    模型路径

        Returns:

        """
        # Load the network in Inference Engine
        trt_logger = trt.Logger(trt.Logger.WARNING)
        with open(trt_path, "rb") as f, trt.Runtime(trt_logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())


    def warm_up(self):
        """预热模型
        """
        # [h w c], 这是opencv读取图片的shape
        x = np.zeros((*self.infer_size, 3), dtype=np.float32)
        self.infer(x)


    def infer(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        """推理单张图片

        Args:
            image (np.ndarray): 图片

        Returns:
            tuple[np.ndarray, float]: hotmap, score
        """
        # 1.保存图片高宽
        image_height, image_width = image.shape[0], image.shape[1]

        # 2.图片预处理
        # 推理时使用的图片大小
        infer_height, infer_width = self.infer_size
        x = self.transform(image=image)['image'] # [c, h, w]
        x = np.expand_dims(x, axis=0)            # [c, h, w] -> [b, c, h, w]
        # x = np.ones((1, 3, 224, 224))
        x = x.astype(dtype=np.float32)

        # 3.推理
        with self.engine.create_execution_context() as context:
            # Set input shape based on image dimensions for inference
            context.set_binding_shape(self.engine.get_binding_index("input"), (1, 3, infer_height, infer_width))
            # Allocate host and device buffers
            bindings = []
            for binding in self.engine:
                binding_idx = self.engine.get_binding_index(binding)
                size = trt.volume(context.get_binding_shape(binding_idx))
                dtype = trt.nptype(self.engine.get_binding_dtype(binding))
                if self.engine.binding_is_input(binding):
                    input_buffer = np.ascontiguousarray(x)
                    input_memory = cuda.mem_alloc(x.nbytes)
                    bindings.append(int(input_memory))
                else:
                    output_buffer = cuda.pagelocked_empty(size, dtype)
                    output_memory = cuda.mem_alloc(output_buffer.nbytes)
                    bindings.append(int(output_memory))

            stream = cuda.Stream()
            # Transfer input data to the GPU.
            cuda.memcpy_htod_async(input_memory, input_buffer, stream)
            # Run inference
            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            # Transfer prediction output from the GPU.
            cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
            # Synchronize the stream
            stream.synchronize()

            predictions = np.reshape(output_buffer, (1, 2, infer_height, infer_width))

        # 4.预测后处理
        y = softmax(predictions, axis=1)    # [1, 2, H, W]
        y = y[0][1]                         # [H, W] 取出1,代表有问题的层
        y = y.squeeze()                     # [H, W]

        # 5.得到分数
        # Paper 4.2: the mean of the scores of the top 100 most abnormal pixel points in the image is used as the anomaly score at the image-level
        score = np.flip(np.sort(y.reshape(-1), axis=0), axis=0)[:100].mean()

        # 6.还原到原图尺寸
        y = cv2.resize(y, (image_width, image_height))

        return y, score


def single(json_path: str, trt_path: str, image_path: str, save_path: str) -> None:
    """预测单张图片

    Args:
        json_path (str):      配置文件路径
        trt_path (str):       onnx_path
        image_path (str):     图片路径
        save_path (str):      保存图片路径
    """
    # 1.获取推理器
    inference = TrtInference(json_path, trt_path)

    # 2.打开图片
    image = load_image(image_path)

    # 3.推理
    start = time.time()
    anomaly_map, score = inference.infer(image)    # [900, 900] [1]
    print(score)

    # 4.生成mask,mask边缘,热力图叠加原图
    mask, mask_outline, superimposed_map = gen_images(image, anomaly_map)
    end = time.time()
    print("infer time:", end - start)

    # 5.保存图片
    save_image(save_path, image, mask, mask_outline, superimposed_map, score)


def multi(json_path: str, trt_path: str, image_dir: str, save_dir: str) -> None:
    """预测多张图片

    Args:
        json_path (str):          配置文件路径
        trt_path (str):           onnx_path
        image_dir (str):          图片文件夹
        save_dir (str, optional): 保存图片路径,没有就不保存. Defaults to None.
    """
    # 0.检查保存路径
    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            print(f"mkdir {save_dir}")
    else:
        print("保存路径为None,不会保存图片")

    # 1.获取推理器
    inference = TrtInference(json_path, trt_path)

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
        anomaly_map, score = inference.infer(image)    # [900, 900] [1]
        print(score)

        # 6.生成mask,mask边缘,热力图叠加原图
        mask, mask_outline, superimposed_map = gen_images(image, anomaly_map)
        end = time.time()

        infer_times.append(end - start)
        print("infer time:", end - start)

        if save_dir is not None:
            # 7.保存图片
            save_path = os.path.join(save_dir, img)
            save_image(save_path, image, mask, mask_outline, superimposed_map, score)

    print("avg infer time: ", mean(infer_times))


if __name__ == "__main__":
    image_path = "./datasets/MVTec/bottle/test/broken_large/000.png"
    image_dir  = "./datasets/MVTec/bottle/test/broken_large"
    json_path  = "./saved_model/mvtec/MemSeg-bottle/config.json"
    trt_path   = "./saved_model/mvtec/MemSeg-bottle/best_model.engine"
    save_path  = "./saved_model/mvtec/MemSeg-bottle/tensorrt_async_output.jpg"
    save_dir   = "./saved_model/mvtec/MemSeg-bottle/tensorrt_async_result"
    # single(json_path, trt_path, image_path, save_path)
    # multi(json_path, trt_path, image_dir, save_dir)

    image_path = "./datasets/custom/test/bad/001.jpg"
    image_dir  = "./datasets/custom/test/bad"
    json_path  = "./saved_model/MemSeg-custom/config.json"
    trt_path   = "./saved_model/MemSeg-custom/best_model.engine"
    save_path  = "./saved_model/MemSeg-custom/tensorrt_async_output.jpg"
    save_dir   = "./saved_model/MemSeg-custom/tensorrt_async_result"
    single(json_path, trt_path, image_path, save_path)
    # multi(json_path, trt_path, image_dir, save_dir)
