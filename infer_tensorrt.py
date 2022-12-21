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


# refer https://github.com/NVIDIA/TensorRT/blob/main/samples/python/efficientnet/infer.py
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


    def get_model(self, engine_path: str):
        """获取tensorrt模型

        Args:
            engine_path (str):  模型路径

        """
        # Load TRT engine
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            assert runtime
            self.engine = runtime.deserialize_cuda_engine(f.read())
        assert self.engine
        self.context = self.engine.create_execution_context()
        assert self.context

        # Setup I/O bindings
        self.inputs = []        # inputs
        self.outputs = []       # outputs
        self.allocations = []   # inputs&outputs cuda memorys
        for i in range(self.engine.num_bindings):
            is_input = False
            if trt.__version__ < "8.5":
                if self.engine.binding_is_input(i):
                    is_input = True
                name = self.engine.get_binding_name(i)
                dtype = self.engine.get_binding_dtype(i)
                shape = self.engine.get_binding_shape(i)
            else:
                # 8.5 api
                name = self.engine.get_tensor_name(i)
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    is_input = True
                dtype = self.engine.get_tensor_dtype(name)
                shape = self.engine.get_tensor_shape(name) # both engine and context have this func and the returns are same.

            if is_input:
                self.batch_size = shape[0]
            dtype = np.dtype(trt.nptype(dtype))
            size = dtype.itemsize
            for s in shape:
                size *= s
            allocation = cuda.mem_alloc(size) # allocate cuda memory
            binding = {
                "index": i,
                "name": name,
                "dtype": dtype,
                "shape": list(shape),
                "allocation": allocation,
            }
            self.allocations.append(allocation) # allocate cuda memory
            if is_input:
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

        assert self.batch_size > 0
        assert len(self.inputs) > 0
        assert len(self.outputs) > 0
        assert len(self.allocations) > 0


    def input_spec(self, i:int = 0):
        """
        Get the specs for the input tensor of the network. Useful to prepare memory allocations.
        :params:
            i: the index of input

        :return: Two items, the shape of the input tensor and its (numpy) datatype.
        """
        return self.inputs[i]["shape"], self.inputs[i]["dtype"]


    def output_spec(self, i:int = 0):
        """
        Get the specs for the output tensor of the network. Useful to prepare memory allocations.
        :params:
            i: the index of input

        :return: Two items, the shape of the output tensor and its (numpy) datatype.
        """
        return self.outputs[i]["shape"], self.outputs[i]["dtype"]


    def warm_up(self):
        """预热模型
        """
        # [h w c], 这是opencv读取图片的shape
        x = np.zeros((*self.infer_size, 3), dtype=np.float32)
        self.infer(x)


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
        infer_height, infer_width = self.infer_size # 推理时使用的图片大小
        x = self.transform(image=image)['image'] # [c, h, w]
        x = np.expand_dims(x, axis=0)            # [c, h, w] -> [b, c, h, w]
        # x = np.ones((1, 3, 224, 224))
        x = x.astype(dtype=np.float32)

        # 3.推理
        # 准备存放输出结果
        output = np.zeros(*self.output_spec())

        # Process I/O and execute the network
        cuda.memcpy_htod(self.inputs[0]["allocation"], np.ascontiguousarray(x)) # cpu memory to gpu memory
        self.context.execute_v2(self.allocations)
        cuda.memcpy_dtoh(output, self.outputs[0]["allocation"])                 # gpu memory to cpu memory

        predictions = np.reshape(output, (1, 2, infer_height, infer_width))

        # 4.预测后处理
        y = softmax(predictions, axis=1)    # [1, 2, H, W]
        y = y[0][1]                         # [H, W] 取出1,代表有问题的层
        y = y.squeeze()                     # [H, W]

        # 5.还原到原图尺寸
        y = cv2.resize(y, (image_width, image_height))

        return y


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
    anomaly_map = inference.infer(image)    # [900, 900]

    # 4.生成mask,mask边缘,热力图叠加原图
    mask, mask_outline, superimposed_map = gen_images(image, anomaly_map)
    end = time.time()

    print("infer time:", end - start)

    # 5.保存图片
    save_image(save_path, image, mask, mask_outline, superimposed_map)


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
    trt_path   = "./saved_model/mvtec/MemSeg-bottle/best_model.engine" # trtexec --onnx=best_model.onnx --saveEngine=best_model.engine
    save_path  = "./saved_model/mvtec/MemSeg-bottle/tensorrt_output.jpg"
    save_dir   = "./saved_model/mvtec/MemSeg-bottle/tensorrt_result"
    # single(json_path, trt_path, image_path, save_path)
    # multi(json_path, trt_path, image_dir, save_dir)

    image_path = "./datasets/custom/test/bad/001.jpg"
    image_dir  = "./datasets/custom/test/bad"
    json_path  = "./saved_model/MemSeg-custom/config.json"
    trt_path   = "./saved_model/MemSeg-custom/best_model.engine"
    save_path  = "./saved_model/MemSeg-custom/tensorrt_output.jpg"
    save_dir   = "./saved_model/MemSeg-custom/tensorrt_result"
    single(json_path, trt_path, image_path, save_path)
    # multi(json_path, trt_path, image_dir, save_dir)
