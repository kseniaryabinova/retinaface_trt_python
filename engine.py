import time

import cv2
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

import ctypes


class Engine:
    trt_logger = trt.Logger(trt.Logger.VERBOSE)

    def __init__(self, path_to_engine, path_to_plugins=None, batch_size=None):

        if path_to_plugins is not None:
            self._load_plugins(path_to_plugins)

        self.engine = self._load_engine(path_to_engine)
        print(self.engine.get_binding_shape(0))

        if batch_size is None or not 0 < batch_size < self.engine.max_batch_size:
            self.batch_size = self.engine.max_batch_size
        else:
            self.batch_size = batch_size

        self.host_inputs = []
        self.cuda_inputs = []
        self.host_outputs = []
        self.cuda_outputs = []
        self.bindings = []

        self.stream = cuda.Stream()
        self.context = self._create_context()
        _, self.num_channels, self.net_h, self.net_w = \
            self.context.get_binding_shape(0)

        print('input tensor = {}'.format(self.context.get_binding_shape(0)))

    def _load_engine(self, path_to_engine):
        with open(path_to_engine, 'rb') as file, \
                trt.Runtime(Engine.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(file.read())

    def _load_plugins(self, path_to_plugins):
        ctypes.CDLL(path_to_plugins)
        trt.init_libnvinfer_plugins(Engine.trt_logger, '')

    def _create_context(self):
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * \
                   self.batch_size

            host_mem = cuda.pagelocked_empty(size, np.float32)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(cuda_mem))
            if self.engine.binding_is_input(binding):
                self.host_inputs.append(host_mem)
                self.cuda_inputs.append(cuda_mem)
            else:
                self.host_outputs.append(host_mem)
                self.cuda_outputs.append(cuda_mem)
        return self.engine.create_execution_context()


class RetinaFace(Engine):
    def __init__(self, path_to_engine, path_to_plugins):
        super().__init__(path_to_engine, path_to_plugins)
        self.mean = (104, 117, 123)

    def preprocess(self, image):
        h, w, _ = image.shape
        new_size = (int(self.net_w), int(self.net_h))

        image = cv2.resize(image, (self.net_w, self.net_h))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose(2, 0, 1).astype(np.float32)

        for c in range(self.num_channels):
            image[c] -= self.mean[c]

        image = np.expand_dims(image, axis=0)

        rescale_factor_w = w / new_size[0]
        rescale_factor_h = h / new_size[1]
        return image, rescale_factor_w, rescale_factor_h

    def detect(self, image, thresh=0.5):
        preprocessed_image, rescale_factor_w, rescale_factor_h = self.preprocess(image)
        np.copyto(self.host_inputs[0], preprocessed_image.ravel())

        cuda.memcpy_htod_async(
            self.cuda_inputs[0], self.host_inputs[0], self.stream)
        start = time.time()
        self.context.execute_async(
            batch_size=self.batch_size,
            bindings=self.bindings,
            stream_handle=self.stream.handle)
        self.stream.synchronize()
        finish = time.time()
        print('[{}] inference time {} sec'.format(i, finish - start))
        cuda.memcpy_dtoh_async(self.host_outputs[0], self.cuda_outputs[0],
                               self.stream)
        cuda.memcpy_dtoh_async(self.host_outputs[1], self.cuda_outputs[1],
                               self.stream)
        cuda.memcpy_dtoh_async(self.host_outputs[2], self.cuda_outputs[2],
                               self.stream)
        self.stream.synchronize()

        boxes, scores, landmarks = self.host_outputs
        boxes = boxes.reshape((scores.shape[0], 4))
        landmarks = landmarks.reshape((scores.shape[0], 10))

        return self.postprocess(boxes, scores, landmarks,
                                rescale_factor_w, rescale_factor_h, thresh)

    def postprocess(self, raw_boxes, raw_scores, raw_landmarks,
                    rescale_factor_w, rescale_factor_h, thresh):
        indices = np.where(raw_scores > thresh)[0]
        scores = raw_scores[indices]
        boxes = raw_boxes[indices]
        landmarks = raw_landmarks[indices]

        boxes[:, 0] = (boxes[:, 0]) * rescale_factor_w
        boxes[:, 1] = (boxes[:, 1]) * rescale_factor_h
        boxes[:, 2] = (boxes[:, 2]) * rescale_factor_w
        boxes[:, 3] = (boxes[:, 3]) * rescale_factor_h

        landmarks[:, 0] = (landmarks[:, 0]) * rescale_factor_w
        landmarks[:, 1] = (landmarks[:, 1]) * rescale_factor_h
        landmarks[:, 2] = (landmarks[:, 2]) * rescale_factor_w
        landmarks[:, 3] = (landmarks[:, 3]) * rescale_factor_h
        landmarks[:, 4] = (landmarks[:, 4]) * rescale_factor_w
        landmarks[:, 5] = (landmarks[:, 5]) * rescale_factor_h
        landmarks[:, 6] = (landmarks[:, 6]) * rescale_factor_w
        landmarks[:, 7] = (landmarks[:, 7]) * rescale_factor_h
        landmarks[:, 8] = (landmarks[:, 8]) * rescale_factor_w
        landmarks[:, 9] = (landmarks[:, 9]) * rescale_factor_h

        return scores, boxes, landmarks


if __name__ == '__main__':
    retinaface = RetinaFace(
        '/tmp/projest_521/cmake-build-debug-neuro3/retinaface_mobile0.25.plan',
        '/tmp/projest_521/cmake-build-debug-neuro3/libretinaface_plugins.so')

    img = cv2.imread('aaaa.jpeg')
    img_to_save = img.copy()

    for i in range(10):
        scores, boxes, landmarks = retinaface.detect(img.copy(), thresh=0.2)

    for i in range(scores.shape[0]):
        x1, y1, x2, y2 = boxes[i]
        p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y, p5x, p5y = landmarks[i]

        cv2.rectangle(img_to_save, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.circle(img_to_save, (p1x, p1y), 3, (0, 0, 255))
        cv2.circle(img_to_save, (p2x, p2y), 3, (0, 255, 255))
        cv2.circle(img_to_save, (p3x, p3y), 3, (255, 0, 255))
        cv2.circle(img_to_save, (p4x, p4y), 3, (0, 255, 0))
        cv2.circle(img_to_save, (p5x, p5y), 3, (255, 0, 0))

    cv2.imwrite('prediction.jpg', img_to_save)
