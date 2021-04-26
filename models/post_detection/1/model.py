# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import sys
import json
import cv2
import torch
import torchvision
import time

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args['model_config'])

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT0")

        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(output0_config['data_type'])

        # Get OUTPUT0 configuration
        output1_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT1")

        # Convert Triton types to numpy types
        self.output1_dtype = pb_utils.triton_string_to_numpy(output1_config['data_type'])

        self.confThreshold = 0.4
        self.class_id = 0

    def xywh2xyxy(self, x):
        """

        Args:
            x:

        Returns:

        """
        y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        return y

    def non_max_suppression(self, pred, conf_thres=0.4, iou_thres=0.5, classes=0, agnostic=False):
        """Performs Non-Maximum Suppression (NMS) on inference results

        Returns:
             detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
        """
        prediction = torch.from_numpy(pred.astype(np.float32))
        if prediction.dtype is torch.float16:
            prediction = prediction.float()
        nc = prediction[0].shape[1] - 5
        xc = prediction[..., 4] > conf_thres
        min_wh, max_wh = 2, 4096
        max_det = 100
        time_limit = 10.0
        multi_label = nc > 1
        output = [None] * prediction.shape[0]
        t = time.time()
        for xi, x in enumerate(prediction):
            x = x[xc[xi]]
            if not x.shape[0]:
                continue
            x[:, 5:] *= x[:, 4:5]
            box = self.xywh2xyxy(x[:, :4])
            if multi_label:
                i, j = (x[:, 5:] > conf_thres).nonzero().t()
                x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
            else:
                conf, j = x[:, 5:].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
            if classes:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
            n = x.shape[0]
            if not n:
                continue
            c = x[:, 5:6] * (0 if agnostic else max_wh)
            boxes, scores = x[:, :4] + c, x[:, 4]
            i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
            if i.shape[0] > max_det:
                i = i[:max_det]
            output[xi] = x[i]
            if (time.time() - t) > time_limit:
                break
        return output

    def scale_coords(self, img1_shape, coords, img0_shape, ratio_pad=None):
        """

        Args:
            img1_shape:
            coords:
            img0_shape:
            ratio_pad:

        Returns:

        """
        if ratio_pad is None:
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]
        coords[:, [0, 2]] -= pad[0]
        coords[:, [1, 3]] -= pad[1]
        coords[:, :4] /= gain
        self.clip_coords(coords, img0_shape)
        return coords

    def clip_coords(self, boxes, img_shape):
        """

        Args:
            boxes:
            img_shape:

        Returns:

        """
        boxes[:, 0].clamp_(0, img_shape[1])
        boxes[:, 1].clamp_(0, img_shape[0])
        boxes[:, 2].clamp_(0, img_shape[1])
        boxes[:, 3].clamp_(0, img_shape[0])

    def get_bbox(self, image, detections_bs):
        boxes = self.non_max_suppression(detections_bs)
        image_shape = image.shape
        outputs = [[-1, 0, 0, 0, 0, 0]]
        crops = []
        if len(boxes) > 0:
            for i, det in enumerate(boxes):
                if det is not None and len(det):
                    det[:, :4] = self.scale_coords((640, 640), det[:, :4],
                                                   (image_shape[0], image_shape[1], image_shape[2])).round()

                    for *xyxy, conf, cls in det:
                        x_min = (xyxy[0] / float(image_shape[1]))
                        y_min = (xyxy[1] / float(image_shape[0]))
                        x_max = (xyxy[2] / float(image_shape[1]))
                        y_max = (xyxy[3] / float(image_shape[0]))
                        score = conf
                        class_id = int(cls)

                        if class_id == self.class_id and score > self.confThreshold:
                            outputs.append([class_id, score, x_min, y_min, x_max, y_max])
                            crops.append([image[y_min:y_max, x_min:x_max]])
        return outputs, crops

    def resize_image(self, input_image, target_size=416, mode=None):
        """Resize input to target size.

        Args:
            img: a ndarray, image data.
            target_size: an integer

        Return:
            img: a ndarray, image data.
            scale: a list of two elements, [col_scale, row_scale], indicates the ratio of resized length / original length.
        """
        img = input_image.copy()
        (rows, cols, _) = img.shape
        if mode:
            img = cv2.resize(img, (int(target_size), int(target_size)), mode)
        else:
            img = cv2.resize(img, (int(target_size), int(target_size)))

        scale = [float(target_size) / cols, float(target_size) / rows]

        return img, scale

    def process_classfi_data(self, crops):
        # 分类图像预处理
        new_crops = []
        for image_np in crops:
            image_np = self.resize_image(image_np, 260, 'inter_area')
            image_np = image_np.astype(np.float32)
            image_np /= 255.
            image_np -= 0.5
            image_np *= 2
            new_crops.append(image_np)
        # 可能一个目标都没有，但是流程好像得走完，因此随机生成一张图片防止分类器没有输入而报错
        new_crops.append(np.random.rand(260, 260, 3))
        return np.asarray(new_crops)

    def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        output0_dtype = self.output0_dtype
        output1_dtype = self.output1_dtype
        responses = []

        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT0").as_numpy()
            in_0 = np.transpose(in_0, (1, 2, 0))
            in_1 = pb_utils.get_input_tensor_by_name(request, "INPUT1").as_numpy()

            bboxs, crops = self.get_bbox(in_0, in_1)
            crops = self.process_classfi_data(crops)
            out_tensor_0 = pb_utils.Tensor("OUTPUT0", crops.astype(output0_dtype))
            out_tensor_1 = pb_utils.Tensor("OUTPUT1", np.asarray(bboxs).astype(output1_dtype))
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor_0, out_tensor_1])
            responses.append(inference_response)
        return responses


def finalize(self):
    """`finalize` is called only once when the model is being unloaded.
    Implementing `finalize` function is OPTIONAL. This function allows
    the model to perform any necessary clean ups before exit.
    """
    print('Cleaning up...')
