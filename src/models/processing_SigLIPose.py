from typing import List, Optional, Tuple, Union

import numpy as np

import transformers
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.models.vitpose.image_processing_vitpose import (
    get_keypoint_predictions,
    post_dark_unbiased_data_processing,
)
from transformers.processing_utils import ProcessorMixin


logger = transformers.utils.logging.get_logger("transformers")


class SigLIPoseProcessor(ProcessorMixin):
    r"""
    Constructs a SigLIp2 processor which wraps a SigLIp2 image processor and a Gemma tokenizer into a single processor.

    [`SigLIp2Processor`] offers all the functionalities of [`SigLIp2ImageProcessor`] and [`GemmaTokenizerFast`]. See the
    [`~SigLIp2Processor.__call__`] and [`~SigLIp2Processor.decode`] for more information.

    Args:
        image_processor ([`SigLIp2ImageProcessor`]):
            The image processor is a required input.

    """

    attributes = ["image_processor"]

    image_processor_class = "AutoImageProcessor"

    def __init__(
        self,
        image_processor,
        patch_size: int = 16,
        output_size: Tuple[int, int] = (16 * 4, 16 * 4),
        num_joints: int = 24,
        sigma: int = 2.0,
    ):
        self.patch_size = patch_size
        self.output_size = output_size
        self.num_joints = num_joints
        self.sigma = sigma

        super().__init__(image_processor)

    def keypoints_from_heatmaps_bottomup(
        self, heatmaps: np.ndarray, original_size: Tuple[int, int], kernel: int = 11, threshold: float = 0.1
    ):
        """
        Bottom-up 방식으로 히트맵에서 키포인트 추출

        Args:
            heatmaps: [batch, num_joints, height, width] - 모델 출력 히트맵
            original_size: (width, height) - 원본 이미지 크기
            kernel: DARK 후처리용 커널 크기
            threshold: 키포인트 검출 임계값

        Returns:
            List[Dict]: 각 이미지별 키포인트 정보
        """
        # 입력 검증
        if not isinstance(heatmaps, np.ndarray):
            raise TypeError("heatmaps should be np.ndarray")
        if heatmaps.ndim != 4:
            raise ValueError("heatmaps should be 4-dimensional")

        batch_size, num_joints, heatmap_height, heatmap_width = heatmaps.shape
        results = []

        for batch_idx in range(batch_size):
            batch_heatmap = heatmaps[batch_idx : batch_idx + 1]  # [1, num_joints, H, W]

            # 1. 히트맵에서 초기 좌표 추출
            coords, scores = get_keypoint_predictions(batch_heatmap)
            coords = coords[0]  # [num_joints, 2]
            scores = scores[0]  # [num_joints, 1]

            # 2. DARK 후처리를 위한 올바른 차원 준비
            # coords를 (1, num_joints, 2) 형태로 reshape
            coords_for_dark = coords.reshape(1, num_joints, 2)

            # DARK 후처리로 정밀도 향상
            refined_coords = post_dark_unbiased_data_processing(coords_for_dark, batch_heatmap, kernel=kernel)
            refined_coords = refined_coords[0]  # [num_joints, 2]

            # 3. 히트맵 좌표를 원본 이미지 좌표로 변환
            final_coords = self.transform_coords_to_original(
                refined_coords, heatmap_size=(heatmap_height, heatmap_width), original_size=original_size
            )

            # 4. 결과 구성 (임계값 필터링은 나중에 적용)
            keypoints = []
            for joint_idx in range(num_joints):
                confidence = float(scores[joint_idx, 0])
                keypoints.append(
                    {
                        "joint_id": joint_idx,
                        "x": float(final_coords[joint_idx, 0]),
                        "y": float(final_coords[joint_idx, 1]),
                        "confidence": confidence,
                        "visible": confidence > threshold,  # 임계값 정보 보존
                    }
                )

            results.append({"keypoints": keypoints, "num_keypoints": len(keypoints)})

        return results

    def transform_coords_to_original(
        self, coords: np.ndarray, heatmap_size: Tuple[int, int], original_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        히트맵 좌표를 원본 이미지 좌표로 변환 (VitPose의 unbiased 방식 적용)

        Args:
            coords: [num_joints, 2] - 히트맵 내 좌표
            heatmap_size: (height, width) - 히트맵 크기
            original_size: (width, height) - 원본 이미지 크기

        Returns:
            np.ndarray: [num_joints, 2] - 원본 이미지 좌표
        """
        if coords.shape[1] != 2:
            raise ValueError("coords should have shape [num_joints, 2]")

        heatmap_h, heatmap_w = heatmap_size
        orig_w, orig_h = original_size

        # VitPose의 unbiased 방식을 따라 스케일 계산
        # 정규화 팩터 200을 사용하여 scale 계산
        normalize_factor = 200.0

        # 이미지의 aspect ratio 고려
        aspect_ratio = orig_w / orig_h

        # 중심점과 스케일 계산 (전체 이미지를 대상으로)
        center = np.array([orig_w * 0.5, orig_h * 0.5], dtype=np.float32)

        # 이미지 크기에 맞춰 스케일 조정
        if orig_w > aspect_ratio * orig_h:
            h_for_scale = orig_w * 1.0 / aspect_ratio
            scale = np.array([orig_w / normalize_factor, h_for_scale / normalize_factor], dtype=np.float32)
        else:
            w_for_scale = orig_h * aspect_ratio
            scale = np.array([w_for_scale / normalize_factor, orig_h / normalize_factor], dtype=np.float32)

        # padding factor 적용
        scale = scale * 1.25

        # VitPose의 transform_preds 로직 적용
        scale = scale * 200.0  # 정규화된 스케일을 다시 복원

        # unbiased 데이터 처리 방식
        scale_y = scale[1] / (heatmap_h - 1.0)
        scale_x = scale[0] / (heatmap_w - 1.0)

        # 좌표 변환
        transformed_coords = np.ones_like(coords)
        transformed_coords[:, 0] = coords[:, 0] * scale_x + center[0] - scale[0] * 0.5
        transformed_coords[:, 1] = coords[:, 1] * scale_y + center[1] - scale[1] * 0.5

        return transformed_coords

    def post_process_pose_estimation_bottomup(
        self, outputs, original_sizes: List[Tuple[int, int]], threshold: float = 0.1, kernel_size: int = 11
    ):
        """
        Bottom-up 포즈 추정 후처리

        Args:
            outputs: 모델 출력 (heatmaps 포함)
            original_sizes: 각 이미지의 원본 크기 [(width, height), ...]
            threshold: 키포인트 검출 임계값
            kernel_size: DARK 후처리 커널 크기

        Returns:
            List[Dict]: 각 이미지별 포즈 정보
        """
        # 입력 검증
        if not hasattr(outputs, "heatmaps"):
            raise ValueError("outputs must have 'heatmaps' attribute")

        heatmaps = outputs.heatmaps

        # PyTorch tensor인 경우 numpy로 변환
        if hasattr(heatmaps, "cpu"):
            heatmaps = heatmaps.cpu().numpy()
        elif hasattr(heatmaps, "numpy"):
            heatmaps = heatmaps.numpy()

        # numpy array가 아닌 경우 변환 시도
        if not isinstance(heatmaps, np.ndarray):
            try:
                heatmaps = np.array(heatmaps)
            except Exception as e:
                raise ValueError(f"Cannot convert heatmaps to numpy array: {e}")

        batch_size = heatmaps.shape[0]

        if len(original_sizes) != batch_size:
            raise ValueError(f"original_sizes length {len(original_sizes)} != batch_size {batch_size}")

        # 배치 전체를 한번에 처리하여 효율성 향상
        all_results = self.keypoints_from_heatmaps_bottomup(
            heatmaps,
            original_size=original_sizes[0],  # 첫 번째 이미지 크기를 기준으로 처리
            kernel=kernel_size,
            threshold=threshold,
        )

        # 각 이미지별로 다른 original_size가 있는 경우 개별 처리
        if len(set(original_sizes)) > 1:  # 서로 다른 크기가 있는 경우
            results = []
            for i in range(batch_size):
                batch_heatmap = heatmaps[i : i + 1]  # [1, num_joints, H, W]
                original_size = original_sizes[i]

                pose_result = self.keypoints_from_heatmaps_bottomup(
                    batch_heatmap, original_size=original_size, kernel=kernel_size, threshold=threshold
                )[0]

                # threshold 적용하여 최종 결과 필터링
                filtered_keypoints = [kp for kp in pose_result["keypoints"] if kp["confidence"] > threshold]

                pose_result["keypoints"] = filtered_keypoints
                pose_result["num_keypoints"] = len(filtered_keypoints)

                results.append(pose_result)
        else:
            # 모든 이미지가 같은 크기인 경우 배치 처리 결과 사용
            results = []
            for result in all_results:
                # threshold 적용하여 최종 결과 필터링
                filtered_keypoints = [kp for kp in result["keypoints"] if kp["confidence"] > threshold]

                result["keypoints"] = filtered_keypoints
                result["num_keypoints"] = len(filtered_keypoints)
                results.append(result)

        return results

    def __call__(
        self,
        images: Optional[Union[ImageInput, list[ImageInput], list[list[ImageInput]]]] = None,
        return_tensors: Optional[Union[str, "TensorType"]] = "pt",  # noqa: F821
        **kwargs,
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to GemmaTokenizerFast's [`~GemmaTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` argument to
        SigLIp2ImageProcessor's [`~SigLIp2ImageProcessor.__call__`] if `images` is not `None`. Please refer to the docstring
        of the above two methods for more information.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `list[PIL.Image.Image]`, `list[np.ndarray]`, `list[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`str`, `list[str]`, `list[list[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `max_length`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            max_length (`int`, *optional*, defaults to 64):
                Maximum length of the returned list and optionally padding length (see above).
            truncation (`bool`, *optional*, defaults to `True`):
                Activates truncation to cut input sequences longer than `max_length` to `max_length`.
            return_tensors (`str` or [`~utils.TensorType`], *optional*, defaults to `'pt'`):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
            - **pixel_attention_mask** -- Attention mask for the pixel values. Returned when `images` is not `None`.
            - **spatial_shapes** -- The number of horizontal and vertical patches per image.
              Returned when `images` is not `None`.
        """

        image_features = self.image_processor(images, **kwargs)

        return BatchFeature(data=dict(**image_features), tensor_type=return_tensors)

    # copied from ViTPose.mmpose.datasets.pipelines.bottom_up_transform.py:HeatmapGenerator
    def generate_heatmaps(self, keypoints):
        def put_gaussian_on_heatmap(heatmap, center, gaussian):
            """
            heatmap: (64, 64) ndarray
            center: (x, y) 좌표 (float or int, 64x64 기준)
            gaussian: (size, size) ndarray
            """
            size = gaussian.shape[0]
            x0, y0 = int(center[0]), int(center[1])
            radius = size // 2

            # heatmap 범위 내에서만 가우시안 분포를 더함
            left = max(0, x0 - radius)
            right = min(heatmap.shape[1], x0 + radius + 1)
            top = max(0, y0 - radius)
            bottom = min(heatmap.shape[0], y0 + radius + 1)

            g_left = max(0, radius - x0)
            g_right = g_left + (right - left)
            g_top = max(0, radius - y0)
            g_bottom = g_top + (bottom - top)

            heatmap[top:bottom, left:right] = np.maximum(
                heatmap[top:bottom, left:right], gaussian[g_top:g_bottom, g_left:g_right]
            )

        """
        keypoints: (joint_num, 2) ndarray, 64x64 기준 좌표
        """
        heatmaps = np.zeros([self.num_joints] + self.output_size, dtype=np.float32)
        size = int(6 * self.sigma + 3)
        x = np.arange(0, size, 1, np.float32)
        y = x[:, None]
        x0, y0 = 3 * self.sigma + 1, 3 * self.sigma + 1
        gaussian = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma**2))

        for i, (x, y) in enumerate(keypoints):
            put_gaussian_on_heatmap(heatmaps[i], (x, y), gaussian)

        return heatmaps

    @property
    def model_input_names(self):
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(image_processor_input_names))


__all__ = ["SigLIPoseProcessor"]
