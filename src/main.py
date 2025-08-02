import logging
import os
import sys

import datasets
import numpy as np
import torch
from datasets import load_dataset
from models import SiglipForPoseEstimation, SigLIPoseProcessor
from setproctitle import setproctitle
from trl import TrlParser

import transformers
from transformers import Trainer, TrainingArguments, set_seed
from transformers.data.data_collator import DataCollatorMixin
from transformers.image_processing_utils import ImageProcessingMixin


logger = transformers.utils.logging.get_logger("transformers")

classification_map = {
    "스탠딩 사이드 크런치": 0,
    "스탠딩 니업": 1,
    "버피 테스트": 2,
    "스텝 포워드 다이나믹 런지": 3,
    "스텝 백워드 다이나믹 런지": 4,
    "사이드 런지": 5,
    "크로스 런지": 6,
    "굿모닝": 7,
    "프런트 레이즈": 8,
    "업라이트로우": 9,
    "바벨 스티프 데드리프트": 10,
    "바벨 로우": 11,
    "라잉 레그 레이즈": 12,
    "크런치": 13,
    "바이시클 크런치": 14,
    "시저크로스": 15,
    "힙쓰러스트": 16,
    "플랭크": 17,
    "푸시업": 18,
    "니푸쉬업": 19,
}


class SigLIPoseDataCollator(DataCollatorMixin):
    """
    Data collator for SigLIPose.
    This collator stacks the pixel values and labels into numpy arrays.
    """

    return_tensors: str = "pt"

    def torch_call(self, feature_ls):
        pixel_values, split_index = list(), list()
        labels, exercise = list(), list()
        for feature in feature_ls:
            pixel_values.append(feature["pixel_values"])
            split_index.append(len(feature["pixel_values"]))
            labels.append(feature["labels"])
            exercise.append(feature["exercise"].item())

        return {
            "pixel_values": torch.concat(pixel_values),
            "labels": torch.concat(labels),
            "exercise": torch.tensor(exercise),
            "split_index": split_index,
        }


def main(train_args: TrainingArguments) -> None:
    def preprocessor(example):
        process_finish_ls = list()
        for row_dataset in list(zip(*[example[key] for key in example])):
            row_dataset = {key: value for key, value in zip(example.keys(), row_dataset)}  # noqa: C416

            if row_dataset["metadata"]["exercise"] not in classification_map:
                logger.warning(f"Skipping dataset with exercise: {row_dataset['metadata']['exercise']}")
                continue

            sub_process_ls = list()
            for point in row_dataset["point_ls"]:
                image = row_dataset["image_ls"][point["frame"] - 1]
                label2id = sorted(config.label2id.items(), key=lambda x: x[1])

                # 키포인트를 numpy 배열로 변환 (x, y, visibility=1)
                keypoint_ls = np.array([[point[key[0]]["x"], point[key[0]]["y"]] for key in label2id])

                org_w, org_h = row_dataset["metadata"]["width"], row_dataset["metadata"]["height"]
                tgt_w, tgt_h = processor.output_size

                keypoints_scaled = np.zeros_like(keypoint_ls, dtype=np.float32)
                keypoints_scaled[:, 0] = keypoint_ls[:, 0] / org_w * tgt_w
                keypoints_scaled[:, 1] = keypoint_ls[:, 1] / org_h * tgt_h
                keypoints_scaled = np.round(keypoints_scaled).astype(int)

                # 이미지 처리
                output = processor(image)
                labels = processor.generate_heatmaps(keypoints_scaled)
                output["labels"] = labels

                sub_process_ls.append(output)
            sub_return_dict = dict()
            for res in sub_process_ls:
                for key, value in res.items():
                    sub_return_dict.setdefault(key, []).append(value)

            # [1, 3, 256, 256] -> [frame-num, 3, 256, 256]
            sub_return_dict["pixel_values"] = np.concatenate(sub_return_dict["pixel_values"])

            # [24, 16, 16] -> [frame-num, 24, 16, 16]
            sub_return_dict["labels"] = np.stack(sub_return_dict["labels"])
            sub_return_dict["exercise"] = classification_map[row_dataset["metadata"]["exercise"]]

            process_finish_ls.append(sub_return_dict)

        return_dict = dict()
        for res in process_finish_ls:
            for key, value in res.items():
                return_dict.setdefault(key, []).append(value)

        return return_dict

    model_name = "/home/jp/output_dir/SigLIPose2"
    model = SiglipForPoseEstimation.from_pretrained(model_name)
    # model = torch.compile(model.train())
    config = model.config
    processor = SigLIPoseProcessor.from_pretrained(model_name)

    fix_split_moduls = set(model._no_split_modules).intersection(
        {module.__class__.__name__ for module in model.modules()}
    )
    model._no_split_modules = list(fix_split_moduls)

    dataset = load_dataset("jp1924/FitnessPoseImageDataset")

    # 캐시 디렉토리 설정
    cache_dir = "/home/jp/.cache/[SigLIPose]preprocess/FitnessPoseImageDataset"
    os.makedirs(cache_dir, exist_ok=True)

    # 각 split별 캐시 파일 경로 설정
    cache_file_names = {split: os.path.join(cache_dir, f"{split}_processed.arrow") for split in dataset.keys()}

    dataset = dataset.map(
        preprocessor,
        batched=True,
        batch_size=100,
        num_proc=4,
        remove_columns=dataset["train"].column_names,
        cache_file_names=cache_file_names,
        desc="Processing dataset",
    )
    dataset.set_format("torch")

    collator = SigLIPoseDataCollator()
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=processor,
        data_collator=collator,
    )
    trainer.train()


def train() -> None:
    pass


if __name__ == "__main__":
    parser = TrlParser([TrainingArguments])
    train_args, remain_args = parser.parse_args_and_config(return_remaining_strings=True)

    if remain_args and train_args.distributed_state.is_local_main_process:
        logger.info(f"remain_args: {remain_args}")

    if train_args.seed is not None:
        set_seed(train_args.seed)

    if train_args.run_name is not None:
        setproctitle(train_args.run_name)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = train_args.get_process_log_level()
    logger.setLevel(log_level)

    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    main(train_args)

# from collections import Counter


# Counter([metadata["exercise"] for metadata in dataset["train"]["metadata"]])
{
    "id": "001-1-1-01-Z17_C",
    "image_ls": [],
    "point_ls": [
        {
            "Nose": {"x": 934, "y": 319},
            "Left Eye": {"x": 945, "y": 309},
            "Right Eye": {"x": 922, "y": 309},
            "Left Ear": {"x": 958, "y": 326},
            "Right Ear": {"x": 902, "y": 326},
            "Left Shoulder": {"x": 981, "y": 396},
            "Right Shoulder": {"x": 876, "y": 396},
            "Left Elbow": {"x": 1075, "y": 352},
            "Right Elbow": {"x": 780, "y": 373},
            "Left Wrist": {"x": 1003, "y": 324},
            "Right Wrist": {"x": 850, "y": 336},
            "Left Hip": {"x": 964, "y": 559},
            "Right Hip": {"x": 898, "y": 566},
            "Left Knee": {"x": 984, "y": 717},
            "Right Knee": {"x": 898, "y": 718},
            "Left Ankle": {"x": 997, "y": 844},
            "Right Ankle": {"x": 907, "y": 845},
            "Neck": {"x": 932, "y": 354},
            "Left Palm": {"x": 982, "y": 321},
            "Right Palm": {"x": 871, "y": 331},
            "Back": {"x": 930, "y": 452},
            "Waist": {"x": 930, "y": 507},
            "Left Foot": {"x": 1004, "y": 867},
            "Right Foot": {"x": 904, "y": 868},
            "frame": 1,
        },
    ],
    "metadata": {
        "key": "001",
        "type": "맨몸 운동",
        "pose": "선 자세",
        "exercise": "스탠딩 사이드 크런치",
        "conditions": [
            {"condition": "척추의 중립", "value": True},
            {"condition": "시선 정면 유지", "value": True},
            {"condition": "수축시 무릎과 팔꿈치가 충분히 가까움", "value": True},
            {"condition": "무릎이 몸통 측면에서 올라오는지 여부", "value": True},
            {"condition": "양 손이 머리 뒤에 위치", "value": True},
        ],
        "description": "1 시선 정면, 2 척추 중립, 3 수축시 팔꿈치와 무릎 가깝고, 4 무릎이 몸통 측면에서 올라오며 5 양 손이 머리 뒤에 위치",
        "info": "Day05_200925_F/1/C/001-1-1-01-Z17_C",
        "width": 1920,
        "height": 1080,
    },
}
