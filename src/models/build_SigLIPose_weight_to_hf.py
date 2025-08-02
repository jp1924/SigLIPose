from transformers import AutoConfig, AutoImageProcessor, AutoModel

from .configuration_SigLIPose import SigLIPoseConfig
from .modeling_SigLIPose import SiglipForPoseEstimation
from .processing_SigLIPose import SigLIPoseProcessor


# Siglip2ForPoseEstimation.from_pretrained("/home/jp/output_dir/SigLIPose2")


def main() -> None:
    keypoints = [
        "Nose",
        "Left Eye",
        "Right Eye",
        "Left Ear",
        "Right Ear",
        "Left Shoulder",
        "Right Shoulder",
        "Left Elbow",
        "Right Elbow",
        "Left Wrist",
        "Right Wrist",
        "Left Hip",
        "Right Hip",
        "Left Knee",
        "Right Knee",
        "Left Ankle",
        "Right Ankle",
        "Neck",
        "Left Palm",
        "Right Palm",
        "Back",
        "Waist",
        "Left Foot",
        "Right Foot",
    ]
    label2id = {key: idx for idx, key in enumerate(keypoints)}
    id2label = {idx: key for idx, key in enumerate(keypoints)}

    # NOTE: 코드를 hard-code로 되어 있기 때문에 코드 수정해 가면서 설정값 반영 시키셈

    output_dir = "/home/jp/output_dir/SigLIPose2"
    backbone_name = "google/siglip-base-patch16-256"
    backbone_config = AutoConfig.from_pretrained(backbone_name)
    backbone_model = AutoModel.from_pretrained(backbone_name)
    image_processor = AutoImageProcessor.from_pretrained(backbone_name)

    if hasattr(backbone_config, "vision_config"):
        backbone_config = backbone_config.vision_config
    if hasattr(backbone_model, "vision_model"):
        backbone_model = backbone_model.vision_model

    size = image_processor.size

    config = SigLIPoseConfig(
        backbone_config=backbone_config,
        id2label=id2label,
        label2id=label2id,
        image_size=(size["height"], size["width"]),
    )

    patch_height = config.image_size[0] // config.backbone_config.patch_size
    patch_width = config.image_size[1] // config.backbone_config.patch_size
    processor = SigLIPoseProcessor(
        image_processor=image_processor,
        patch_size=16,
        output_size=(patch_height, patch_width),
        num_joints=len(keypoints),
        sigma=2.0,
    )

    model = SiglipForPoseEstimation(config)
    model.backbone = backbone_model
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    # model.backbone.vision_model.encoder.layers[0].self_attn.k_proj.weight
    # backbone_model.vision_model.encoder.layers[0].self_attn.k_proj.weight


if "__main__" in __name__:
    main()

[
    "Nose",
    "Left Eye",
    "Right Eye",
    "Left Ear",
    "Right Ear",
    "Left Shoulder",
    "Right Shoulder",
    "Left Elbow",
    "Right Elbow",
    "Left Wrist",
    "Right Wrist",
    "Left Hip",
    "Right Hip",
    "Left Knee",
    "Right Knee",
    "Left Ankle",
    "Right Ankle",
    "Neck",
    "Left Palm",
    "Right Palm",
    "Back",
    "Waist",
    "Left Foot",
    "Right Foot",
]

[
    "로잉머신",
    "풀업",
    "바벨 스티프 데드리프트",
    "바벨 데드리프트",
    "스탠딩 사이드 크런치",
    "바벨 컬 ",
    "업라이트로우",
    "사이드 런지",
    "스탠딩 니업",
    "케이블 푸시 다운",
    "스텝 백워드 다이나믹 런지",
    "크로스 런지",
    "굿모닝",
    "사이드 레터럴 레이즈",
    "랫풀 다운",
    "바벨 로우",
    "바벨 런지",
    "덤벨 풀 오버",
    "라잉 트라이셉스 익스텐션",
    "딥스",
    "버피 테스트",
    "케이블 크런치",
    "오버 헤드 프레스",
    "덤벨 벤트오버 로우",
    "행잉 레그 레이즈",
    "스텝 포워드 다이나믹 런지",
    "프런트 레이즈",
    "바벨 스쿼트",
    "덤벨 컬",
    "덤벨 체스트 플라이",
    "페이스 풀",
    "덤벨 인클라인 체스트 플라이",
]
