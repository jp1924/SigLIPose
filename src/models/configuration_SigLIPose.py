from typing import Optional, Tuple

from transformers import AutoConfig, Siglip2VisionConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.utils import logging


logger = logging.get_logger(__name__)


class SigLIPoseConfig(PretrainedConfig):
    model_type = "siglipose"
    sub_configs = {"backbone_config": AutoConfig}

    def __init__(
        self,
        backbone_config: Optional[PretrainedConfig] = None,
        use_pretrained_backbone: bool = False,
        use_timm_backbone: bool = False,
        backbone_kwargs: Optional[dict] = None,
        initializer_range: float = 0.02,
        scale_factor: int = 4,
        use_simple_decoder: bool = True,
        image_size: Tuple[int, int] = (224, 224),
        num_classes: int = 20,
        classifier_hidden_dim: int = 256,
        classifier_encoder_out_channels: int = 64,
        classifier_encoder_pool_size: int = 4,
        classifier_temporal_num_layers: int = 2,
        classifier_temporal_dropout: float = 0.1,
        **kwargs,
    ):
        if use_pretrained_backbone:
            logger.info(
                "`use_pretrained_backbone` is `True`. For the pure inference purpose of VitPose weight do not set this value."
            )
        if use_timm_backbone:
            raise ValueError("use_timm_backbone set `True` is not supported at the moment.")
        if isinstance(backbone_config, dict):
            backbone_config = CONFIG_MAPPING[backbone_config["model_type"]](**backbone_config)

        self.backbone_config = backbone_config
        self.use_pretrained_backbone = use_pretrained_backbone
        self.use_timm_backbone = use_timm_backbone
        self.backbone_kwargs = backbone_kwargs

        self.initializer_range = initializer_range
        self.scale_factor = scale_factor
        self.use_simple_decoder = use_simple_decoder
        self.image_size = image_size

        # Classifier config
        self.num_classes = num_classes
        self.classifier_hidden_dim = classifier_hidden_dim
        self.classifier_encoder_out_channels = classifier_encoder_out_channels
        self.classifier_encoder_pool_size = classifier_encoder_pool_size
        self.classifier_temporal_num_layers = classifier_temporal_num_layers
        self.classifier_temporal_dropout = classifier_temporal_dropout

        super().__init__(**kwargs)


__all__ = ["SigLIPoseConfig"]
