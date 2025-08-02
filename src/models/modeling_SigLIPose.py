from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn

from transformers import AutoModel, logging
from transformers.models.siglip.modeling_siglip import SiglipPreTrainedModel
from transformers.utils import ModelOutput, auto_docstring

from .configuration_SigLIPose import SigLIPoseConfig


# from transformers.models.vitpose.modeling_vitpose import

logger = logging.get_logger(__name__)


@dataclass
@auto_docstring(
    custom_intro="""
    Class for outputs of pose estimation models.
    """
)
class SigLIPoseEstimatorOutput(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Combined loss of heatmap MSE loss and exercise classification loss.
    heatmaps (`torch.FloatTensor` of shape `(batch_size, num_keypoints, height, width)`):
        Heatmaps as predicted by the model.
    exercise_logits (`torch.FloatTensor` of shape `(batch_size, num_classes)`, *optional*):
        Exercise classification logits.
    hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
        Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
        one for the output of each stage) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states
        (also called feature maps) of the model at the output of each stage.
    """

    loss: Optional[torch.FloatTensor] = None
    heatmaps: Optional[torch.FloatTensor] = None
    exercise_logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None


def flip_back(output_flipped, flip_pairs, target_type="gaussian-heatmap"):
    """Flip the flipped heatmaps back to the original form.

    Args:
        output_flipped (`torch.tensor` of shape `(batch_size, num_keypoints, height, width)`):
            The output heatmaps obtained from the flipped images.
        flip_pairs (`torch.Tensor` of shape `(num_keypoints, 2)`):
            Pairs of keypoints which are mirrored (for example, left ear -- right ear).
        target_type (`str`, *optional*, defaults to `"gaussian-heatmap"`):
            Target type to use. Can be gaussian-heatmap or combined-target.
            gaussian-heatmap: Classification target with gaussian distribution.
            combined-target: The combination of classification target (response map) and regression target (offset map).
            Paper ref: Huang et al. The Devil is in the Details: Delving into Unbiased Data Processing for Human Pose Estimation (CVPR 2020).

    Returns:
        torch.Tensor: heatmaps that flipped back to the original image
    """
    if target_type not in ["gaussian-heatmap", "combined-target"]:
        raise ValueError("target_type should be gaussian-heatmap or combined-target")

    if output_flipped.ndim != 4:
        raise ValueError("output_flipped should be [batch_size, num_keypoints, height, width]")
    batch_size, num_keypoints, height, width = output_flipped.shape
    channels = 1
    if target_type == "combined-target":
        channels = 3
        output_flipped[:, 1::3, ...] = -output_flipped[:, 1::3, ...]
    output_flipped = output_flipped.reshape(batch_size, -1, channels, height, width)
    output_flipped_back = output_flipped.clone()

    # Swap left-right parts
    for left, right in flip_pairs.tolist():
        output_flipped_back[:, left, ...] = output_flipped[:, right, ...]
        output_flipped_back[:, right, ...] = output_flipped[:, left, ...]
    output_flipped_back = output_flipped_back.reshape((batch_size, num_keypoints, height, width))
    # Flip horizontally
    output_flipped_back = output_flipped_back.flip(-1)
    return output_flipped_back


class SigLIPoseSimpleDecoder(nn.Module):
    """
    Simple decoding head consisting of a ReLU activation, 4x upsampling and a 3x3 convolution, turning the
    feature maps into heatmaps.
    """

    def __init__(self, config: SigLIPoseConfig) -> None:
        super().__init__()

        self.activation = nn.ReLU()
        self.upsampling = nn.Upsample(scale_factor=config.scale_factor, mode="bilinear", align_corners=False)
        self.conv = nn.Conv2d(
            config.backbone_config.hidden_size, config.num_labels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, hidden_state: torch.Tensor, flip_pairs: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Transform input: ReLU + upsample
        hidden_state = self.activation(hidden_state)
        hidden_state = self.upsampling(hidden_state)
        heatmaps = self.conv(hidden_state)

        if flip_pairs is not None:
            heatmaps = flip_back(heatmaps, flip_pairs)

        return heatmaps


class SigLIPosePretrainedModel(SiglipPreTrainedModel):
    """
    Base class for SigLIPose models.
    This class inherits from `SiglipPreTrainedModel` and is used to initialize the model weights.
    """

    config_class = SigLIPoseConfig
    base_model_prefix = "siglipose"
    main_input_name = "pixel_values"

    def _init_weights(self, module):
        """Initialize the weights following HuggingFace style."""
        if isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param.data)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param.data)
                elif "bias" in name:
                    nn.init.constant_(param.data, 0)
        elif isinstance(module, nn.Module):
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)


class PoseSequenceClassifier(nn.Module):
    """히트맵 시퀀스를 받아서 동작 분류"""

    def __init__(self, config: SigLIPoseConfig):
        """
        PoseSequenceClassifier for action classification from heatmap sequences.

        Args:
            config (SigLIPoseConfig): Model configuration with classifier parameters.
        """
        super().__init__()

        self.num_joints = config.num_labels
        self.num_classes = config.num_classes
        self.hidden_dim = config.classifier_hidden_dim
        self.heatmap_encoder_out_channels = config.classifier_encoder_out_channels
        self.heatmap_encoder_pool_size = config.classifier_encoder_pool_size
        self.temporal_num_layers = config.classifier_temporal_num_layers
        self.temporal_dropout = config.classifier_temporal_dropout

        # 히트맵을 feature로 변환 (nn.Sequential 대신 개별 레이어)
        self.conv = nn.Conv2d(self.num_joints, self.heatmap_encoder_out_channels, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((self.heatmap_encoder_pool_size, self.heatmap_encoder_pool_size))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(
            self.heatmap_encoder_out_channels * self.heatmap_encoder_pool_size * self.heatmap_encoder_pool_size,
            self.hidden_dim,
        )
        self.relu2 = nn.ReLU()

        # 시간축 처리 (LSTM)
        self.temporal_encoder = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.temporal_num_layers,
            batch_first=True,
            dropout=self.temporal_dropout if self.temporal_num_layers > 1 else 0.0,
        )

        # 분류 헤드 (nn.Sequential 대신 개별 레이어)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc3 = nn.Linear(self.hidden_dim // 2, self.num_classes)

    def forward(self, heatmap_sequence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for action classification.

        Args:
            heatmap_sequence (torch.Tensor): [batch, sequence_length, num_joints, height, width]
        Returns:
            torch.Tensor: [batch, num_classes] action logits
        """
        batch_size, seq_len = heatmap_sequence.shape[:2]

        # Encode each frame's heatmap
        frame_features = []
        for i in range(seq_len):
            x = heatmap_sequence[:, i]  # [batch, num_joints, H, W]
            x = self.conv(x)
            x = self.relu1(x)
            x = self.pool(x)
            x = self.flatten(x)
            x = self.fc1(x)
            x = self.relu2(x)
            frame_features.append(x)

        sequence_features = torch.stack(frame_features, dim=1)  # [batch, seq_len, hidden_dim]

        # Temporal encoding
        lstm_out, (hidden, _) = self.temporal_encoder(sequence_features)
        final_features = hidden[-1]  # [batch, hidden_dim]

        # Classification
        x = self.fc2(final_features)
        x = self.relu3(x)
        x = self.dropout(x)
        action_logits = self.fc3(x)

        return action_logits


class SiglipForPoseEstimation(SigLIPosePretrainedModel):
    def __init__(self, config: SigLIPoseConfig):
        super().__init__(config)
        self.config = config

        backbone = AutoModel.from_config(config.backbone_config)
        if hasattr(backbone, "vision_model"):
            backbone = backbone.vision_model
        self.backbone = backbone

        self.head = SigLIPoseSimpleDecoder(config)

        self.layer_norm = nn.LayerNorm(
            self.config.backbone_config.hidden_size,
            eps=self.config.backbone_config.layer_norm_eps,
        )
        self.classifier = PoseSequenceClassifier(config)
        self.init_weights()

    def forward(
        self,
        pixel_values: torch.Tensor,
        split_index: Optional[torch.Tensor] = None,
        flip_pairs: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        exercise: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, SigLIPoseEstimatorOutput]:
        if flip_pairs:
            raise NotImplementedError("flip_pairs is not supported at the moment.")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        outputs = self.backbone(
            pixel_values,
            output_hidden_states=True,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )

        # TODO: 나중에 backbone 모델과 같이 stage 구분을 할 것
        feature_maps = self.layer_norm(outputs.last_hidden_state)  # outputs.feature_maps

        # copied from https://github.com/huggingface/transformers/blob/main/src/transformers/models/vitpose/modeling_vitpose.py#L273-L281
        batch_size = feature_maps.shape[0]
        patch_height = self.config.image_size[0] // self.config.backbone_config.patch_size
        patch_width = self.config.image_size[1] // self.config.backbone_config.patch_size

        sequence_output = feature_maps.permute(0, 2, 1)
        sequence_output = sequence_output.reshape(batch_size, -1, patch_height, patch_width).contiguous()

        heatmaps = self.head(sequence_output, flip_pairs=flip_pairs)

        # Always compute exercise logits for inference

        logits_ls = list()
        for feat in heatmaps.split(split_index):
            logits_ls.append(self.classifier(feat.unsqueeze(0)))

        exercise_logits = torch.concat(logits_ls)

        beta = 0.5
        heatmap_loss = 0.0
        if labels is not None:
            loss_fct = nn.MSELoss()
            heatmap_loss = loss_fct(heatmaps, labels)

        exercise_loss = 0.0
        if exercise is not None:
            loss_fct = nn.CrossEntropyLoss()
            exercise_loss = loss_fct(exercise_logits, exercise)

        loss = (heatmap_loss * beta) + exercise_loss

        if not return_dict:
            if output_hidden_states:
                output = (heatmaps, exercise_logits) + outputs[1:]
            else:
                output = (heatmaps, exercise_logits) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SigLIPoseEstimatorOutput(
            loss=loss,
            heatmaps=heatmaps,
            exercise_logits=exercise_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = ["SiglipForPoseEstimation"]
