# app/model_adapter.py
"""
Contains adapters to provide a unified interface for different model architectures
(CLIP, SigLIP, DINOv2) specifically for the purpose of ONNX export.
"""

from abc import ABC, abstractmethod
from typing import Any


class ModelAdapter(ABC):
    """Abstract Base Class for a model adapter."""

    @abstractmethod
    def get_processor_class(self) -> Any:
        """Returns the correct processor class from the transformers library."""
        pass

    @abstractmethod
    def get_model_class(self) -> Any:
        """Returns the correct model class from the transformers library."""
        pass

    def has_text_model(self) -> bool:
        """Returns True if the model includes a text-processing component."""
        return True

    def get_input_size(self, processor: Any) -> tuple[int, int]:
        """Determines the correct input image size from the processor's configuration."""
        # This logic works for generic AutoProcessors
        image_proc = getattr(processor, "image_processor", processor)
        size_config = getattr(image_proc, "size", {}) or getattr(image_proc, "crop_size", {})

        # Handle different config structures (e.g. {"height": 224} vs {"shortest_edge": 224})
        if isinstance(size_config, dict):
            if "height" in size_config:
                return (size_config["height"], size_config["width"])
            if "shortest_edge" in size_config:
                s = size_config["shortest_edge"]
                return (s, s)

        return (224, 224)  # A safe default

    @abstractmethod
    def get_vision_wrapper(self, model: Any, torch_module: Any) -> Any:
        """
        Returns a torch.nn.Module wrapper for ONNX export of the vision model.

        Args:
            model: The loaded HuggingFace model.
            torch_module: The 'torch' module passed explicitly to avoid top-level import.
        """
        pass

    @abstractmethod
    def get_text_wrapper(self, model: Any, torch_module: Any) -> Any:
        """
        Returns a torch.nn.Module wrapper for ONNX export of the text model.
        """
        pass


class ClipAdapter(ModelAdapter):
    """Adapter for standard CLIP models (e.g., from OpenAI, LAION)."""

    def get_processor_class(self):
        from transformers import AutoProcessor

        return AutoProcessor

    def get_model_class(self):
        from transformers import CLIPModel

        return CLIPModel

    def get_vision_wrapper(self, model, torch_module):
        class VisionModelWrapper(torch_module.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, pixel_values):
                return self.model.get_image_features(pixel_values=pixel_values)

        return VisionModelWrapper(model)

    def get_text_wrapper(self, model, torch_module):
        class TextModelWrapper(torch_module.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, input_ids, attention_mask):
                return self.model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)

        return TextModelWrapper(model)


class SiglipAdapter(ModelAdapter):
    """Adapter for SigLIP models (e.g., from Google)."""

    def get_processor_class(self):
        from transformers import AutoProcessor

        return AutoProcessor

    def get_model_class(self):
        from transformers import SiglipModel

        return SiglipModel

    def get_vision_wrapper(self, model, torch_module):
        class VisionModelWrapper(torch_module.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, pixel_values):
                return self.model.get_image_features(pixel_values=pixel_values)

        return VisionModelWrapper(model)

    def get_text_wrapper(self, model, torch_module):
        class TextModelWrapper(torch_module.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, input_ids, attention_mask):
                return self.model.get_text_features(input_ids=input_ids, attention_mask=attention_mask)

        return TextModelWrapper(model)


class DinoV2Adapter(ModelAdapter):
    """
    Adapter for DINOv2 models (e.g., from Facebook/Meta).
    These are vision-only models (no text search support).
    """

    def get_processor_class(self):
        from transformers import AutoImageProcessor

        return AutoImageProcessor

    def get_model_class(self):
        from transformers import AutoModel

        return AutoModel

    def has_text_model(self) -> bool:
        return False

    def get_vision_wrapper(self, model, torch_module):
        class DinoVisionModelWrapper(torch_module.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, pixel_values):
                # DINOv2 returns a sequence of patch embeddings (batch, num_patches, dim)
                # We take the first one ([CLS] token equivalent) as the global image representation.
                outputs = self.model(pixel_values)
                return outputs.last_hidden_state[:, 0]

        return DinoVisionModelWrapper(model)

    def get_text_wrapper(self, model, torch_module):
        return None  # DINOv2 has no text model


def get_model_adapter(model_hf_name: str) -> ModelAdapter:
    """Factory function to get the correct adapter based on the model name."""
    name_lower = model_hf_name.lower()

    if "siglip" in name_lower:
        return SiglipAdapter()

    if "dinov2" in name_lower or "ijepa" in name_lower:
        return DinoV2Adapter()

    # Default to CLIP
    return ClipAdapter()
