from typing import Literal

import torch
from torchvision.transforms.functional import to_pil_image as pil_image_from_tensor

from invokeai.app.models.image import ImageCategory, ResourceOrigin
from invokeai.app.invocations.primitives import (
    ImageField,
    ImageOutput,
)

from invokeai.backend.stable_diffusion.diffusers_pipeline import (
    image_resized_to_grid_as_tensor,
)
from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    InputField,
    InvocationContext,
    invocation,
    invocation_output,
    OutputField
)

MASK_TYPES: list = [
    "shadows",
    "highlights",
    "midtones"
]

CIELAB_CHANNELS: list = ["L", "A", "B"]


@invocation_output("shmmask_output")
class ShadowsHighlightsMidtonesMasksOutput(BaseInvocationOutput):
    highlights_mask: ImageField = OutputField(default=None, description="Soft-edged highlights mask")
    midtones_mask: ImageField = OutputField(default=None, description="Soft-edged midtones mask")
    shadows_mask: ImageField = OutputField(default=None, description="Soft-edged shadows mask")
    width: int = OutputField(description="Width of the input/outputs")
    height: int = OutputField(description="Height of the input/outputs")


@invocation(
    "lab_channel",
    title="Extract CIELAB Channel",
    tags=["image", "channel", "mask", "cielab", "lab"],
    category="image"
)
class ExtractCIELABChannelInvocation(BaseInvocation):
    """Get a selected channel from L*a*b* color space"""

    image: ImageField = InputField(description="Image from which to get channel")
    channel: Literal[tuple(CIELAB_CHANNELS)] = InputField(default=CIELAB_CHANNELS[0], description="Channel to extract")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image_in = context.services.images.get_pil_image(self.image.image_name)

        image_out = image_in.convert("LAB")
        image_out = image_out.getchannel(self.channel)
        
        image_dto = context.services.images.create(
            image=image_out,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate
        )
        return ImageOutput(
            image=ImageField(image_name=image_dto.image_name),
            width=image_dto.width,
            height=image_dto.height
        )
        

    
@invocation(
    "shmmask",
    title="Shadows/Highlights/Midtones Mask from Image",
    tags=["mask", "image", "shadows", "highlights", "midtones"],
    category="image"
)
class ShadowsHighlightsMidtonesMaskInvocation(BaseInvocation):
    """Extract a Shadows/Highlights/Midtones mask from an image"""

    image: ImageField = InputField(description="Image from which to extract mask")
    highlight_threshold: float = InputField(default=0.75, description="Threshold beyond which mask values will be at extremum")
    upper_mid_threshold: float = InputField(default=0.7, description="Threshold to which to extend mask border by 0..1 gradient")
    lower_mid_threshold: float = InputField(default=0.3, description="Threshold to which to extend mask border by 0..1 gradient")
    shadow_threshold: float = InputField(default=0.25, description="Threshold beyond which mask values will be at extremum")

    def get_highlights_mask(self, image_tensor):
        img_tensor = image_tensor.clone()
        threshold_h, threshold_s = self.highlight_threshold, self.upper_mid_threshold
        ones_tensor = torch.ones(img_tensor.shape)
        zeros_tensor = torch.zeros(img_tensor.shape)

        zeros_mask = torch.ge(img_tensor, threshold_h)
        ones_mask = torch.lt(img_tensor, threshold_s)
        if not (threshold_h == threshold_s):
            mask_hi = torch.ge(img_tensor, threshold_s)
            mask_lo = torch.lt(img_tensor, threshold_h)
            mask = torch.logical_and(mask_hi, mask_lo)
            masked = img_tensor[mask]
            if 0 < masked.numel():
              vmax, vmin = masked.max(), masked.min()
              if (vmax == vmin):
                  img_tensor[mask] = 0.5 * ones_tensor
              else:
                  img_tensor[mask] = torch.sub(1.0, (img_tensor[mask] - vmin) / (vmax - vmin)) # hi is 0

        img_tensor[ones_mask] = ones_tensor[ones_mask]
        img_tensor[zeros_mask] = zeros_tensor[zeros_mask]

        return img_tensor


    def get_shadows_mask(self, image_tensor):
        img_tensor = image_tensor.clone()
        threshold_h, threshold_s = self.shadow_threshold, self.lower_mid_threshold
        ones_tensor = torch.ones(img_tensor.shape)
        zeros_tensor = torch.zeros(img_tensor.shape)

        zeros_mask = torch.le(img_tensor, threshold_h)
        ones_mask = torch.gt(img_tensor, threshold_s)
        if not (threshold_h == threshold_s):
            mask_hi = torch.le(img_tensor, threshold_s)
            mask_lo = torch.gt(img_tensor, threshold_h)
            mask = torch.logical_and(mask_hi, mask_lo)
            masked = img_tensor[mask]
            if 0 < masked.numel():
                vmax, vmin = masked.max(), masked.min()
                if (vmax == vmin):
                    img_tensor[mask] = 0.5 * ones_tensor
                else:
                    img_tensor[mask] = (img_tensor[mask] - vmin) / (vmax - vmin) # lo is 0

        img_tensor[ones_mask] = ones_tensor[ones_mask]
        img_tensor[zeros_mask] = zeros_tensor[zeros_mask]

        return img_tensor


    def get_midtones_mask(self, image_tensor):
        img_tensor = image_tensor.clone()
        h_threshold_hard, h_threshold_soft = self.highlight_threshold, self.upper_mid_threshold
        s_threshold_hard, s_threshold_soft = self.shadow_threshold, self.lower_mid_threshold
        ones_tensor = torch.ones(img_tensor.shape)
        zeros_tensor = torch.zeros(img_tensor.shape)

        mask_lo = torch.le(img_tensor, h_threshold_soft)
        mask_hi = torch.ge(img_tensor, s_threshold_soft)
        mid_mask = torch.logical_and(mask_hi, mask_lo)
        highlight_ones_mask = torch.gt(img_tensor, h_threshold_hard)
        shadows_ones_mask = torch.lt(img_tensor, s_threshold_hard)
        mask_top_hi = torch.gt(img_tensor, h_threshold_soft)
        mask_top_lo = torch.le(img_tensor, h_threshold_hard)
        mask_top = torch.logical_and(mask_top_hi, mask_top_lo)
        mask_bottom_hi = torch.ge(img_tensor, s_threshold_hard)
        mask_bottom_lo = torch.lt(img_tensor, s_threshold_soft)
        mask_bottom = torch.logical_and(mask_bottom_hi, mask_bottom_lo)

        if not (h_threshold_hard == h_threshold_soft):
            masked = img_tensor[mask_top]
            if 0 < masked.numel():
                vmax_top, vmin_top = masked.max(), masked.min()
                if (vmax_top == vmin_top):
                    img_tensor[mask_top] = 0.5 * ones_tensor
                else:
                    img_tensor[mask_top] = (img_tensor[mask_top] - vmin_top) / (vmax_top - vmin_top) # hi is 1
            
        if not (s_threshold_hard == s_threshold_soft):
            masked = img_tensor[mask_bottom]
            if 0 < masked.numel():
                vmax_bottom, vmin_bottom = masked.max(), masked.min()
                if (vmax_bottom == vmin_bottom):
                    img_tensor[mask_bottom] = 0.5 * ones_tensor
                else:
                    img_tensor[mask_bottom] = torch.sub(1.0, (img_tensor[mask_bottom] - vmin_bottom) / (vmax_bottom - vmin_bottom)) # lo is 1

        img_tensor[mid_mask] = zeros_tensor[mid_mask]
        img_tensor[highlight_ones_mask] = ones_tensor[highlight_ones_mask]
        img_tensor[shadows_ones_mask] = ones_tensor[shadows_ones_mask]

        return img_tensor


    def invoke(self, context: InvocationContext) -> ShadowsHighlightsMidtonesMasksOutput:
        image_in = context.services.images.get_pil_image(self.image.image_name)
        if image_in.mode != "L":
            image_in = image_in.convert("L")
        image_tensor = image_resized_to_grid_as_tensor(image_in, normalize=False)
        h_image_out = pil_image_from_tensor(self.get_highlights_mask(image_tensor), mode="L")
        h_image_dto = context.services.images.create(
            image=h_image_out,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate
        )
        m_image_out = pil_image_from_tensor(self.get_midtones_mask(image_tensor), mode="L")
        m_image_dto = context.services.images.create(
            image=m_image_out,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate
        )
        s_image_out = pil_image_from_tensor(self.get_shadows_mask(image_tensor), mode="L")
        s_image_dto = context.services.images.create(
            image=s_image_out,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate
        )
        return ShadowsHighlightsMidtonesMasksOutput(
            highlights_mask=ImageField(image_name=h_image_dto.image_name),
            midtones_mask=ImageField(image_name=m_image_dto.image_name),
            shadows_mask=ImageField(image_name=s_image_dto.image_name),
            width=h_image_dto.width,
            height=h_image_dto.height
        )
