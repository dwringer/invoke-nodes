
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

from PIL import Image, ImageFilter
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
    InputField,
    InvocationContext,
    invocation,
)


@invocation(
    "txt2mask_clipseg",
    title="Text to Mask (Clipseg)",
    tags=["image", "mask", "clip", "clipseg", "txt2mask"],
    category="image",
    version="1.0.0",
)
class TextToMaskClipsegInvocation(BaseInvocation):
    """Uses the Clipseg model to generate an image mask from a text prompt"""

    image: ImageField = InputField(description="The image from which to create a mask")
    prompt: str = InputField(description="The prompt with which to create a mask")
    smoothing: float = InputField(
        default=2.0, description="Radius of blur to apply before thresholding"
    )
    subject_threshold: float = InputField(
        default=0.5, description="Threshold past which is considered the subject"
    )
    background_threshold: float = InputField(
        default=0.5, description="Threshold below which is considered the background"
    )

    def get_threshold_mask(self, image_tensor):
        img_tensor = image_tensor.clone()
        threshold_h, threshold_s = self.subject_threshold, self.background_threshold
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
    

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image_in = context.services.images.get_pil_image(self.image.image_name)
        image_size = image_in.size
        image_out = None

        processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")

        model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

        image_in = image_in.convert("RGB")
        
        input_args = processor(
            text=[self.prompt], images=[image_in], padding="max_length", return_tensors="pt"
        )

        with torch.no_grad():
            output = model(**input_args)
            
        predictions = output.logits.unsqueeze(0)

        image_out = pil_image_from_tensor(torch.sigmoid(predictions[0,:,:]), mode="L")
        image_out = image_out.resize(image_size)
        image_out = image_out.filter(ImageFilter.GaussianBlur(radius=self.smoothing))
        image_out = image_resized_to_grid_as_tensor(image_out, normalize=False)
        image_out = self.get_threshold_mask(image_out)
        image_out = pil_image_from_tensor(image_out)

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
