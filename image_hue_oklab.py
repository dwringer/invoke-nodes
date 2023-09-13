from math import pi as PI

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
    "img_hue_rotate_oklab",
    title="Adjust Image Hue (Oklab)",
    tags=["image", "hue", "oklab", "lch"],
    category="image",
    version="1.0.0",
)
class AdjustImageHueOklabInvocation(BaseInvocation):
    """Adjusts the Hue of an image by rotating it in Oklab space (perceptually uniform but non-reversible)"""
    
    image: ImageField = InputField(description="The image to adjust")
    degrees: float = InputField(default=0.0, description="Degrees by which to rotate image hue")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image_in = context.services.images.get_pil_image(self.image.image_name)

        rgb_tensor = image_resized_to_grid_as_tensor(image_in, normalize=False)  # 0..1 values

        # gamma expansion of sRGB to linear-light RGB
        linear_rgb_tensor = torch.pow(torch.div(torch.add(rgb_tensor, 0.055), 1.055), 2.4)
        linear_rgb_tensor_1 = torch.div(rgb_tensor, 12.92)
        mask = torch.le(rgb_tensor, 0.0404482362771082)
        linear_rgb_tensor[mask] = linear_rgb_tensor_1[mask]

        # linear RGB to LMS
        lms_matrix = torch.tensor([[0.4122214708, 0.5363325363, 0.0514459929],
                                   [0.2119034982, 0.6806995451, 0.1073969566],
                                   [0.0883024619, 0.2817188376, 0.6299787005]])

        lms_tensor = torch.einsum('cwh, kc -> kwh', linear_rgb_tensor, lms_matrix)

        # LMS to L*a*b*
        lms_tensor_1 = torch.pow(lms_tensor, 1./3.)
        lab_matrix = torch.tensor([[0.2104542553,  0.7936177850, -0.0040720468],
                                   [1.9779984951, -2.4285922050,  0.4505937099],
                                   [0.0259040371,  0.7827717662, -0.8086757660]])
        
        lab_tensor = torch.einsum('kwh, lk -> lwh', lms_tensor_1, lab_matrix)

        # L*a*b* to L*C*h
        c_tensor = torch.sqrt(torch.add(torch.pow(lab_tensor[1,:,:], 2.0), torch.pow(lab_tensor[2,:,:], 2.0)))
        h_tensor = torch.atan2(lab_tensor[2,:,:], lab_tensor[1,:,:])
        
        # Rotate h
        rot_rads = (self.degrees / 180.0)*PI
        
        h_rot = torch.add(h_tensor, rot_rads)
        h_rot = torch.sub(torch.remainder(torch.add(h_rot, PI), 2*PI), PI)

        # L*C*h to L*a*b*
        lab_tensor[1,:,:] = torch.mul(c_tensor, torch.cos(h_rot))
        lab_tensor[2,:,:] = torch.mul(c_tensor, torch.sin(h_rot))

        # L*a*b* to LMS
        lms_matrix_1 = torch.tensor([[1.,  0.3963377774,  0.2158037573],
                                     [1., -0.1055613458, -0.0638541728],
                                     [1., -0.0894841775, -1.2914855480]])

        lms_tensor_1 = torch.einsum('lwh, kl -> kwh', lab_tensor, lms_matrix_1)
        lms_tensor = torch.pow(lms_tensor_1, 3.)

        # LMS to linear RGB
        rgb_matrix = torch.tensor([[ 4.0767416621, -3.3077115913,  0.2309699292],
                                   [-1.2684380046,  2.6097574011, -0.3413193965],
                                   [-0.0041960863, -0.7034186147,  1.7076147010]])

        linear_rgb_tensor = torch.einsum('kwh, sk -> swh', lms_tensor, rgb_matrix)

        # Restrict color values to what can be displayed with RGB
        linear_rgb_tensor = linear_rgb_tensor.clamp(0., 1.)

        # gamma correction of linear RGB to sRGB
        mask = torch.lt(linear_rgb_tensor, 0.0404482362771082 / 12.92)
        
        rgb_tensor = torch.sub(torch.mul(torch.pow(linear_rgb_tensor, (1/2.4)), 1.055), 0.055)
        rgb_tensor[mask] = torch.mul(linear_rgb_tensor[mask], 12.92)

        image_out = pil_image_from_tensor(rgb_tensor, mode="RGB")

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

