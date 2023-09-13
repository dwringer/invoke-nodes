from math import log, pi as PI
import os.path

import PIL.Image, PIL.ImageCms
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
    "img_hue_rotate",
    title="Adjust Image Hue (CIELAB)",
    tags=["image", "hue", "cielab", "lab"],
    category="image",
    version="1.0.0",
)
class ImageHueRotationInvocation(BaseInvocation):
    """Adjusts the Hue of an image by rotating it in CIELAB L*C*h polar coordinates"""
    
    image: ImageField = InputField(description="The image to adjust")
    degrees: float = InputField(default=0.0, description="Degrees by which to rotate image hue in CIELAB space")

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image_in = context.services.images.get_pil_image(self.image.image_name)

        image_out = image_in.convert("RGB")

        profile_srgb = PIL.ImageCms.createProfile("sRGB")
        profile_lab = None
        profile_uplab = None
        if os.path.isfile("CIELab_to_UPLab.icc"):
            profile_uplab = PIL.ImageCms.getOpenProfile("CIELab_to_UPLab.icc")
        if profile_uplab is None:
            profile_lab = PIL.ImageCms.createProfile("LAB", colorTemp=6500)
        else:
            profile_lab = PIL.ImageCms.createProfile("LAB", colorTemp=5000)

        lab_transform = PIL.ImageCms.buildTransformFromOpenProfiles(
            profile_srgb, profile_lab, "RGB", "LAB", renderingIntent=2, flags=0x2400
        )
        image_out = PIL.ImageCms.applyTransform(image_out, lab_transform)
        if not (profile_uplab is None):
          uplab_transform = PIL.ImageCms.buildTransformFromOpenProfiles(
              profile_lab, profile_uplab, "LAB", "LAB", flags=0x2400
          )
          image_out = PIL.ImageCms.applyTransform(image_out, uplab_transform)
        
        channel_l = image_out.getchannel("L")
        channel_a = image_out.getchannel("A")
        channel_b = image_out.getchannel("B")

        l_tensor = image_resized_to_grid_as_tensor(channel_l, normalize=False)
        a_tensor = image_resized_to_grid_as_tensor(channel_a, normalize=True)
        b_tensor = image_resized_to_grid_as_tensor(channel_b, normalize=True)

        # L*a*b* to L*C*h
        c_tensor = torch.sqrt(torch.add(torch.pow(a_tensor, 2.0), torch.pow(b_tensor, 2.0)))
        h_tensor = torch.atan2(b_tensor, a_tensor)
        
        # Rotate h
        rot_rads = (self.degrees / 180.0)*PI
        
        h_rot = torch.add(h_tensor, rot_rads)
        h_rot = torch.sub(torch.remainder(torch.add(h_rot, PI), 2*PI), PI)

        # L*C*h to L*a*b*
        a_tensor = torch.mul(c_tensor, torch.cos(h_rot))
        b_tensor = torch.mul(c_tensor, torch.sin(h_rot))
        
        # -1..1 -> 0..1 for all elts of a, b
        a_tensor = torch.div(torch.add(a_tensor, 1.0), 2.0)
        b_tensor = torch.div(torch.add(b_tensor, 1.0), 2.0)

        l_img = pil_image_from_tensor(l_tensor)
        a_img = pil_image_from_tensor(a_tensor)
        b_img = pil_image_from_tensor(b_tensor)
        
        image_out = PIL.Image.merge("LAB", (l_img, a_img, b_img))

        if not (profile_uplab is None):
            deuplab_transform = PIL.ImageCms.buildTransformFromOpenProfiles(
                profile_uplab, profile_lab, "LAB", "LAB", flags=0x2400
            )
            image_out = PIL.ImageCms.applyTransform(image_out, deuplab_transform)

        rgb_transform = PIL.ImageCms.buildTransformFromOpenProfiles(
            profile_lab, profile_srgb, "LAB", "RGB", renderingIntent=2, flags=0x2400
        )
        image_out = PIL.ImageCms.applyTransform(image_out, rgb_transform)

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

