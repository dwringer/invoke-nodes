from typing import Literal

from PIL import Image, ImageDraw, ImageFont, ImageOps
if not hasattr(Image, 'Resampling'):
    Image.Resampling = Image  # (Compatibilty for Pillow earlier than v9)
from pydantic import  Field


from invokeai.app.models.image import ImageCategory, ImageField, ResourceOrigin
from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    InvocationContext,
    InvocationConfig,
)
from invokeai.app.invocations.image import (
    PILInvocationConfig,
    ImageOutput
)

class TextMaskInvocation(BaseInvocation, PILInvocationConfig):
    """Creates a 2D rendering of a text mask from a given font"""

    # fmt: off
    type: Literal["text_mask"] = "text_mask"

    # Inputs
    width: int = Field(default=512, description="The width of the desired mask")
    height: int = Field(default=512, description="The height of the desired mask")
    text: str = Field(default="", description="The text to render")
    font: str = Field(default="", description="Path to a FreeType-supported TTF/OTF font file")
    size: int = Field(default=64, description="Desired point size of text to use")
    angle: float = Field(default=0.0, description="Angle of rotation to apply to the text")
    x_offset: int = Field(default=24, description="x-offset for text rendering")
    y_offset: int = Field(default=36, description="y-offset for text rendering")
    invert: bool = Field(default=False, description="Whether to invert color of the output")
    # fmt: on

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "title": "Text Mask",
                "tags": ["image", "text", "mask"]
            },
        }

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image_out = Image.new(
            mode="RGBA", size=(self.width, self.height), color=(0, 0, 0, 255)
        )

        font = ImageFont.truetype(self.font, self.size)
        text_writer = ImageDraw.Draw(image_out)
        text_writer.text((self.x_offset, self.y_offset), self.text, font=font, fill=(255, 255, 255, 255))
        if self.angle != 0.0:
            image_out = image_out.rotate(
                self.angle,
                resample=Image.Resampling.BICUBIC,
                expand=False,
                center=(self.x_offset, self.y_offset),
                fillcolor=(0,0,0,255)
            )
        if self.invert:
            image_out = ImageOps.invert(image_out.convert('RGB'))
        image_dto = context.services.images.create(
            image=image_out,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate
        )
        return ImageOutput(image=ImageField(image_name=image_dto.image_name),
                           width=image_dto.width,
                           height=image_dto.height,
        )
