from ast import literal_eval as tuple_from_string
from functools import reduce
from typing import Literal, Optional

from PIL import Image, ImageOps, ImageChops, ImageDraw, ImageColor
from pydantic import Field

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


class ImageCompositorInvocation(BaseInvocation, PILInvocationConfig):
    """Removes backdrop from subject image then overlays subject on background image"""
    # fmt: off
    type: Literal["img_composite"] = "img_composite"

    # Inputs
    image_subject:    Optional[ImageField] = Field(
        default=None, description="Image of the subject on a plain monochrome background"
    )
    image_background: Optional[ImageField] = Field(
        default=None, description="Image of a background scene"
    )
    chroma_key: str = Field(
        default="", description="Can be empty for corner flood select, or CSS-3 color or tuple"
    )
    threshold:  int = Field(
        default=50, description="Subject isolation flood-fill threshold"
    )
    fill_x:    bool = Field(default=False, description="Scale base subject image to fit background width")
    fill_y:    bool = Field(default=True,  description="Scale base subject image to fit background height")
    x_offset:   int = Field(default=0, description="x-offset for the subject")
    y_offset:   int = Field(default=0, description="y-offset for the subject")
    # fmt: on

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "title": "Image Compositor",
                "tags": ["image", "compose", "chroma", "key"]
            },
        }

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image_background = context.services.images.get_pil_image(self.image_background.image_name).convert(mode="RGBA")
        image_subject    = context.services.images.get_pil_image(self.image_subject.image_name).convert(mode="RGBA")
        subject_aspect   = image_subject.width / image_subject.height

        # Handle backdrop removal:
        chroma_key = self.chroma_key.strip()
        if 0 < len(chroma_key):
            # Remove pixels by chroma key:
            if chroma_key[0] == "(":
                chroma_key = tuple_from_string(chroma_key)
                while len(chroma_key) < 3:
                    chroma_key = tuple(list(chroma_key) + [0])
                if len(chroma_key) == 3:
                    chroma_key = tuple(list(chroma_key) + [255])
            else:
                chroma_key = ImageColor.getcolor(chroma_key, "RGBA")
            threshold = self.threshold ** 2.0  # to compare vs squared color distance from key
            pixels = image_subject.load()
            for i in range(image_subject.width):
                for j in range(image_subject.height):
                    if reduce(
                            lambda a, b: a + b, [(pixels[i, j][k] - chroma_key[k])**2 for k in range(len(chroma_key))]
                    ) < threshold:
                        pixels[i, j] = tuple([0 for k in range(len(chroma_key))])
        else:
            # Remove pixels by flood select from corners:
            ImageDraw.floodfill(image_subject, (0,                     0),                      (0, 0, 0, 0), thresh=self.threshold)
            ImageDraw.floodfill(image_subject, (0,                     image_subject.height-1), (0, 0, 0, 0), thresh=self.threshold)
            ImageDraw.floodfill(image_subject, (image_subject.width-1, 0),                      (0, 0, 0, 0), thresh=self.threshold)
            ImageDraw.floodfill(image_subject, (image_subject.width-1, image_subject.height-1), (0, 0, 0, 0), thresh=self.threshold)

        # Scale and position the subject:
        aspect_background = image_background.width / image_background.height
        aspect_subject    = image_subject.width / image_subject.height
        if self.fill_x and self.fill_y:
            image_subject = image_subject.resize((image_background.width, image_background.height))
        elif ((self.fill_x and (aspect_background < aspect_subject)) or
              (self.fill_y and (aspect_subject <= aspect_background))):
            image_subject = ImageOps.pad(image_subject, (image_background.width, image_background.height), color=(0, 0, 0, 0))
        elif ((self.fill_x and (aspect_subject <= aspect_background)) or
              (self.fill_y and (aspect_background < aspect_subject))):
            image_subject = ImageOps.fit(image_subject, (image_background.width, image_background.height))
        if (self.x_offset != 0) or (self.y_offset != 0):
            image_subject = ImageChops.offset(image_subject, self.x_offset, yoffset=-1*self.y_offset)

        new_image = Image.alpha_composite(image_background, image_subject)
        new_image.convert(mode="RGB")
        image_dto = context.services.images.create(
            image=new_image,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
        )

        return ImageOutput(
            image=ImageField(image_name=image_dto.image_name),
            width=image_dto.width,
            height=image_dto.height,
        )
