import math
from typing import Literal

from pydantic import BaseModel, Field
import numpy as np

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    InvocationContext,
    InvocationConfig,
)


class IdealSizeStepperOutput(BaseInvocationOutput):
    """Class to encapsulate up to three pairs of int outputs corresponding to WxH image sizes"""

    # fmt: off
    type: Literal["ideal_size_stepper_output"] = "ideal_size_stepper_output"
    width_a:             int = Field(description="The ideal width of the first intermediate image in pixels")
    height_a:            int = Field(description="The ideal height of the first intermediate image in pixels")
    width_b:             int = Field(description="The ideal width of the second intermediate image in pixels")
    height_b:            int = Field(description="The ideal height of the second intermediate image in pixels")
    width_c:             int = Field(description="The ideal width of the third intermediate image in pixels")
    height_c:            int = Field(description="The ideal height of the third intermediate image in pixels")
    # fmt: on

    class Config:
        schema_extra = {"required": ["type", "width_a", "height_a", "width_b", "height_b", "width_c", "height_c"]}


class IdealSizeStepperInvocation(BaseInvocation):
    """Calculates the ideal size for intermediate generations given full size and minimum size dimensions"""

    # fmt: off
    type: Literal["ideal_size_stepper"] = "ideal_size_stepper"

    # Inputs
    full_width:  int = Field(default=None,  description="Full size width")
    full_height: int = Field(default=None,  description="Full size height")
    ideal_width:   int = Field(default=None,  description="Optimized size width")
    ideal_height:  int = Field(default=None,  description="Optimized size height")
    stage_b:   bool = Field(default=False, description="Output two intermediates (A, B)")
    stage_c:   bool = Field(default=False, description="Output all three intermediates (A, B, C)")
    # fmt: on
    
    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "title": "Ideal Size Stepper",
                "tags": ["math", "size", "upscale"]
            },
        }

    def invoke(self, context:InvocationContext) -> IdealSizeStepperOutput:
        aspect = self.full_width / self.full_height
        dims   = [[-1, -1], [-1, -1], [-1, -1]]
        steps  = 3 if self.stage_c else (2 if self.stage_b else 1)

        maxArea = self.full_width * self.full_height
        optArea = self.ideal_width * self.ideal_height
        
        maxAreaLog = math.log(maxArea)
        optAreaLog = math.log(optArea)
        dl = maxAreaLog - optAreaLog
        stepLog = dl / (steps + 1)
        
        for i in range(steps):
            h = int(math.sqrt( (math.e ** (optAreaLog + ((1+i) * stepLog))) / aspect ))
            h += 8 - (h % 8)
            w = int(h * aspect)
            w -= w % 8
            dims[i] = [w, h]

        return IdealSizeStepperOutput(width_a=dims[0][0],
                                      height_a=dims[0][1],
                                      width_b=dims[1][0],
                                      height_b=dims[1][1],
                                      width_c=dims[2][0],
                                      height_c=dims[2][1])
