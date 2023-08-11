import random
from typing import Literal

from pydantic import Field

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    InvocationContext,
    InvocationConfig
)

class RandomSwitchOutput(BaseInvocationOutput):
    """Class to encapsulate output of WxH from random switch node"""

    # fmt: off
    type: Literal["random_switch_output"] = "random_switch_output"
    output_1: int = Field(description="The first output")
    output_2: int = Field(description="The second output")

    # fmt: on

    class Config:
        schema_extra = {"required": ["type", "output_1", "output_2"]}


class FinalSizeOutput(BaseInvocationOutput):
    """Class to encapsulate output of WxH from Final Size & Orientation node"""

    # fmt: off
    type: Literal["final_size_output"] = "final_size_output"
    width: int = Field(description="The value assigned to width")
    height: int = Field(description="The value assigned to height")

    # fmt: on

    class Config:
        schema_extra = {"required": ["type", "width", "height"]}


class RandomSwitchInvocation(BaseInvocation):
    """Randomly switches its two outputs between the two inputs (integers)"""

    # fmt: off
    type: Literal["random_switch"] = "random_switch"

    # Inputs
    integer_a: int = Field(default=None, description="The first input")
    integer_b: int = Field(default=None, description="The second input")

    # fmt: on

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "title": "Random Switch (Integers)",
                "tags": ["random", "switch"]
            },
        }

    def invoke(self, context:InvocationContext) -> RandomSwitchOutput:
        rand = random.random()
        return RandomSwitchOutput(
            output_1=self.integer_a if rand < 0.5 else self.integer_b,
            output_2=self.integer_b if rand < 0.5 else self.integer_a
            )


class FinalSizeAndOrientationInvocation(BaseInvocation):
    """Input a pair of dimensions and choose portrait, landscape, or random orientation"""

    # fmt: off
    type: Literal["final_size_and_orientation"] = "final_size_and_orientation"

    # Inputs
    dimension_a: int = Field(default=None, description="Size of the desired image resolution's first dimension")
    dimension_b: int = Field(default=None, description="Size of the desired image resolution's second dimension")
    orientation: Literal["random", "landscape", "portrait"] = Field(default="landscape", description="Desired orientation for the final image")

    # fmt: on

    class Config(InvocationConfig):
        schema_extra = {
            "ui": {
                "title": "Final Size & Orientation",
                "tags": ["random", "switch", "size", "orientation"]
            },
        }

    def invoke(self, context:InvocationContext) -> FinalSizeOutput:
        output_a, output_b = -1, -1
        if self.orientation == "random":
            rand = random.random()
            output_a = self.dimension_a if rand < 0.5 else self.dimension_b
            output_b = self.dimension_b if rand < 0.5 else self.dimension_a
        else:
            longest, shortest = tuple(sorted([self.dimension_a, self.dimension_b], reverse=True))
            if self.orientation == "portrait":
                output_a, output_b = shortest, longest
            elif self.orientation == "landscape":
                output_a, output_b = longest, shortest
        return FinalSizeOutput(width=output_a, height=output_b)
