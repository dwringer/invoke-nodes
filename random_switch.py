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
