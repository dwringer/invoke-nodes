import random
from typing import Literal

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    InvocationContext,
    InputField,
    OutputField,
    invocation,
    invocation_output
)

@invocation_output("random_switch_output")
class RandomSwitchOutput(BaseInvocationOutput):
    """Class to encapsulate output of WxH from random switch node"""
    output_1: int = OutputField(description="The first output")
    output_2: int = OutputField(description="The second output")


@invocation_output("final_size_output")
class FinalSizeOutput(BaseInvocationOutput):
    """Class to encapsulate output of WxH from Final Size & Orientation node"""
    width: int = OutputField(description="The value assigned to width")
    height: int = OutputField(description="The value assigned to height")


@invocation(
    "random_switch",
    title="Random Switch (Integers)",
    tags=["random", "switch"],
    category="math",
    version="1.0.0",
)
class RandomSwitchInvocation(BaseInvocation):
    """Randomly switches its two outputs between the two inputs (integers)"""
    integer_a: int = InputField(default=None, description="The first input")
    integer_b: int = InputField(default=None, description="The second input")

    def invoke(self, context:InvocationContext) -> RandomSwitchOutput:
        rand = random.random()
        return RandomSwitchOutput(
            output_1=self.integer_a if rand < 0.5 else self.integer_b,
            output_2=self.integer_b if rand < 0.5 else self.integer_a
            )


@invocation(
    "final_size_and_orientation",
    title="Final Size & Orientation",
    tags=["random", "switch", "size", "orientation"],
    category="math",
    version="1.0.0",
)
class FinalSizeAndOrientationInvocation(BaseInvocation):
    """Input a pair of dimensions and choose portrait, landscape, or random orientation"""
    # Inputs
    dimension_a: int = InputField(default=None, description="Size of the desired image resolution's first dimension")
    dimension_b: int = InputField(default=None, description="Size of the desired image resolution's second dimension")
    orientation: Literal["random", "landscape", "portrait"] = InputField(default="landscape", description="Desired orientation for the final image")

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
