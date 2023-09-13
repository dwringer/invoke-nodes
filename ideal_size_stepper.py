import math
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

@invocation_output("ideal_size_stepper_output")
class IdealSizeStepperOutput(BaseInvocationOutput):
    """Class to encapsulate up to three pairs of int outputs corresponding to WxH image sizes"""
    width_a:             int = OutputField(description="The ideal width of the first intermediate image in pixels")
    height_a:            int = OutputField(description="The ideal height of the first intermediate image in pixels")
    width_b:             int = OutputField(description="The ideal width of the second intermediate image in pixels")
    height_b:            int = OutputField(description="The ideal height of the second intermediate image in pixels")
    width_c:             int = OutputField(description="The ideal width of the third intermediate image in pixels")
    height_c:            int = OutputField(description="The ideal height of the third intermediate image in pixels")

TAPERS_IMPLEMENTED: list = [
    "Proportional (log area)",
    "Faster growth (log^2 area)",
    "Even faster (linear diag.)",
    "Largest (linear area)",
]

TAPER_FIELDNAMES: list = [
    "<Taper A>",
    "<Taper B>",
    "<Taper C>",
]
        
@invocation(
    "ideal_size_stepper",
    title="Ideal Size Stepper",
    tags=["math", "size", "upscale"],
    category="math",
    version="1.0.0",
)
class IdealSizeStepperInvocation(BaseInvocation):
    """Calculates the ideal size for intermediate generations given full size and minimum size dimensions"""
    full_width:  int = InputField(default=None,  description="Full size width")
    full_height: int = InputField(default=None,  description="Full size height")
    ideal_width:   int = InputField(default=None,  description="Optimized size width")
    ideal_height:  int = InputField(default=None,  description="Optimized size height")
    taper_a: Literal[tuple(TAPERS_IMPLEMENTED)] = InputField(
        default="Proportional (log area)", description="Taper used for scaling the intermediate dimensions"
    )
    taper_b: Literal[tuple(["<Disabled>"] + TAPER_FIELDNAMES[:1] + TAPERS_IMPLEMENTED)] = InputField(
        default="<Disabled>", description="If enabled, computes second intermediate stage (else, copies A outputs)"
    )
    taper_c: Literal[tuple(["<Disabled>"] + TAPER_FIELDNAMES[:2] + TAPERS_IMPLEMENTED)] = InputField(
        default="<Disabled>", description="If enabled, allocates 3 intermediate stages (else, copies B outputs)"
    )

    def get_v(self, width, height, taper):
        v = None
        if   taper == "Even faster (linear diag.)":
            v = math.sqrt( width**2 + height**2 )
        elif taper == "Largest (linear area)":
            v = width * height
        elif taper == "Proportional (log area)":
            v = math.log( width * height )
        elif taper == "Faster growth (log^2 area)":
            v = math.log( width * height ) ** 2
        return v

    def get_next_h(self, i, dv, v_ideal, taper):
        h = None
        if   taper == "Even faster (linear diag.)":
            h = int( (( v_ideal + ((1+i) * dv) ) * self.ideal_height) /
                     math.sqrt(self.ideal_width**2 + self.ideal_height**2) )
        elif taper == "Largest (linear area)":
            h = int(math.sqrt( (v_ideal + ((1+i) * dv)) /
                               (self.ideal_width / self.ideal_height) ))
        elif taper == "Proportional (log area)":
            h = int(math.sqrt( math.e ** (v_ideal + ((1+i) * dv)) /
                               (self.ideal_width / self.ideal_height) ))
        elif taper == "Faster growth (log^2 area)":
            h = int(math.sqrt( math.e ** math.sqrt(v_ideal + ((1+i) * dv)) /
                               (self.ideal_width / self.ideal_height) ))
        return h

    def invoke(self, context:InvocationContext) -> IdealSizeStepperOutput:
        aspect = self.full_width / self.full_height
        tapers = [self.taper_a, self.taper_b, self.taper_c]
        dims   = [[-1, -1] for t in tapers]
        steps  = len(tapers)

        # Determine total step count by last non-disabled taper selected:
        for t in reversed(tapers):
            if t == "<Disabled>":
                steps -= 1
            else:
                break

        # Copy taper settings to subsequent tapers that reuse them by name:
        for i, taper in enumerate(tapers[1:]):
            if taper != "<Disabled>":
                for j, tfname in enumerate(TAPER_FIELDNAMES[:i+1]):
                    if taper == tfname:
                        k = j
                        while (0 < k) and (tapers[k] == "<Disabled>"):
                            k -= 1
                        tapers[i+1] = tapers[k]
                        break

        # Compute dimensions stages:
        for i in range(len(tapers)):
            if tapers[i] != "<Disabled>":
                v_max = self.get_v(self.full_width, self.full_height, tapers[i])
                v_ideal = self.get_v(self.ideal_width, self.ideal_height, tapers[i])
                deltav = v_max - v_ideal
                dv = deltav / (steps + 1)
                h = self.get_next_h(i, dv, v_ideal, tapers[i])
                h += 8 - (h % 8)
                w = int(h*aspect)
                w -= w % 8
                dims[i] = [w, h]
            else:
                dims[i] = dims[i-1]

        return IdealSizeStepperOutput(
            width_a=dims[0][0], height_a=dims[0][1],
            width_b=dims[1][0], height_b=dims[1][1],
            width_c=dims[2][0], height_c=dims[2][1],
        )
