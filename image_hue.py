# Copyright (c) 2023 Darren Ringer <dwringer@gmail.com>
# Parts based on Oklab: Copyright (c) 2021 Björn Ottosson <https://bottosson.github.io/>

import os.path
from math import sqrt, pi as PI
from typing import Literal

import numpy
import PIL.Image
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


MAX_FLOAT = torch.finfo(torch.tensor(1.).dtype).max


def srgb_from_linear_srgb(linear_srgb_tensor, alpha=0.05, steps=1):
    if 0 < alpha:
        linear_srgb_tensor = gamut_clip_tensor(linear_srgb_tensor, alpha=alpha, steps=steps)
    linear_srgb_tensor = linear_srgb_tensor.clamp(0., 1.)
    mask = torch.lt(linear_srgb_tensor, 0.0404482362771082 / 12.92)
    rgb_tensor = torch.sub(torch.mul(torch.pow(linear_srgb_tensor, (1/2.4)), 1.055), 0.055)
    rgb_tensor[mask] = torch.mul(linear_srgb_tensor[mask], 12.92)

    return rgb_tensor
    

def linear_srgb_from_srgb(srgb_tensor):
    linear_srgb_tensor = torch.pow(torch.div(torch.add(srgb_tensor, 0.055), 1.055), 2.4)
    linear_srgb_tensor_1 = torch.div(srgb_tensor, 12.92)
    mask = torch.le(srgb_tensor, 0.0404482362771082)
    linear_srgb_tensor[mask] = linear_srgb_tensor_1[mask]

    return linear_srgb_tensor

COLOR_SPACES = [
    # "Okhsl",  # Not yet implemented
    "Okhsv",
    "HSV",
    "Oklab",
    "CIELAB",
    "UPLab (w/CIELab_to_UPLab.icc)",
]

@invocation(
    "img_hue_adjust_plus",
    title="Adjust Image Hue Plus",
    tags=["image", "hue", "oklab", "cielab", "uplab", "lch", "hsv", "hsl", "lab"],
    category="image",
    version="1.0.0",
)
class AdjustImageHuePlusInvocation(BaseInvocation):
    """Adjusts the Hue of an image by rotating it in the selected color space"""

    image: ImageField = InputField(description="The image to adjust")
    space: Literal[tuple(COLOR_SPACES)] = InputField(
        default=COLOR_SPACES[0], description="Color space in which to rotate hue by polar coords."
    )
    degrees: float = InputField(default=0.0, description="Degrees by which to rotate image hue")
    ok_adaptive_gamut: float = InputField(
        default=0.5, description="Lower preserves lightness at the expense of chroma (Oklab/Okhsv)"
    )
    ok_high_precision: bool = InputField(
        default=True, description="Use more steps in computing gamut (Oklab/Okhsv)"
    )

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image_in = context.services.images.get_pil_image(self.image.image_name)
        image_out = None
        image_mode = image_in.mode
        image_in = image_in.convert("RGB")

        space = self.space.split()[0].lower()
        if space == "hsv":
            hsv_tensor = image_resized_to_grid_as_tensor(image_in.convert('HSV'), normalize=False)  # 0..1 vals
            hsv_tensor[0,:,:] = torch.remainder(torch.add(hsv_tensor[0,:,:], torch.div(self.degrees, 360.)), 1.)
            image_out = pil_image_from_tensor(hsv_tensor, mode="HSV")
            
        elif space == "okhsv":
            rgb_tensor = image_resized_to_grid_as_tensor(image_in.convert("RGB"), normalize=False)  # 0..1 vals

            hsv_tensor = okhsv_from_srgb(rgb_tensor, steps=(3 if self.ok_high_precision else 1))

            h_tensor = hsv_tensor[0,:,:]

            h_rot = torch.remainder(torch.add(h_tensor, torch.div(self.degrees, 360.)), 1.)

            hsv_tensor[0,:,:] = h_rot

            rgb_tensor = srgb_from_okhsv(hsv_tensor, alpha=(0.05 if self.ok_high_precision else 0.0))

            image_out = pil_image_from_tensor(rgb_tensor, mode="RGB")

        elif (space == "cielab") or (space == "uplab"):
            profile_srgb = PIL.ImageCms.createProfile("sRGB")
            profile_lab = None
            profile_uplab = None
            if space == "uplab":
                if os.path.isfile("CIELab_to_UPLab.icc"):
                    profile_uplab = PIL.ImageCms.getOpenProfile("CIELab_to_UPLab.icc")
            if profile_uplab is None:
                profile_lab = PIL.ImageCms.createProfile("LAB", colorTemp=6500)
            else:
                profile_lab = PIL.ImageCms.createProfile("LAB", colorTemp=5000)

            lab_transform = PIL.ImageCms.buildTransformFromOpenProfiles(
                profile_srgb, profile_lab, "RGB", "LAB", renderingIntent=2, flags=0x2400
            )
            image_out = PIL.ImageCms.applyTransform(image_in, lab_transform)
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

        elif space == "oklab":
            rgb_tensor = image_resized_to_grid_as_tensor(image_in.convert("RGB"), normalize=False)  # 0..1 values

            linear_srgb_tensor = linear_srgb_from_srgb(rgb_tensor)

            lab_tensor = oklab_from_linear_srgb(linear_srgb_tensor)

            # L*a*b* to L*C*h
            c_tensor = torch.sqrt(torch.add(torch.pow(lab_tensor[1,:,:], 2.0),
                                            torch.pow(lab_tensor[2,:,:], 2.0)))
            h_tensor = torch.atan2(lab_tensor[2,:,:], lab_tensor[1,:,:])

            # Rotate h
            rot_rads = (self.degrees / 180.0)*PI

            h_rot = torch.add(h_tensor, rot_rads)
            h_rot = torch.remainder(torch.add(h_rot, 2*PI), 2*PI)

            # L*C*h to L*a*b*
            lab_tensor[1,:,:] = torch.mul(c_tensor, torch.cos(h_rot))
            lab_tensor[2,:,:] = torch.mul(c_tensor, torch.sin(h_rot))

            linear_srgb_tensor = linear_srgb_from_oklab(lab_tensor)

            rgb_tensor = srgb_from_linear_srgb(
                linear_srgb_tensor, alpha=self.ok_adaptive_gamut, steps=(3 if self.ok_high_precision else 1)
            )
        
            image_out = pil_image_from_tensor(rgb_tensor, mode="RGB")

        image_out = image_out.convert(image_mode)

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


def max_srgb_saturation_tensor(units_ab_tensor, steps=1):
    rgb_k_matrix = torch.tensor([[1.19086277,  1.76576728,  0.59662641,  0.75515197, 0.56771245],
                                 [0.73956515, -0.45954494,  0.08285427,  0.12541070, 0.14503204],
                                 [1.35733652, -0.00915799, -1.15130210, -0.50559606, 0.00692167]])
   
    rgb_w_matrix = torch.tensor([[ 4.0767416621, -3.3077115913,  0.2309699292],
                                 [-1.2684380046,  2.6097574011, -0.3413193965],
                                 [-0.0041960863, -0.7034186147,  1.7076147010]])

    rgb_index_firstout_tensor = torch.empty(units_ab_tensor.shape[1:])
    cond_r_tensor = torch.add(torch.mul(-1.88170328, units_ab_tensor[0,:,:]),
                              torch.mul(-0.80936493, units_ab_tensor[1,:,:]))
    cond_g_tensor = torch.add(torch.mul(1.81444104, units_ab_tensor[0,:,:]),
                              torch.mul(-1.19445276, units_ab_tensor[1,:,:]))

    terms_tensor = torch.stack([torch.ones(units_ab_tensor.shape[1:]),
                                units_ab_tensor[0,:,:],
                                units_ab_tensor[1,:,:],
                                torch.pow(units_ab_tensor[0,:,:], 2.),
                                torch.mul(units_ab_tensor[0,:,:],
                                          units_ab_tensor[1,:,:])])

    s_tensor = torch.empty(units_ab_tensor.shape[1:])
    s_tensor = torch.where(
        torch.gt(cond_r_tensor, 1.),
        torch.einsum('twh, t -> wh', terms_tensor, rgb_k_matrix[0]),
        torch.where(torch.gt(cond_g_tensor, 1.),
                    torch.einsum('twh, t -> wh', terms_tensor, rgb_k_matrix[1]),
                    torch.einsum('twh, t -> wh', terms_tensor, rgb_k_matrix[2])))
    
    k_lms_matrix = torch.tensor([[ 0.3963377774,  0.2158037573],
                                 [-0.1055613458, -0.0638541728],
                                 [-0.0894841775, -1.2914855480]])

    k_lms_tensor = torch.einsum('tc, cwh -> twh', k_lms_matrix, units_ab_tensor)

    for i in range(steps):
        root_lms_tensor = torch.add(torch.mul(k_lms_tensor, s_tensor), 1.)
        lms_tensor = torch.pow(root_lms_tensor, 3.)
        lms_ds_tensor = torch.mul(torch.mul(k_lms_tensor, torch.pow(root_lms_tensor, 2.)), 3.)
        lms_ds2_tensor = torch.mul(torch.mul(torch.pow(k_lms_tensor, 2.), root_lms_tensor), 6.)
        f_tensor = torch.where(
            torch.gt(cond_r_tensor, 1.),
            torch.einsum('c, cwh -> wh', rgb_w_matrix[0], lms_tensor),
            torch.where(torch.gt(cond_g_tensor, 1.),
                        torch.einsum('c, cwh -> wh', rgb_w_matrix[1], lms_tensor),
                        torch.einsum('c, cwh -> wh', rgb_w_matrix[2], lms_tensor)))
        f_tensor_1 = torch.where(
            torch.gt(cond_r_tensor, 1.),
            torch.einsum('c, cwh -> wh', rgb_w_matrix[0], lms_ds_tensor),
            torch.where(torch.gt(cond_g_tensor, 1.),
                        torch.einsum('c, cwh -> wh', rgb_w_matrix[1], lms_ds_tensor),
                        torch.einsum('c, cwh -> wh', rgb_w_matrix[2], lms_ds_tensor)))
        f_tensor_2 = torch.where(
            torch.gt(cond_r_tensor, 1.),
            torch.einsum('c, cwh -> wh', rgb_w_matrix[0], lms_ds2_tensor),
            torch.where(torch.gt(cond_g_tensor, 1.),
                        torch.einsum('c, cwh -> wh', rgb_w_matrix[1], lms_ds2_tensor),
                        torch.einsum('c, cwh -> wh', rgb_w_matrix[2], lms_ds2_tensor)))
        s_tensor = torch.sub(s_tensor,
                             torch.div(torch.mul(f_tensor, f_tensor_1),
                                       torch.sub(torch.pow(f_tensor_1, 2.),
                                                 torch.mul(torch.mul(f_tensor, f_tensor_2), 0.5))))

    return s_tensor


def linear_srgb_from_oklab(oklab_tensor):
    # L*a*b* to LMS
    lms_matrix_1 = torch.tensor([[1.,  0.3963377774,  0.2158037573],
                                 [1., -0.1055613458, -0.0638541728],
                                 [1., -0.0894841775, -1.2914855480]])

    lms_tensor_1 = torch.einsum('lwh, kl -> kwh', oklab_tensor, lms_matrix_1)
    lms_tensor = torch.pow(lms_tensor_1, 3.)

    # LMS to linear RGB
    rgb_matrix = torch.tensor([[ 4.0767416621, -3.3077115913,  0.2309699292],
                               [-1.2684380046,  2.6097574011, -0.3413193965],
                               [-0.0041960863, -0.7034186147,  1.7076147010]])

    linear_srgb_tensor = torch.einsum('kwh, sk -> swh', lms_tensor, rgb_matrix)

    return linear_srgb_tensor
    

def oklab_from_linear_srgb(linear_srgb_tensor):
    # linear RGB to LMS
    lms_matrix = torch.tensor([[0.4122214708, 0.5363325363, 0.0514459929],
                               [0.2119034982, 0.6806995451, 0.1073969566],
                               [0.0883024619, 0.2817188376, 0.6299787005]])

    lms_tensor = torch.einsum('cwh, kc -> kwh', linear_srgb_tensor, lms_matrix)

    # LMS to L*a*b*
    lms_tensor_neg_mask = torch.lt(lms_tensor, 0.)
    lms_tensor[lms_tensor_neg_mask] = torch.mul(lms_tensor[lms_tensor_neg_mask], -1.)
    lms_tensor_1 = torch.pow(lms_tensor, 1./3.)
    lms_tensor[lms_tensor_neg_mask] = torch.mul(lms_tensor[lms_tensor_neg_mask], -1.)
    lms_tensor_1[lms_tensor_neg_mask] = torch.mul(lms_tensor_1[lms_tensor_neg_mask], -1.)
    lab_matrix = torch.tensor([[0.2104542553,  0.7936177850, -0.0040720468],
                               [1.9779984951, -2.4285922050,  0.4505937099],
                               [0.0259040371,  0.7827717662, -0.8086757660]])

    lab_tensor = torch.einsum('kwh, lk -> lwh', lms_tensor_1, lab_matrix)

    return lab_tensor


def find_cusp_tensor(units_ab_tensor, steps=1):
    s_cusp_tensor = max_srgb_saturation_tensor(units_ab_tensor, steps=steps)

    oklab_tensor = torch.stack([torch.ones(s_cusp_tensor.shape),
                                torch.mul(s_cusp_tensor, units_ab_tensor[0,:,:]),
                                torch.mul(s_cusp_tensor, units_ab_tensor[1,:,:])])

    rgb_at_max_tensor = linear_srgb_from_oklab(oklab_tensor)

    l_cusp_tensor = torch.pow(torch.div(1., rgb_at_max_tensor.max(0).values), 1./3.)
    c_cusp_tensor = torch.mul(l_cusp_tensor, s_cusp_tensor)

    return torch.stack([l_cusp_tensor, c_cusp_tensor])


def find_gamut_intersection_tensor(units_ab_tensor, l_1_tensor, c_1_tensor, l_0_tensor, steps=1, steps_outer=1):
    lc_cusps_tensor = find_cusp_tensor(units_ab_tensor, steps=steps)

    # if (((l_1 - l_0) * c_cusp -
    #      (l_cusp - l_0) * c_1) <= 0.):
    cond_tensor = torch.sub(torch.mul(torch.sub(l_1_tensor, l_0_tensor), lc_cusps_tensor[1,:,:]),
                            torch.mul(torch.sub(lc_cusps_tensor[0,:,:], l_0_tensor), c_1_tensor))
    
    t_tensor = torch.where(
        torch.le(cond_tensor, 0.),  # cond <= 0

        #  t = (c_cusp * l_0) /
        #      ((c_1 * l_cusp) + (c_cusp * (l_0 - l_1)))
        torch.div(torch.mul(lc_cusps_tensor[1,:,:], l_0_tensor),
                  torch.add(torch.mul(c_1_tensor, lc_cusps_tensor[0,:,:]),
                            torch.mul(lc_cusps_tensor[1,:,:],
                                      torch.sub(l_0_tensor, l_1_tensor)))),

        # t = (c_cusp * (l_0-1.)) /
        #     ((c_1 * (l_cusp-1.)) + (c_cusp * (l_0 - l_1)))
        torch.div(torch.mul(lc_cusps_tensor[1,:,:], torch.sub(l_0_tensor, 1.)),
                  torch.add(torch.mul(c_1_tensor, torch.sub(lc_cusps_tensor[0,:,:], 1.)),
                            torch.mul(lc_cusps_tensor[1,:,:],
                                      torch.sub(l_0_tensor, l_1_tensor))))
    )

    for i in range(steps_outer):
        dl_tensor = torch.sub(l_1_tensor, l_0_tensor)
        dc_tensor = c_1_tensor

        k_lms_matrix = torch.tensor([[ 0.3963377774,  0.2158037573],
                                     [-0.1055613458, -0.0638541728],
                                     [-0.0894841775, -1.2914855480]])
        k_lms_tensor = torch.einsum('tc, cwh -> twh', k_lms_matrix, units_ab_tensor)

        lms_dt_tensor = torch.add(torch.mul(k_lms_tensor, dc_tensor), dl_tensor)

        for j in range(steps):

            
            l_tensor = torch.add(torch.mul(l_0_tensor, torch.add(torch.mul(t_tensor, -1.), 1.)),
                                 torch.mul(t_tensor, l_1_tensor))
            c_tensor = torch.mul(t_tensor, c_1_tensor)

            root_lms_tensor = torch.add(torch.mul(k_lms_tensor, c_tensor), l_tensor)

            lms_tensor = torch.pow(root_lms_tensor, 3.)
            lms_dt_tensor_1 = torch.mul(torch.mul(torch.pow(root_lms_tensor, 2.), lms_dt_tensor), 3.)
            lms_dt2_tensor = torch.mul(torch.mul(torch.pow(lms_dt_tensor, 2.), root_lms_tensor), 6.)
            
            rgb_matrix = torch.tensor([[ 4.0767416621, -3.3077115913,  0.2309699292],
                                       [-1.2684380046,  2.6097574011, -0.3413193965],
                                       [-0.0041960863, -0.7034186147,  1.7076147010]])

            rgb_tensor = torch.sub(torch.einsum('qt, twh -> qwh', rgb_matrix, lms_tensor), 1.)
            rgb_tensor_1 = torch.einsum('qt, twh -> qwh', rgb_matrix, lms_dt_tensor_1)
            rgb_tensor_2 = torch.einsum('qt, twh -> qwh', rgb_matrix, lms_dt2_tensor)

            u_rgb_tensor = torch.div(rgb_tensor_1,
                                     torch.sub(torch.pow(rgb_tensor_1, 2.),
                                               torch.mul(torch.mul(rgb_tensor, rgb_tensor_2), 0.5)))

            t_rgb_tensor = torch.mul(torch.mul(rgb_tensor, -1.), u_rgb_tensor)

            max_floats = torch.mul(MAX_FLOAT, torch.ones(t_rgb_tensor.shape))
            
            t_rgb_tensor = torch.where(torch.lt(u_rgb_tensor, 0.), max_floats, t_rgb_tensor)

            t_tensor = torch.where(
                torch.gt(cond_tensor, 0.),
                torch.add(t_tensor, t_rgb_tensor.min(0).values),
                t_tensor
            )
    
    return t_tensor


def gamut_clip_tensor(rgb_tensor, alpha=0.05, steps=1, steps_outer=1):
    lab_tensor = oklab_from_linear_srgb(rgb_tensor)
    epsilon = 0.00001
    chroma_tensor = torch.sqrt(torch.add(torch.pow(lab_tensor[1,:,:], 2.), torch.pow(lab_tensor[2,:,:], 2.)))
    chroma_tensor = torch.where(torch.lt(chroma_tensor, epsilon), epsilon, chroma_tensor)
    
    units_ab_tensor = torch.div(lab_tensor[1:,:,:], chroma_tensor)

    l_d_tensor = torch.sub(lab_tensor[0], 0.5)
    e_1_tensor = torch.add(torch.add(torch.abs(l_d_tensor), torch.mul(chroma_tensor, alpha)), 0.5)
    l_0_tensor = torch.mul(
        torch.add(torch.mul(torch.sign(l_d_tensor),
                            torch.sub(e_1_tensor,
                                      torch.sqrt(torch.sub(torch.pow(e_1_tensor, 2.),
                                                           torch.mul(torch.abs(l_d_tensor), 2.))))),
                  1.),
        0.5)

    t_tensor = find_gamut_intersection_tensor(
        units_ab_tensor, lab_tensor[0,:,:], chroma_tensor, l_0_tensor, steps=steps, steps_outer=steps_outer
    )
    l_clipped_tensor = torch.add(torch.mul(l_0_tensor, torch.add(torch.mul(t_tensor, -1), 1.)),
                                 torch.mul(t_tensor, lab_tensor[0,:,:]))
    c_clipped_tensor = torch.mul(t_tensor, chroma_tensor)

    return torch.where(torch.logical_or(torch.gt(rgb_tensor.max(0).values, 1.),
                                        torch.lt(rgb_tensor.min(0).values, 0.)),
                       
                       linear_srgb_from_oklab(torch.stack(
                           [
                               l_clipped_tensor,
                               torch.mul(c_clipped_tensor, units_ab_tensor[0,:,:]),
                               torch.mul(c_clipped_tensor, units_ab_tensor[1,:,:])
                           ]
                       )),

                       rgb_tensor)


def st_cusps_from_lc(lc_cusps_tensor):
    return torch.stack(
        [
            torch.div(lc_cusps_tensor[1,:,:], lc_cusps_tensor[0,:,:]),
            torch.div(lc_cusps_tensor[1,:,:], torch.add(torch.mul(lc_cusps_tensor[0,:,:], -1.), 1))
        ]
    )


def toe(x_tensor):
    k_1 = 0.206
    k_2 = 0.03
    k_3 = (1. + k_1) / (1. + k_2)
    #  0.5f * (k_3 * x - k_1 + sqrtf((k_3 * x - k_1) * (k_3 * x - k_1) + 4 * k_2 * k_3 * x));

    return torch.mul(
        torch.add(
            torch.sub(
                torch.mul(x_tensor, k_3),
                k_1),
            torch.sqrt(
                torch.add(
                    torch.pow(torch.sub(torch.mul(x_tensor, k_3), k_1), 2.),
                    torch.mul(torch.mul(torch.mul(x_tensor, k_3), k_2), 4.)
                )
            )
        ),
        0.5
    )


def toe_inverse(x_tensor):
    k_1 = 0.206
    k_2 = 0.03
    k_3 = (1. + k_1) / (1. + k_2)

    # (x * x + k_1 * x) / (k_3 * (x + k_2))
    return torch.div(
        torch.add(
            torch.pow(x_tensor, 2.),
            torch.mul(x_tensor, k_1)
        ),
        torch.mul(
            torch.add(
                x_tensor,
                k_2
            ),
            k_3
        )        
    )


def srgb_from_okhsv(okhsv_tensor, alpha=0.05, steps=1):
    units_ab_tensor = torch.stack(
        [
            torch.cos(torch.mul(okhsv_tensor[0,:,:], 2.*PI)),
            torch.sin(torch.mul(okhsv_tensor[0,:,:], 2.*PI))
        ]
    )
    lc_cusps_tensor = find_cusp_tensor(units_ab_tensor, steps=steps)
    st_max_tensor = st_cusps_from_lc(lc_cusps_tensor)
    s_0_tensor = torch.tensor(0.5).expand(st_max_tensor.shape[1:])
    k_tensor = torch.add(torch.mul(torch.div(s_0_tensor, st_max_tensor[0,:,:]), -1.), 1)

    # First compute L and V assuming a perfect triangular gamut
    lc_v_base_tensor = torch.add(s_0_tensor, torch.sub(st_max_tensor[1,:,:],
                                                       torch.mul(st_max_tensor[1,:,:],
                                                                 torch.mul(k_tensor,
                                                                           okhsv_tensor[1,:,:]))))
    lc_v_tensor = torch.stack(
        [
            torch.add(torch.div(torch.mul(torch.mul(okhsv_tensor[1,:,:], s_0_tensor), -1.),
                                lc_v_base_tensor),
                      1.),
            torch.div(torch.mul(torch.mul(okhsv_tensor[1,:,:], st_max_tensor[1,:,:]), s_0_tensor),
                      lc_v_base_tensor)
        ]
    )

    lc_tensor = torch.mul(okhsv_tensor[2,:,:], lc_v_tensor)

    l_vt_tensor = toe_inverse(lc_v_tensor[0,:,:])
    c_vt_tensor = torch.mul(lc_v_tensor[1,:,:], torch.div(l_vt_tensor, lc_v_tensor[0,:,:]))

    l_new_tensor = toe_inverse(lc_tensor[0,:,:])
    lc_tensor[1,:,:] = torch.mul(lc_tensor[1,:,:], torch.div(l_new_tensor, lc_tensor[0,:,:]))
    lc_tensor[0,:,:] = l_new_tensor

    rgb_scale_tensor = linear_srgb_from_oklab(
        torch.stack(
            [
                l_vt_tensor,
                torch.mul(units_ab_tensor[0,:,:], c_vt_tensor),
                torch.mul(units_ab_tensor[1,:,:], c_vt_tensor)
            ]
        )
    )

    scale_l_tensor = torch.pow(
        torch.div(1., torch.max(rgb_scale_tensor.max(0).values,
                                torch.zeros(rgb_scale_tensor.shape[1:]))),
        1./3.
    )
    lc_tensor = torch.mul(lc_tensor, scale_l_tensor.expand(lc_tensor.shape))

    rgb_tensor = linear_srgb_from_oklab(
        torch.stack(
            [
                lc_tensor[0,:,:],
                torch.mul(units_ab_tensor[0,:,:], lc_tensor[1,:,:]),
                torch.mul(units_ab_tensor[1,:,:], lc_tensor[1,:,:])
            ]
        )
    )

    return srgb_from_linear_srgb(rgb_tensor, alpha=alpha, steps=steps)


def okhsv_from_srgb(srgb_tensor, steps=1):
    lab_tensor = oklab_from_linear_srgb(linear_srgb_from_srgb(srgb_tensor))

    c_tensor = torch.sqrt(torch.add(torch.pow(lab_tensor[1,:,:], 2.),
                                    torch.pow(lab_tensor[2,:,:], 2.)))
    units_ab_tensor = torch.div(lab_tensor[1:,:,:], c_tensor)

    h_tensor = torch.add(torch.div(torch.mul(torch.atan2(torch.mul(lab_tensor[2,:,:], -1.),
                                                         torch.mul(lab_tensor[1,:,:], -1,)),
                                             0.5),
                                   PI),
                         0.5)
    
    lc_cusps_tensor = find_cusp_tensor(units_ab_tensor, steps=steps)
    st_max_tensor = st_cusps_from_lc(lc_cusps_tensor)
    s_0_tensor = torch.tensor(0.5).expand(st_max_tensor.shape[1:])
    k_tensor = torch.add(torch.mul(torch.div(s_0_tensor, st_max_tensor[0,:,:]), -1.), 1)

    t_tensor = torch.div(st_max_tensor[1,:,:],
                         torch.add(c_tensor, torch.mul(lab_tensor[0,:,:], st_max_tensor[1,:,:])))

    l_v_tensor = torch.mul(t_tensor, lab_tensor[0,:,:])
    c_v_tensor = torch.mul(t_tensor, c_tensor)

    l_vt_tensor = toe_inverse(l_v_tensor)
    c_vt_tensor = torch.mul(c_v_tensor, torch.div(l_vt_tensor, l_v_tensor))

    rgb_scale_tensor = linear_srgb_from_oklab(
        torch.stack(
            [
                l_vt_tensor,
                torch.mul(units_ab_tensor[0,:,:], c_vt_tensor),
                torch.mul(units_ab_tensor[1,:,:], c_vt_tensor)
            ]
        )
    )

    scale_l_tensor = torch.pow(
        torch.div(1., torch.max(rgb_scale_tensor.max(0).values,
                                torch.zeros(rgb_scale_tensor.shape[1:]))),
        1./3.
    )

    lab_tensor[0,:,:] = torch.div(lab_tensor[0,:,:], scale_l_tensor)
    c_tensor = torch.div(c_tensor, scale_l_tensor)

    c_tensor = torch.mul(c_tensor, torch.div(toe(lab_tensor[0,:,:]), lab_tensor[0,:,:]))
    lab_tensor[0,:,:] = toe(lab_tensor[0,:,:])

    v_tensor = torch.div(lab_tensor[0,:,:], l_v_tensor)
    s_tensor = torch.div(torch.mul(torch.add(s_0_tensor, st_max_tensor[1,:,:]), c_v_tensor),
                         torch.add(torch.mul(st_max_tensor[1,:,:], s_0_tensor),
                                   torch.mul(st_max_tensor[1,:,:], torch.mul(k_tensor, c_v_tensor))))

    return torch.stack([h_tensor, s_tensor, v_tensor])
