### InvokeAI Nodes

Note: This repo houses the development branch of these nodes, and potentially buggy, experimental, or otherwise unreleased code. To obtain the latest release versions of these nodes which *have* been released, please visit the following repos:

- https://github.com/dwringer/composition-nodes
- https://github.com/dwringer/depth-from-obj-node
- https://github.com/dwringer/generative-grammar-prompt-nodes
- https://github.com/dwringer/size-stepper-nodes

--------------------------------
### Depth Map from Wavefront OBJ

**Description:** Render depth maps from Wavefront .obj files (triangulated) using this simple 3D renderer utilizing numpy and matplotlib to compute and color the scene. There are simple parameters to change the FOV, camera position, and model orientation.

To be imported, an .obj must use triangulated meshes, so make sure to enable that option if exporting from a 3D modeling program. This renderer makes each triangle a solid color based on its average depth, so it will cause anomalies if your .obj has large triangles. In Blender, the Remesh modifier can be helpful to subdivide a mesh into small pieces that work well given these limitations.

**Node Link:** https://github.com/dwringer/depth-from-obj-node

**Example Usage:**
![depth from obj usage graph](https://raw.githubusercontent.com/dwringer/depth-from-obj-node/main/depth_from_obj_usage.jpg)

--------------------------------
### Generative Grammar-Based Prompt Nodes

**Description:** This set of 3 nodes generates prompts from simple user-defined grammar rules (loaded from custom files - examples provided below). The prompts are made by recursively expanding a special template string, replacing nonterminal "parts-of-speech" until no more nonterminal terms remain in the string.

This includes 3 Nodes:
- *Lookup Table from File* - loads a YAML file "prompt" section (or of a whole folder of YAML's) into a JSON-ified dictionary (Lookups output)
- *Lookups Entry from Prompt* - places a single entry in a new Lookups output under the specified heading
- *Prompt from Lookup Table* - uses a Collection of Lookups as grammar rules from which to randomly generate prompts.

**Node Link:** https://github.com/dwringer/generative-grammar-prompt-nodes

**Example Usage:**
![lookups usage example graph](https://raw.githubusercontent.com/dwringer/generative-grammar-prompt-nodes/main/lookuptables_usage.jpg)

--------------------------------
### Image and Mask Composition Pack

![composition pack main image](https://raw.githubusercontent.com/dwringer/composition-nodes/main/composition_pack.jpg)

**Description:** This is a pack of nodes for composing masks and images, including a simple text mask creator and both image and latent offset nodes. The offsets wrap around, so these can be used in conjunction with the Seamless node to progressively generate centered on different parts of the seamless tiling.

This includes 15 Nodes:
- *Adjust Image Hue Plus* - Rotate the hue of an image in one of several different color spaces.
- *Blend Latents (Masked)* - Use a mask to blend part of one latents tensor into another. Can be used to "renoise" sections during a multi-stage [masked] denoising process.
- *Enhance Image* - Boost or reduce color saturation, contrast, brightness, sharpness, or invert colors of any image at any stage with this simple wrapper for pillow [PIL]'s ImageEnhance module.
- *Equivalent Achromatic Lightness* - Calculates image lightness accounting for Helmholtz-Kohlrausch effect based on a method described by High, Green, and Nussbaum (2023).
- *Text to Mask (Clipseg)* - Input a prompt and an image to generate a mask representing areas of the image matched by the prompt.
- *Text to Mask Advanced (Clipseg)* - Output up to four prompt masks combined with logical "and", logical "or", or as separate channels of an RGBA image.
- *Image Compositor* - Take a subject from an image with a flat backdrop and layer it on another image using a chroma key or flood select background removal.
- *Image Layer Blend* - Perform a layered blend of two images using alpha compositing. Opacity of top layer is selectable.
- *Image Dilate or Erode* - Dilate or expand a mask (or any image!). This is equivalent to an expand/contract operation.
- *Image Value Thresholds* - Clip an image to pure black/white beyond specified thresholds.
- *Offset Latents* - Offset a latents tensor in the vertical and/or horizontal dimensions, wrapping it around.
- *Offset Image* - Offset an image in the vertical and/or horizontal dimensions, wrapping it around.
- *Rotate Image* - Rotate an image in degrees about its center, optionally resizing the image boundaries to fit.
- *Shadows/Highlights/Midtones* - Extract three masks (with adjustable hard or soft thresholds) representing shadows, midtones, and highlights regions of an image.
- *Text Mask (simple 2D)* - create and position a white on black (or black on white) line of text using any font locally available to Invoke.

**Node Link:** https://github.com/dwringer/composition-nodes

**Example Usage:**
![composition nodes usage graph](https://raw.githubusercontent.com/dwringer/composition-nodes/main/composition_nodes_usage.jpg)

--------------------------------
### Size Stepper Nodes

**Description:** This is a set of nodes for calculating the necessary size increments for doing upscaling workflows. Use the *Final Size & Orientation* node to enter your full size dimensions and orientation (portrait/landscape/random), then plug that and your initial generation dimensions into the *Ideal Size Stepper* and get 1, 2, or 3 intermediate pairs of dimensions for upscaling. Note this does not output the initial size or full size dimensions: the 1, 2, or 3 outputs of this node are only the intermediate sizes.

A third node is included, *Random Switch (Integers)*, which is just a generic version of Final Size with no orientation selection.

**Node Link:** https://github.com/dwringer/size-stepper-nodes

**Example Usage:**
![size stepper usage graph](https://raw.githubusercontent.com/dwringer/size-stepper-nodes/main/size_nodes_usage.jpg)

