# Image Denoising Demonstrator

This context defines the domain language for a course project that demonstrates image denoising methods and compares their behavior on noisy images.

## Language

**Demonstrator**:
A local interactive application used to upload, corrupt, denoise, and compare images during presentation.
_Avoid_: website, script, notebook

**Live Computation**:
Processing that runs after the presenter uploads an image or changes parameters, using the current image rather than precomputed examples.
_Avoid_: static gallery, prepared results

**Course Demonstration**:
A presentation-focused workflow that makes algorithm behavior visible through interactive comparison.
_Avoid_: batch benchmark, offline experiment manager

**Source Image**:
The image uploaded by the presenter as the clean reference or as an already noisy input.
_Avoid_: sample image, bundled example

**Noise Scenario**:
A selected noise condition applied during **Live Computation**, or an uploaded image treated as already noisy.
_Avoid_: dataset, preset result

**Frequency Filter**:
A two-dimensional radial frequency response applied to an image spectrum during denoising.
_Avoid_: audio filter, one-dimensional filter

**Band-Stop Filter**:
A frequency-domain filter that suppresses selected frequency bands or localized periodic-noise components in the image spectrum.
_Avoid_: notch filter, trap filter

**Reference Metric**:
A quantitative image-quality score computed only when a clean reference image is available.
_Avoid_: universal score, real-photo score

**No-Reference Review**:
A qualitative comparison used when the input image is already noisy and no clean reference exists.
_Avoid_: fake PSNR, fake SSIM

**Demo Preset**:
An interactive shortcut that fills parameters and runs comparisons immediately for presentation.
_Avoid_: precomputed result, static example

## Relationships

- A **Demonstrator** supports **Live Computation** for a **Course Demonstration**.
- A **Course Demonstration** compares multiple denoising methods against the same **Source Image**.
- A **Noise Scenario** either corrupts a clean **Source Image** or accepts the **Source Image** as already noisy.
- A **Frequency Filter** includes low-pass responses such as Gaussian, Butterworth, Chebyshev I, Chebyshev II, and elliptic forms.
- A **Band-Stop Filter** is an optional **Frequency Filter** mode for demonstrating frequency-localized noise removal.
- A **Reference Metric** compares a denoised image against a clean **Source Image**.
- A **No-Reference Review** uses visual evidence instead of **Reference Metric** values.
- A **Demo Preset** triggers **Live Computation** rather than loading prepared outputs.

## Example Dialogue

> **Dev:** "Should the **Demonstrator** prioritize batch processing or live comparison?"
> **Domain expert:** "Live comparison, because the presentation needs visible parameter changes and side-by-side results from **Live Computation**."

## Flagged Ambiguities

- "Chebyshev" and "elliptic" were used in an image-denoising context; resolved as two-dimensional radial **Frequency Filter** responses rather than one-dimensional audio filters.
- "notch filter" was replaced by **Band-Stop Filter** as the canonical term.
