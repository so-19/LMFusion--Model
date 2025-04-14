# LMFusion: Implementation Overview

This repository contains our implementation of the LMFusion framework, which extends pretrained language-only LLMs with multimodal generative capabilities. Below is an outline of how we implemented the approach described in the paper.

## Implementation Details

- **Pretrained LLM Extension:**  
  We extend an existing pretrained language-only model by integrating additional components for image generation. The language model modules are kept frozen to preserve their inherent linguistic capabilities.

- **Modality-Specific Components:**  
  - **Attention Modules:** Custom-designed modality-specific attention mechanisms facilitate cross-modal interactions between text and image representations.
  - **Feedforward Layers:** Dedicated feedforward modules process image features, ensuring that the image modality is effectively incorporated into the generative process.

- **Data Handling:**  
  Custom dataloaders were developed to manage image-caption datasets. These tools ensure synchronized processing of both text and image data streams during training while applying necessary data augmentation and preprocessing routines.

- **Dual Objective Training:**  
  To support multimodal joint training, we designed a dual loss framework:
  - **Cross-Entropy Loss:** Used for text generation.
  - **Diffusion-Based Loss:** Employed for high-fidelity image generation, leveraging diffusion models as the generative backbone for images.

- **Modular Training Pipeline:**  
  The training pipeline is structured to:
  - Freeze the language components while training the image-specific modules.
  - Enable efficient cross-modal attention to leverage information exchange between modalities.
  - Provide a flexible architecture that simplifies modifications to individual components such as attention mechanisms, data handlers, or loss functions.

## Paper Reference

For more details on the theoretical framework and experimental evaluations, please refer to the original paper:  
[LMFusion Paper on arXiv](https://arxiv.org/abs/2412.15188)
