# Official Implementation of Multimodal Co-attention Transformer
Abstract: Video has emerged as a pervasive medium for communication, entertainment, and information sharing. With the consumption of video content continuing to increase rapidly, understanding the impact of visual narratives on personality has become a crucial area of research. While text-based personality understanding has been extensively studied in the literature, video-based personality prediction remains relatively under-explored. Existing approaches to video-based personality prediction can be broadly categorized into two directions: learning a joint representation of audio and visual information using fully-connected feed-forward networks, and separating a video into its individual modalities (text, image, and audio), training each modality independently, and then ensembling the results for subsequent personality prediction. However, both approaches have notable limitations: ignoring complex interactions between visual and audio components, or considering all three modalities but not in a joint manner. Furthermore, all methods require high computational costs as they require high-resolution images to train. In this paper, we propose a novel Multimodal Co-attention Transformer neural network for video-based affect prediction. Our approach simultaneously models audio, visual, and text representations, as well as their inter-relations, to achieve accurate and efficient predictions. We demonstrate the effectiveness of our method via extensive experiments on a real-world dataset: First Impressions. Our results show that the proposed model outperforms state-of-the-art approaches while maintaining high computational efficiency. In addition to our performance evaluation, we also conduct interpretability analyses to investigate the contribution across different levels. Our findings reveal valuable insights into personality predictions.

## Model Architecture
![3D Co-attention](https://github.com/nestor-sun/mcoattention/assets/47902113/81685bf0-9603-44f1-bac2-66e3b412edf2)


## What's New
- **[Nov 2023]:** Multimodal Co-attention Transformer for Video-Based Personality Understanding has been accepted as a regular paper.
- **[Nov 2023]:** Code will be released soon.

### Dependencies:
- Python 3.10.9
- CUDA 12.2
- PyTorch 2.0.1
