# LD-ZNet
This is the official repository for the project titled "LD-ZNet: A Latent Diffusion Approach for Text-Based Segmentation". The website for this project is https://koutilya-pnvr.github.io/LD-ZNet/.

## :book: Abstract

We present a technique for segmenting real and AI-generated images using latent diffusion models (LDMs) trained on internet-scale datasets. First, we show that the latent space of LDMs (z-space) is a better input representation compared to other feature representations like RGB images or CLIP encodings for text-based image segmentation. By training the segmentation models on the latent z-space, which creates a compressed representation across several domains like different forms of art, cartoons, illustrations, and photographs, we are also able to bridge the domain gap between real and AI-generated images. We show that the internal features of LDMs contain rich semantic information and present a technique in the form of LD-ZNet to further boost the performance of text-based segmentation. Overall, we show up to 6% improvement over standard baselines for text-to-image segmentation on natural images. For AI-generated imagery, we show close to 20% improvement compared to state-of-the-art techniques.

## :file_folder: AIGI Dataset

The AIGI dataset is made available at https://drive.google.com/drive/u/1/folders/1oZDJu5Y7nqN23Fcb6kCvXy1Do69l_YkQ.
 
## :sparkles: Pretrained Models

The pretrained model for LD-ZNet can be found here: 

## :scroll: Citation
If you find our [paper]([https://arxiv.org/abs/2303.12059](https://openaccess.thecvf.com/content/ICCV2023/papers/PNVR_LD-ZNet_A_Latent_Diffusion_Approach_for_Text-Based_Image_Segmentation_ICCV_2023_paper.pdf)) or this toolbox useful for your research, please cite our work.

```
@InProceedings{PNVR_2023_ICCV,
    author    = {PNVR, Koutilya and Singh, Bharat and Ghosh, Pallabi and Siddiquie, Behjat and Jacobs, David},
    title     = {LD-ZNet: A Latent Diffusion Approach for Text-Based Image Segmentation},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {4157-4168}
}
```
