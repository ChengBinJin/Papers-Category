# Papers-Category
It's the repository for collecting papers that I read and category them according to the different objectives.

- [GANS](#gans)
  - [Training Technique](#training-technique)
  - [Training with Limited Data](#training-with-limited-data)
  - [Structure](#structure)  
  - [Loss](#loss)
  - [Compression](#compression)
  - [VAE](#vae)
  - [Toonify](#toonify)
  - [Supervised Image-to-Image Translation](#supervised-image-to-image-translation)
  - [Unsupervised Image-to-Image Translation](#unsupervised-image-to-image-translation)
  - [Diverse Image-to-Image Translation](#diverse-image-to-image-translation)
  - [Supervised Interpretable GAN Control](#supervised-interpretable-gan-control)
  - [Unsupervised Interpretable GAN Control](#unsupervised-interpretable-gan-control)
  - [GAN Inversion](#gan-inversion)
  - [Anime](#anime)
  - [Cartoon](#cartoon)
  - [Face Attribute Transfer](#face-attribute-transfer)
  - [Face Change](#face-change)  
  - [Face Restoration](#face-restoration)
  - [Inpainting](#Inpainting)
  - [Image Animation](#image-animation)
  - [Fashion](#fashion)
  - [Transformer](#transformer)
- [Style Transfer](#style-transfer)
- [Deep Learning](#deep-learning)
- [Medical](#medical)
- [Image Measurement](#image-measurement)
- [Activity Analysis](#activity-analysis)
- [Attribute Prediction](#attribute-prediction)
- [Surveillance](#surveillance)
- [Visual Planning](#visual-planning)
- [Face](#face)
- [Human Pose](#human-pose)
- [Pose Mesh](#pose-mesh)
- [Hands Pose](#hands-pose)
- [Hands Mesh](#hands-mesh)
- [Fingerprint](#fingerprint)
- [Gaze](#gaze)
- [Iris](#iris)
- [Detection](#detection)
- [Segmentation](#segmentation)
- [Depth Estimation](#depth-estimation)
- [Adversarial Attack](#adversarial-attack)
- [Traditional Machine Learning](#traditional-machine-learning)
- [Image Processing](#image-processing)
- [Depth Camera](#depth-camera)
- [Stereo](#stereo)
- [Robotics](#robotics)

## GANS
#### Training Technique  
- Improved techniques for training GANs, NeurIPS2014 [[Paper](http://papers.nips.cc/paper/6124-improved-techniques-for-training-gans) | [Github](https://github.com/openai/improved-gan)]  
- **InfoGAN**: InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial Nets, NeurIPS2016 [[Paper](https://arxiv.org/abs/1606.03657) | [Github](https://github.com/eriklindernoren/PyTorch-GAN#infogan)]  
#### Training with Limited Data
- **Freeze-D**: Freeze the Discriminator: A Simple Baseline forFine-Tuning GANs, arXiv2020 [[Paper](https://arxiv.org/abs/2002.10964) | [Github](https://github.com/sangwoomo/FreezeD)]  
- **StyleGAN2-ADA**: Training Generative Adversarial Networks with Limited Data, arXiv2020 [[Paper](https://arxiv.org/abs/2006.06676) | [Github](https://github.com/NVlabs/stylegan2-ada)]  
- Towards Faster and Stabilized GAN Training for High-Fidelity Few-Shot Image Synthesis, ICLR2021 [[Paper](https://ui.adsabs.harvard.edu/abs/2021arXiv210104775L/abstract) | [Github](https://github.com/odegeasslbc/FastGAN-pytorch)]  
#### Structure  
- **PGGAN**: Progressive growing of GANs for improved quality, stability, and variation, ICLR2018 [[Paper](https://arxiv.org/abs/1710.10196) | [Github](https://github.com/tkarras/progressive_growing_of_gans)]
- **StyleGANv2**: Analyzing and improving the image quality of StyleGAN, CVPR2020 [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Karras_Analyzing_and_Improving_the_Image_Quality_of_StyleGAN_CVPR_2020_paper.html) | [Github](https://github.com/NVlabs/stylegan2)]  
#### Loss  
- **LSGAN**: Least Squared Generative Adversarial Networks, ICCV2017 [[Paper](https://openaccess.thecvf.com/content_iccv_2017/html/Mao_Least_Squares_Generative_ICCV_2017_paper.html) | [Github](https://github.com/xudonmao/LSGAN)]  
#### Compression
- GAN compression: efficient architectures for interactive conditional GANs, CVPR2020 [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Li_GAN_Compression_Efficient_Architectures_for_Interactive_Conditional_GANs_CVPR_2020_paper.html) | [Github](https://github.com/mit-han-lab/gan-compression)]
#### VAE
- Autoencoding beyond pixels using a learned similarity metric, ICML2016, [[Paper](http://proceedings.mlr.press/v48/larsen16.html) | [Github](https://github.com/andersbll/autoencoding_beyond_pixels)]  
#### Toonify 
- Resolution dependent GAN interpolation for controllable image synthesis between domains, arXiv2020 [[Paper](https://arxiv.org/abs/2010.05334) | [Github](https://github.com/justinpinkney/stylegan2)]  
- Unsupervised image-to-image translation via pre-trained StyleGAN2 Network, arXiv2020 [[Paper](Unsupervised Image-to-Image Translation via Pre-trained StyleGAN2 Network) | [Github](https://github.com/HideUnderBush/UI2I_via_StyleGAN2)]  
- **AgileGAN**: AgileGAN: Stylizing Portraits by Inversion-Consistent Transfer Learning, SIGGRAPH Asia 2021 [[Paper](https://dl.acm.org/doi/abs/10.1145/3450626.3459771?casa_token=105uPjzfGO0AAAAA:NHUC0lz5lu_bCsR1qHIV1NvAUQ5lY9QHmOapWvPJ99ukteduDA25l6aGDPDb1pbzLvVad01Z1St9TNs) | [Code](https://github.com/open-mmlab/MMGEN-FaceStylor)]  
#### Supervised Image-to-Image Translation  
- **pix2pixHD**: High-resolution image synthesis and semantic manipulation with conditional GANs, CVPR2018 [[Paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Wang_High-Resolution_Image_Synthesis_CVPR_2018_paper.html) | [Github](https://github.com/NVIDIA/pix2pixHD)]  
- **pSp**: Encoding in StyleGAN: a StyleGAN encoder for image-to-image translation, ICLR2021 [[Paper](https://arxiv.org/abs/2008.00951) | [Github](https://github.com/eladrich/pixel2style2pixel)]
#### Unsupervised Image-to-Image Translation  
- **StarGANv1**: StarGAN: unified generative adversarial networks for multi-domain image-to-image translation, CVPR2018 [[Paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Choi_StarGAN_Unified_Generative_CVPR_2018_paper.html) | [Github](https://github.com/yunjey/stargan)]  
- The Surprising Effectiveness of Linear Unsupervised Image-to-Image Translation, arXiv2020 [[Paper](https://arxiv.org/abs/2007.12568) | [Github](https://github.com/eitanrich/lin-im2im)]  
- **U-GAT-IT**: U-GAT-IT: unsupervised generative attentional networks with adaptive layer-instance normalization for image-to-image translation, ICLR2020 [[Paper](https://arxiv.org/abs/1907.10830) | [Github](https://github.com/znxlwm/UGATIT-pytorch)]  
- Breaking the cycle - colleagues are all you need, CVPR2020 [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Nizan_Breaking_the_Cycle_-_Colleagues_Are_All_You_Need_CVPR_2020_paper.html) | [Github](https://github.com/Onr/Council-GAN)]
#### Diverse Image-to-Image Translation  
- **BicycleGAN**: Toward Multimodal Image-to-Image Translation, NeurIPS2017 [[Paper](https://arxiv.org/abs/1711.11586) | [Github](https://github.com/junyanz/BicycleGAN)]  
- **SPADE**: Semantic image synthesis with spatially adaptive normalization, CVPR2019 [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Park_Semantic_Image_Synthesis_With_Spatially-Adaptive_Normalization_CVPR_2019_paper.html) | [Github](https://github.com/NVlabs/SPADE)]  
- **SEAN**: SEAN: image synthesis with semantic region-adaptive normalization, CVPR2020 [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhu_SEAN_Image_Synthesis_With_Semantic_Region-Adaptive_Normalization_CVPR_2020_paper.html) | [Github](https://github.com/ZPdesu/SEAN)]  
- **StarGANv2**: StarGAN v2: diverse image synthesis for multiple domains, CVPR2020 [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Choi_StarGAN_v2_Diverse_Image_Synthesis_for_Multiple_Domains_CVPR_2020_paper.html) | [Github](https://github.com/clovaai/stargan-v2)]    
- **pSp**: Encoding in StyleGAN: a StyleGAN encoder for image-to-image translation, ICLR2021 [[Paper](https://arxiv.org/abs/2008.00951) | [Github](https://github.com/eladrich/pixel2style2pixel)]  
- **OASIS**: You Only Need Adversarial Supervision for Semantic Image Synthesis, ICLR2021 [[Paper](https://arxiv.org/abs/2012.04781) | [Github](https://github.com/boschresearch/OASIS)]
#### Supervised Interpretable GAN Control  
- StyleGAN2 distillation of feed-forward image manipulation, arXvi2020 [[Paper](https://arxiv.org/abs/2003.03581) | [Github](https://github.com/EvgenyKashin/stylegan2-distillation)]  
- **StyleFlow**: StyleFlow: attribute-conditioned exploration of StyleGAN-generated image using conditional continuous normalizing flows, arXiv2020 [[Paper](https://ui.adsabs.harvard.edu/abs/2020arXiv200802401A/abstract)]   
- Config: controllable neural face image generation, ECCV2020 [[Paper](https://arxiv.org/abs/2005.02671) | [Github](https://github.com/microsoft/ConfigNet)]
- **InterFaceGAN**: Interpreting the latent space of GANs for semantic face editing, CVPR2020 [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Shen_Interpreting_the_Latent_Space_of_GANs_for_Semantic_Face_Editing_CVPR_2020_paper.html) | [Github](https://github.com/genforce/interfacegan)]  
- **InterFaceGAN++**: InterFaceGAN: interpreting the disentangled face representation learend by GANs, PAMI2020 [[Paper](https://ieeexplore.ieee.org/abstract/document/9241434) | [Github](https://github.com/genforce/interfacegan)]
- **StyleSpace**: StyleSpace Analysis: Disentangled Controls for StyleGAN Image Generation, CVPR2021 [[Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Wu_StyleSpace_Analysis_Disentangled_Controls_for_StyleGAN_Image_Generation_CVPR_2021_paper.html) | [Code](https://github.com/betterze/StyleSpace)]  
#### Unsupervised Interpretable GAN Control  
- Spatially Controllable Image Synthesis with Internal Representation Collaging, arXiv2018 [[Paper](https://arxiv.org/abs/1811.10153) | [Github](https://github.com/quolc/neural-collage)]  
- **GANSpace**: GANSpace: discovering interpretable GAN controls, arXiv2020 [[Paper](https://arxiv.org/abs/2004.02546) | [Github](https://github.com/harskish/ganspace)]  
- Navigating the GAN Parameter Space for Semantic Image Editing, arXiv2020 [[Paper](https://arxiv.org/abs/2011.13786) | [Github](https://github.com/yandex-research/navigan)]  
- Unsupervised discovery of interpretable directions in the GAN latent space, ICML2020 [[Paper](https://arxiv.org/abs/2002.03754) | [Github](https://github.com/anvoynov/GANLatentDiscovery)]  
- Mask-Guided Discovery of Semantic Manifolds in Generative Models, NeursIPS-Worksho2020 [[Paper](https://mengyu.page/files/masked-gan-manifold.pdf) | [Github](https://github.com/bmolab/masked-gan-manifold)]
- The hessian penalty: a weak prior for unsupervised disentanglement, ECCV2020 [[Paper](https://arxiv.org/abs/2008.10599) | [Github](https://github.com/wpeebles/hessian_penalty)]  
- Editing in Style: Uncovering the Local Semantics of GANs, CVPR2020 [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Collins_Editing_in_Style_Uncovering_the_Local_Semantics_of_GANs_CVPR_2020_paper.html) | [Github](https://github.com/IVRL/GANLocalEditing)]
- **SeFa**: Closed-form factorization of latent semantics in GANs, CVPR2021 [[Paper](https://arxiv.org/abs/2007.06600) | [Github](https://github.com/genforce/sefa)]
#### GAN Inversion  
- **Image2StyleGAN**: Image2StyleGAN: how to embed images into the StyleGAN latent space? ICCV2019 [[Paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Abdal_Image2StyleGAN_How_to_Embed_Images_Into_the_StyleGAN_Latent_Space_ICCV_2019_paper.html)]  
- Collaborative Learning for Faster StyleGAN Embedding, arXiv2020 [[Paper](https://arxiv.org/abs/2007.01758)]  
- **IdInvert**: In-domain GAN inversion for real image editing, ECCV2020 [[Paper](https://arxiv.org/abs/2004.00049) | [Github](https://github.com/genforce/idinvert_pytorch)]
- **Image2StyleGAN++**: Image2StyleGAN++: how to edit the embedded images?, CVPR2020 [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Abdal_Image2StyleGAN_How_to_Edit_the_Embedded_Images_CVPR_2020_paper.html)]
- Image processing using multi-code GAN prior, CVPR2020 [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Gu_Image_Processing_Using_Multi-Code_GAN_Prior_CVPR_2020_paper.html) | [Github](https://github.com/genforce/mganprior)]  
#### Anime  
- Illustration2Vec: A Semantic Vector Representation of Illustrations, SIGGRAPH2015 [[Paper](https://dl.acm.org/doi/abs/10.1145/2820903.2820907) | [Github](https://github.com/rezoo/illustration2vec)]  
- Crypoko white paper, 2019 [[Paper](https://crypko.ai/static/files/crypko-whitepaper.pdf)]  
#### Cartoon  
- **CartoonGAN**: CartoonGAN: generative adversarial networks for photo cartoonization, CVPR2018 [[Paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Chen_CartoonGAN_Generative_Adversarial_CVPR_2018_paper.html) | [Github](https://github.com/FlyingGoblin/CartoonGAN)]  
- **Whie-box**: Learning to cartoonize using white-box cartoon representations supplementary materials, CVPR2020 [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Learning_to_Cartoonize_Using_White-Box_Cartoon_Representations_CVPR_2020_paper.html) | [Github](https://github.com/SystemErrorWang/White-box-Cartoonization)]
#### Face Attribute Transfer
- **ATTGAN**: AttGAN: facial attribute editing by only changing what you want, TIP2019, [[Paper](https://arxiv.org/pdf/1711.10678.pdf) | [Github](https://github.com/LynnHo/AttGAN-Tensorflow)]  
- **STGAN**: STGAN: a unified selective transfer network for arbitrary image attribute editing, CVPR2019 [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Liu_STGAN_A_Unified_Selective_Transfer_Network_for_Arbitrary_Image_Attribute_CVPR_2019_paper.html) | [Github](https://github.com/csmliu/STGAN)]  
- **PA-GAN**: PA-GAN: progressive attention generative adversarial network for facial attribute editing, arXiv2020 [[Paper](https://arxiv.org/abs/2007.05892) | [Github](https://github.com/LynnHo/PA-GAN-Tensorflow)]  
- Image-to-Images Translation via Hierarchical Style Disentanglement, CVPR2021 [[Paper](https://arxiv.org/abs/2103.01456) | [Github](https://github.com/imlixinyang/HiSD)]  
#### Face Change
- FaceShifter towards high fidelity and occlusion aware face swapping, CVPR2020 [[Paper](https://arxiv.org/abs/1912.13457) | [Github](https://github.com/mindslab-ai/faceshifter)]
#### Face Restoration
- Learning warped guidance for blind face restoration, ECCV2018 [[Ppaer](https://openaccess.thecvf.com/content_ECCV_2018/html/Xiaoming_Li_Learning_Warped_Guidance_ECCV_2018_paper.html) | [Github](https://github.com/csxmli2016/GFRNet)]  
#### Inpainting  
- Large Scale Image Completion via Co-Modulated Generative Adversarial Networks, ICLR2021 [[Paper](https://openreview.net/forum?id=sSjqmfsk95O)]  
- **PD-GAN**: PD-GAN: Probabilistic Diverse GAN for Image Inpainting, CVPR2021 [[Paper](https://arxiv.org/abs/2105.02201) | [Github](https://github.com/KumapowerLIU/PD-GAN)]  
- DeFLOCNet: Deep Image Editing via Flexible Low-level Control, CVPR2021 [[Paper](https://arxiv.org/abs/2103.12723) | [Github](https://github.com/KumapowerLIU/DeFLOCNet)]  

#### Image Animation
- Generating Videos with Scene Dynamics, NIPS2016 [[Paper](https://proceedings.neurips.cc/paper/2016/hash/04025959b191f8f9de3f924f0940515f-Abstract.html) | [Code](https://github.com/GV1028/videogan)]  
- **X2Face**: X2Face: A Network for Controlling Face Generation using Images, Audio, and Pose Codes, ECCV2018 [[Paper](https://openaccess.thecvf.com/content_ECCV_2018/html/Olivia_Wiles_X2Face_A_network_ECCV_2018_paper.html) | [Code](https://github.com/oawiles/X2Face)]  
- **GANimation**: GANimation: Anatomically-aware Facial Animation from a Single Image, ECCV2018 [[Paper](https://openaccess.thecvf.com/content_ECCV_2018/html/Albert_Pumarola_Anatomically_Coherent_Facial_ECCV_2018_paper.html) | [Github](https://github.com/albertpumarola/GANimation)] 
- **Recycle-GAN**: Recycle-GAN: Unsupervised Video Retargeting, ECCV2018 [[Paper](https://openaccess.thecvf.com/content_ECCV_2018/html/Aayush_Bansal_Recycle-GAN_Unsupervised_Video_ECCV_2018_paper.html) | [Code](https://github.com/aayushbansal/Recycle-GAN)]  
- Synthesizing Images of Humans in Unseen Poses, CVPR2018 [[Paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Balakrishnan_Synthesizing_Images_of_CVPR_2018_paper.html) | [Code](https://github.com/balakg/posewarp-cvpr2018)]
- Every Smile is Unique: Landmark-Guided Diverse Smile Generation, CVPR2018 [[Paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Wang_Every_Smile_Is_CVPR_2018_paper.html)]  
- **MoCoGAN**: MoCoGAN: Decomposing Motion and Content for Video Generation, CVPR2018 [[Paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Tulyakov_MoCoGAN_Decomposing_Motion_CVPR_2018_paper.html) | [Code](https://github.com/sergeytulyakov/mocogan)]  
- **vid2vid**: Video-to-Video Synthesis, NeurIPS2018, [[Paper](https://arxiv.org/abs/1808.06601) | [Github](https://github.com/NVIDIA/vid2vid)]  
- **LWG**: Liquid Warping GAN: A Unified Framework for Human Motion Imitation, Appearance Transfer and Novel View Synthesis, ICCV2019 [[Paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Liu_Liquid_Warping_GAN_A_Unified_Framework_for_Human_Motion_Imitation_ICCV_2019_paper.html) | [Code](https://github.com/svip-lab/impersonator)]
- **MonkeyNet**: Animating Arbitrary Objects via Deep Motion Transfer, CVPR2019 [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Siarohin_Animating_Arbitrary_Objects_via_Deep_Motion_Transfer_CVPR_2019_paper.html) | [Github](https://github.com/AliaksandrSiarohin/monkey-net)]  
- Photo Wake-Up: 3D Character Animation from a Single Photo, CVPR2019 [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Weng_Photo_Wake-Up_3D_Character_Animation_From_a_Single_Photo_CVPR_2019_paper.html)]  
- Textured Neural Avatars, CVPR2019 [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Shysheya_Textured_Neural_Avatars_CVPR_2019_paper.html) | [Code](https://github.com/saic-violet/textured_avatars)]  
- **FOMM**: First Order Motion Model for Image Animation, NeurIPS2019 [[Paper](https://proceedings.neurips.cc/paper/2019/hash/31c0b36aef265d9221af80872ceb62f9-Abstract.html) | [Github](https://github.com/AliaksandrSiarohin/first-order-model)]  
- Motion-supervised Co-part Segmentation, ICPR2021 [[Paper](https://ieeexplore.ieee.org/abstract/document/9412520) | [Code](https://github.com/AliaksandrSiarohin/motion-cosegmentation)]  
- **MRAA**: Motion Representations for Articulated Animation, CVPR2021 [[Paper](https://arxiv.org/abs/2104.11280) | [Github](https://github.com/snap-research/articulated-animation)]  
- Stochastic Image-to-Video Synthesis using cINNs, CVPR2021 [[Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Dorkenwald_Stochastic_Image-to-Video_Synthesis_Using_cINNs_CVPR_2021_paper.html) | [Code](https://github.com/CompVis/image2video-synthesis-using-cINNs)]  
- Few-Shot Human Motion Transfer by Personalized Geometry and Texture Modling, CVPR2021 [[Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Huang_Few-Shot_Human_Motion_Transfer_by_Personalized_Geometry_and_Texture_Modeling_CVPR_2021_paper.html) | [Code](https://github.com/HuangZhiChao95/FewShotMotionTransfer)]  
- Pose-Guided Human Animation from a Single Image in the Wild, CVPR2021 [[Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Yoon_Pose-Guided_Human_Animation_From_a_Single_Image_in_the_Wild_CVPR_2021_paper.html)]  
- **StylePeople**: StylePeople: A Generative Model of Fullbody Human Avatars, CVPR2021 [[Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Grigorev_StylePeople_A_Generative_Model_of_Fullbody_Human_Avatars_CVPR_2021_paper.html) | [Code](https://github.com/Dolorousrtur/style-people)]  


#### Fashion
- Deformable GANs for Pose-based Human Image Generation, CVPR2018 [[Paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Siarohin_Deformable_GANs_for_CVPR_2018_paper.html) | [Github](https://github.com/AliaksandrSiarohin/pose-gan)]
- FiNet: Compatible and Diverse Fashion Image Inpainting, ICCV2019 [[Paper](http://openaccess.thecvf.com/content_ICCV_2019/html/Han_FiNet_Compatible_and_Diverse_Fashion_Image_Inpainting_ICCV_2019_paper.html)]
#### Transformer
- Taming Transformers for High-Resolution Image Synthesis, CVPR2021 [[Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Esser_Taming_Transformers_for_High-Resolution_Image_Synthesis_CVPR_2021_paper.html) | [Code](https://github.com/CompVis/taming-transformers)]  

## Style Transfer
- A Neural Algorithm of Artistic Style, arXiv2014 [[Paper](https://arxiv.org/abs/1508.06576) | [Code](https://github.com/jcjohnson/neural-style)]  
- Image style transfer using convolutional neural networks, CVPR2016 [[Paper](https://openaccess.thecvf.com/content_cvpr_2016/html/Gatys_Image_Style_Transfer_CVPR_2016_paper.html) | [Github](https://github.com/cysmith/neural-style-tf)]
- **AdaIN**: Arbitrary style transfer in real-time with adaptive instance normalization, ICCV2017 [[Paper](https://openaccess.thecvf.com/content_iccv_2017/html/Huang_Arbitrary_Style_Transfer_ICCV_2017_paper.html) | [Github](https://github.com/xunhuang1995/AdaIN-style)]

## Deep Learning
- **AlexNet**: ImageNet classification with deep convolutional neural networks, NeurIPS2011 [[Paper](https://dl.acm.org/doi/abs/10.1145/3065386)]  
- **Adam**: ADAM: A Method for Stochastic Optimization, arXiv2014 [[Paper](https://arxiv.org/abs/1412.6980)]  
- Spatial Transformer Networks, NeurIPS2015, [[Paper](https://arxiv.org/abs/1506.02025) | [Gitub](https://github.com/tensorpack/tensorpack/tree/master/examples/SpatialTransformer)]  
- Learning from Massive Noisy Labeled Data for Image Classification, CVPR2015 [[Paper](https://openaccess.thecvf.com/content_cvpr_2015/html/Xiao_Learning_From_Massive_2015_CVPR_paper.html)]  
- From Generic to Specific Deep Representations for Visual Recognition, CVPR2015 [[Paper](https://www.cv-foundation.org/openaccess/content_cvpr_workshops_2015/W03/html/Azizpour_From_Generic_to_2015_CVPR_paper.html)]  
- **PReLU & He initialization**: Delving deep into rectifiers: surpassing human-level performance on ImageNet classification, ICCV2015 [[Paper](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html)]  
- **InstanceNorm**: Instance normalization: the missing ingredient for fast stylization, arXiv2016 [[Paper](https://arxiv.org/abs/1607.08022)]
- **MobileNetv1**: MobileNets: efficient convolutional neural networks for mobile vision applications, arXiv2017 [[Paper](https://arxiv.org/abs/1704.04861) | [Github](https://github.com/Zehaos/MobileNet)]  
- **ShuffleNet**: ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices, CVPR2018 [[Paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_ShuffleNet_An_Extremely_CVPR_2018_paper.html) | [Code](https://github.com/megvii-model/ShuffleNet-Series)]  
- **MobileNetv2**: MobileNetV2: inverted residuals and linear bottlenecks, CVPR2018 [[Paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.html) | [Github](https://github.com/d-li14/mobilenetv2.pytorch)]  
- **MobileNetv3**: Searching for MobileNetV3, ICCV2019 [[Paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.html) | [Github](https://github.com/leaderj1001/MobileNetV3-Pytorch)]  

## Medical
- Virtual PET Images from CT Data using Deep Convolutional Networks: Initial Results, SASHMI2017 [[Paper](https://link.springer.com/chapter/10.1007/978-3-319-68127-6_6)]  
- Whole Brain Segmentation and Labeling from CT using Synthetic MR Images, MLMI2017 [[Paper](https://link.springer.com/chapter/10.1007/978-3-319-67389-9_34)]  
- nnU-Net: Breaking the Spell on Successful Medial Image Segmentation, arXiv2019 [[Paper](http://rumc-gcorg-p-public.s3.amazonaws.com/evaluation-supplementary/599/351f2fd9-01b3-40e0-802c-2929ba10abd3/nnUnet.pdf)]  
- Generative Adversarial Network in Medical Imaging: A Review, Medical Image Analysis 2019, [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S1361841518308430)]   

## Image Measurement
- **LPIPS**: The unreasonable effectiveness of deep features as a perceptual metric, CVPR2018 [[Paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_The_Unreasonable_Effectiveness_CVPR_2018_paper.html) | [Github](https://github.com/richzhang/PerceptualSimilarity)]

## Activity Analysis
- The recognition of human movement using temporal templates, PAMI2001 [[Paper](https://ieeexplore.ieee.org/abstract/document/910878)]
- Manifold learning for ToF-based human body tracking and activity recognition, BMVC2010 [[Paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.451.3302&rep=rep1&type=pdf)]  
- A discriminative key pose sequence model for recognizing human interactions, ICCV2011 [[Paper](https://ieeexplore.ieee.org/abstract/document/6130458)]
- A large-scale benchmark dataset for event recognition in surveillance video, CVPR2011 [[Paper](https://ieeexplore.ieee.org/abstract/document/5995586)]  
- Detecting Activities of Daily Living in First-Person Camera Views, CVPR2012 [[Paper](https://ieeexplore.ieee.org/abstract/document/6248010)]  
- Discriminative latent models for recognizing contextural group activities, PAMI2012 [[Paper](https://ieeexplore.ieee.org/abstract/document/6095563)]  
- Segmental Multi-Way Local Polling for Video Recognition, ICM2013 [[Paper](https://dl.acm.org/doi/abs/10.1145/2502081.2502167)]  
- A unified tree-based framework for joint action localization, recognition and segmentation, CVIU2013 [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S1077314212001749)]  
- First-Person Activity Recognition: What Are They Doing to Me? CVPR2013 [[Paper](https://www.cv-foundation.org/openaccess/content_cvpr_2013/html/Ryoo_First-Person_Activity_Recognition_2013_CVPR_paper.html)]  
- 3D convolutional neural networks for human action recognition, PAMI2013 [[Paper](https://ieeexplore.ieee.org/abstract/document/6165309)]  
- Real-time human pose recognition in parts from a single depth image, CACM2013 [[Paper](https://dl.acm.org/doi/abs/10.1145/2398356.2398381)]  
- Efficient human pose estimation from single depth images, PAMI2013 [[Paper](https://ieeexplore.ieee.org/abstract/document/6341759)]
- 시각장애인 보조를 위한 영상기반 휴먼 행동 인식 시스템, 한국정보과학회논문지2015 [[Paper](http://kiise.or.kr/e_journal/2015/1/JOK/pdf/17.pdf)]
- Beyond Gaussian Pyramid Multi-skip Feature Stacking for Action Recognition, CVPR2015 [[Paper](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Lan_Beyond_Gaussian_Pyramid_2015_CVPR_paper.html)]  
- Delving into Egocentric Actions, CVPR2015 [[Paper](https://openaccess.thecvf.com/content_cvpr_2015/html/Li_Delving_Into_Egocentric_2015_CVPR_paper.html)]  
- Finding Action Tubes, CVPR2015 [[Paper](https://openaccess.thecvf.com/content_cvpr_2015/html/Gkioxari_Finding_Action_Tubes_2015_CVPR_paper.html) | [Code](https://github.com/gkioxari/ActionTubes#instructions)]  
- Can Humans Fly? Action Understanding with Multiple Classes of Actors, CVPR2015 [[Paper](https://openaccess.thecvf.com/content_cvpr_2015/html/Xu_Can_Humans_Fly_2015_CVPR_paper.html)]  
- Human Action Recognition using Histogram of Motion Intensity and Direction from Multiple Views, IETCV2016 [[Paper](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/iet-cvi.2015.0233)]  
- Two-layer Discriminative Model for Human Activity Recognition, IETCV2016 [[Paper](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/iet-cvi.2015.0235)]  
- A Multi-Stream Bi-Directional Recurrent Neural Network for Fine-Grained Action Detection, CVPR2016 [[Paper](https://openaccess.thecvf.com/content_cvpr_2016/html/Singh_A_Multi-Stream_Bi-Directional_CVPR_2016_paper.html)]  
- Progressively Parsing Interactional Objects for Fine Grained Action Detection, CVPR2016 [[Paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Ni_Progressively_Parsing_Interactional_CVPR_2016_paper.html) | [Code]()]  
- End-to-End Learning of Action Detection from Frame Glimpses in Videos, CVPR2016 [[Code](https://openaccess.thecvf.com/content_cvpr_2016/html/Yeung_End-To-End_Learning_of_CVPR_2016_paper.html)]  

## Attribute Prediction
- Deep learning face attributes in the wild, ICCV2015 [[Paper](https://openaccess.thecvf.com/content_iccv_2015/html/Liu_Deep_Learning_Face_ICCV_2015_paper.html)]

## Surveillance
- A Simplified Nonlinear Regression Method for Human Height Estimation in Video Surveillance, EURASIP Journal on Image and Video Processing 2015 [[Paper](https://jivp-eurasipjournals.springeropen.com/articles/10.1186/s13640-015-0086-1)]  
- 범죄 취약 계층 안전을 위한 CCTV 기반 성별 구분, IPIU2016
- Video Based Child and Adult Classification using Convolutional Neural Network, IPIU2016
- 다수 사람 추적상태에 따른 감시영상 요약 시스템, KUSE Transactions on Computing Practices 2016 [[Paper](https://www.dbpia.co.kr/Journal/articleDetail?nodeId=NODE06599140)]

## Visual Planning
- Self-Supervised Visual Planning with Temporal Skip Connections, CoRL2017 [[Paper](https://proceedings.mlr.press/v78/frederik%20ebert17a/frederik%20ebert17a.pdf)]

## Face
- Are You Really Smiling at Me? Spontaneous versus Posed Enjoyment Smiles, ECCV2012 [[Paper](https://link.springer.com/chapter/10.1007/978-3-642-33712-3_38)]  
- OpenFace: A General-Purpose Face Recognition Library with Mobile Applications, CMU School of Computer Science 2016 [[Paper](http://reports-archive.adm.cs.cmu.edu/anon/anon/usr0/ftp/2016/CMU-CS-16-118.pdf) | [Code](https://github.com/cmusatyalab/openface)]  
- How Far Are We From Solving the 2D & 3D Face Alignment Problem? (And A Dataset of 230,000 3D Facial Landmarks), ICCV2017 [[Paper](https://openaccess.thecvf.com/content_iccv_2017/html/Bulat_How_Far_Are_ICCV_2017_paper.html) | [Code](https://github.com/1adrianb/2D-and-3D-face-alignment)]
- Unsupervised Discovery of Object Landmarks as Structural Representations, CVPR2018 [[Paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_Unsupervised_Discovery_of_CVPR_2018_paper.html) | [Code](https://github.com/YutingZhang/lmdis-rep)]  
- Wing Loss for Robust Facial Landmark Localisation with Convolutional Neural Networks, CVPR2018 [[Paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Feng_Wing_Loss_for_CVPR_2018_paper.html)]  
- Unsupervised Learning of Object Landmarks through Conditional Image Generation, NeurIPS2018 [[Paper](https://arxiv.org/abs/1806.07823) | [Code](https://github.com/tomasjakab/imm)]  
- Laplace Landmark Localization, ICCV2019 [[Paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Robinson_Laplace_Landmark_Localization_ICCV_2019_paper.html)]  
- Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression, ICCV2019 [[Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_Adaptive_Wing_Loss_for_Robust_Face_Alignment_via_Heatmap_Regression_ICCV_2019_paper.pdf) | [Code](https://github.com/protossw512/AdaptiveWingLoss)]  
- **Arcface**: Arcface: Additive Angular Margin Loss for Deep Face Recognition, CVPR2019 [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.html)|[Github](https://github.com/deepinsight/insightface)]  

## HumanPose
- **OpenPose**: Realtim Multi-Person 2D Pose Estimation using Part Affinity Fields, CVPR2017 [[Paper](https://openaccess.thecvf.com/content_cvpr_2017/html/Cao_Realtime_Multi-Person_2D_CVPR_2017_paper.html) | [Code](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation)]  
- Is 2D Heatmap Representation Even Necessary for Human Pose Estimation?, arXiv2021 [[Paper](https://arxiv.org/abs/2107.03332) | [Code](https://github.com/leeyegy/SimDR)]  
- Human Pose Regression with Residual Log-likelihood Estimation, ICCV2021 [[Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Li_Human_Pose_Regression_With_Residual_Log-Likelihood_Estimation_ICCV_2021_paper.html) | [Code](https://github.com/Jeff-sjtu/res-loglikelihood-regression)]  
- **HRNet**: Deep High-Resoluiton Representation Learning for Human Pose Estimation, CVPR2019 [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Sun_Deep_High-Resolution_Representation_Learning_for_Human_Pose_Estimation_CVPR_2019_paper.html) | [Code](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)]  
- **Lite-HRNet**: Lite-HRNet: A Lightweight High-Resolution Network, CVPR2021 [[Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Yu_Lite-HRNet_A_Lightweight_High-Resolution_Network_CVPR_2021_paper.html) | [Code](https://github.com/HRNet/Lite-HRNet)]  

## HumanMesh
- Keep It SMPL: Automatic Estimation of 3D Human Pose and Shape from a Single Image, ECCV2016 [[Paper](https://link.springer.com/chapter/10.1007/978-3-319-46454-1_34) | [Code](https://github.com/Jtoo/fitting_human_smpl_model)]  
- **Pose2Mesh**: Pose2Mesh: Graph Convolutional Network for 3D Human Pose and Mesh Recovery from a 2D Human Pose, ECCV2020 [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-58571-6_45) | [Code](https://github.com/hongsukchoi/Pose2Mesh_RELEASE)]  

## HandsPose
- MediaPipe Hands: On-device Real-time Hand Tracking, arXiv2020 [[Paper](https://arxiv.org/abs/2006.10214) | [Code](https://google.github.io/mediapipe/solutions/hands.html)]  
- InterHand2.6M: A Dataset and Baseline for 3D Interacting Hand Pose Estimation from a Single RGB Image, ECCV2020 [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-58565-5_33) | [Code](https://github.com/facebookresearch/InterHand2.6M)]  
- End-to-End Detection and Pose Estimation of Two Interacting Hands, ICCV2021 [[Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Kim_End-to-End_Detection_and_Pose_Estimation_of_Two_Interacting_Hands_ICCV_2021_paper.html)]  

## HandsMesh
- **MobileHand**: MobileHand: Real-Time 3D Hand Shape and Pose Estimation from Color Image, ICNIP2020 [[Paper](https://link.springer.com/chapter/10.1007/978-3-030-63820-7_52) | [Code](https://github.com/gmntu/mobilehand)]  
- **MEgATrack**: MEgATrack: Monochrome Egocentric Articulated Hand Tracking for Virtual Reality, TOG2020 [[Paper](https://dl.acm.org/doi/abs/10.1145/3386569.3392452) | [Code](https://github.com/milkcat0904/MegaTrack-pytorch)]  
- **Youtube3d**: Weakly-Supervised Mesh-Convolutional Hand Reconstruction in the Wild, CVPR2020 [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Kulon_Weakly-Supervised_Mesh-Convolutional_Hand_Reconstruction_in_the_Wild_CVPR_2020_paper.html) | [Code](https://github.com/arielai/youtube_3d_hands)]  
- Towards Accurate Alignment in Real-time 3D Hand-Mesh Reconstruction, ICCV2021 [[Paper](https://openaccess.thecvf.com/content/ICCV2021/html/Tang_Towards_Accurate_Alignment_in_Real-Time_3D_Hand-Mesh_Reconstruction_ICCV_2021_paper.html) | [Code](https://github.com/wbstx/handAR)]  

##  Fingerprint
- Segmentation of fingerprint images using the directional image, PR1987 [[Paper](https://www.sciencedirect.com/science/article/abs/pii/0031320387900690)]
- Segmentation of fingerprint images - a composite method, PR1989, [[Paper](https://www.sciencedirect.com/science/article/abs/pii/0031320389900472)] 
- Assessing the difficulty level of fingerprint datasets based on relative quality measures, ICHBB2011 [[Paper](https://ieeexplore.ieee.org/abstract/document/6094295)]  
- Type-independent pixel-level alignment point detection for fingerprints, ICHB2011, [[Paper](https://ieeexplore.ieee.org/abstract/document/6094351)]  
- Assessing the level of difficulty of fingerprint datasets based on relative quality measures, IS2014, [[Paper](https://www.sciencedirect.com/science/article/pii/S0020025513004003)]  

## Gaze
- DeepWarp: Photorealistic Image Resynthesis for Gaze Manipulation
Authors, ECCV2016 [[Paper](https://link.springer.com/chapter/10.1007/978-3-319-46475-6_20) | [Code](https://github.com/BlueWinters/DeepWarp)]

## Iris
- Synthesis of Large Realistic Iris Databases using Patch-based Sampling, ICPR2008, [[Paper](https://ieeexplore.ieee.org/abstract/document/4761674)]    
- Iris-GAN: Learning to Generate Realistic Iris Images Using Convolutional GAN, arXiv2018 [[Paper](https://arxiv.org/abs/1812.04822)]  
- Robust iris segmentation via simple circular and linear filter, JEI2008, [[Paper](https://www.spiedigitallibrary.org/journals/Journal-of-Electronic-Imaging/volume-17/issue-4/043027/Robust-iris-segmentation-via-simple-circular-and-linear-filters/10.1117/1.3050067.short?SSO=1)]  
- A new texture analysis appraoch for iris recognition, CCSP2014, [[Paper](https://www.sciencedirect.com/science/article/pii/S2212671614001024)]  
- MinENet: A Dilated CNN for Semantic Segmentation of Eye Features, ICCVW2019 [[Paper](https://openaccess.thecvf.com/content_ICCVW_2019/html/OpenEDS/Perry_MinENet_A_Dilated_CNN_for_Semantic_Segmentation_of_Eye_Features_ICCVW_2019_paper.html)]  

## Detection
- **HOG**: Histograms of oriented gradients for human detection, CVPR2005 [[Paper](https://ieeexplore.ieee.org/abstract/document/1467360)]  
- Human detection using oriented histograms of oriented gradients, ECCV2006 [[Paper](https://link.springer.com/chapter/10.1007/11744047_33)]  
- Hybrid cascade boosting machine using variant scale blocks based HOG feature for pedestrain detection, Neurocomputing2014 [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S0925231214000277)]  
- Recognition using Visual Phrases, CVPR2011 [[Paper](https://ieeexplore.ieee.org/abstract/document/5995711)]  
- 다양한 사람 방향을 고려한 파트 영역 기반 사람 영역 검출, JKIISE2013 [[Paper](http://www.dbpia.co.kr/Journal/articleDetail?nodeId=NODE02217486)]  
- **R-CNN**: Rich feature hierarchies for accurate object detection and semantic segmentation, CVPR2014 [[Paper](https://openaccess.thecvf.com/content_cvpr_2014/html/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.html)]  
- End-to-End People Detection in Crowded Scenes, CVPR2016 [[Paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Stewart_End-To-End_People_Detection_CVPR_2016_paper.html)]  
- Feature pyramid networks for object detection, CVPR2017 [[Paper](https://openaccess.thecvf.com/content_cvpr_2017/html/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.html) | [Github](https://github.com/facebookresearch/detectron)]  
- Bounding Box Regression with Uncertainty for Accurate Object Detection, CVPR2019 [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/html/He_Bounding_Box_Regression_With_Uncertainty_for_Accurate_Object_Detection_CVPR_2019_paper.html) | [Code](https://github.com/yihui-he/KL-Loss)]  

## Segmentation
- **UNet**: U-Net: Convolutional Networks for Biomedical Image Segmentation, MICCAI2015 [[Paper](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28) | [Code](https://github.com/ChengBinJin/U-Net-TensorFlow)]  
- **FCN**: Fully Convolutional Networks for Semantic Segmentation, CVPR2015 [[Paper](https://openaccess.thecvf.com/content_cvpr_2015/html/Long_Fully_Convolutional_Networks_2015_CVPR_paper.html) | [Code](https://github.com/shelhamer/fcn.berkeleyvision.org)]  

## Depth Estimation
- Make3d: Depth Perception from a Single Still Image, AAAI2008 [[Paper](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/viewer.html?pdfurl=https%3A%2F%2Fwww.aaai.org%2FPapers%2FAAAI%2F2008%2FAAAI08-265.pdf&clen=1745360&chunk=true)]  
- Towards Unified Depth and Semantic Prediction from a Single Image, CVPR2015 [[Paper](https://openaccess.thecvf.com/content_cvpr_2015/html/Wang_Towards_Unified_Depth_2015_CVPR_paper.html)]  

## Adversarial Attack  
- Intriguing properties of neural networks, arXiv2013 [[Paper](https://arxiv.org/abs/1312.6199)]

## Traditional Machine Learning
- **SVM**: A tutorial on support vector machines for pattern recognition, DMKD1998 [[Paper](https://link.springer.com/article/10.1023/A:1009715923555)]  
- Singular Value Decomposition and Principal Component Analysis, A Practical Approach to Microarray Data Analysis 2003 [[Paper](https://arxiv.org/ftp/physics/papers/0208/0208101.pdf)]

## Image Processing
- A threshold selection method from gray-level histograms, SMC1979 [[Paper](https://cw.fel.cvut.cz/wiki/_media/courses/a6m33bio/otsu.pdf)]  
- Dynamic histogram warping of image pairs for constant image brightness, ICIP1995 [[Paper](https://ieeexplore.ieee.org/abstract/document/537491)]  
- Just Noticeable Defocus Blur Detection and Estimation, CVPR2015 [[Paper](https://openaccess.thecvf.com/content_cvpr_2015/html/Shi_Just_Noticeable_Defocus_2015_CVPR_paper.html)]  

## Depth Camera
- Time-of-flight sensors in computer graphics, CGF2010, [[Paper](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-8659.2009.01583.x)]  

## Stereo
- A Computer Algorithm for Reconstructing a Scene from Two Projections, Nature1981 [[Paper](https://www.nature.com/articles/293133a0)]
- Stereo Matching using Belief Propagation, PAMI2003 [[Paper](https://ieeexplore.ieee.org/abstract/document/1206509)]  
- Efficient Stereo Matching for Belief Propagation using Plane Convergence, IPIU2010  
- Fast Stereo Matching based on Plane-Converging Belief Propagation using GPU, The Institute of Electronics Engineers of Korea SP 2011, [[Paper](https://www.koreascience.or.kr/article/JAKO201112961963891.page)]  
- Displets-resolving stereo ambiguities using object knowledge, CVPR2015, [[Paper](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Guney_Displets_Resolving_Stereo_2015_CVPR_paper.html) | [Github](https://github.com/edz-o/displet)]  

## Robotics
- Continuous-Curvature Paths for Autonomous Vehicles, ICRA1989 [[Paper](https://ieeexplore.ieee.org/abstract/document/100153)]  
