# Papers-Category
It's the repository for collecting papers that I read and category them according to the different objectives.

- [GANS](#gans)
  - [Training Technique](#training-technique)
  - [Structure](#structure)
  - [Compression](#compression)
  - [VAE](#vae)
  - [Layer Swapping](#layer-swapping)
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
- [Style Transfer](#style-transfer)
- [Deep Learning](#deep-learning)
- [Image Measurement](#image-measurement)
- [Activity Analysis](#activity-analysis)
- [Attribute Prediction](#attribute-prediction)
- [Fingerprint](#fingerprint)
- [Iris](#iris)
- [Detection](#detection)
- [Adversarial Attack](#adversarial-attack)
- [Traditional Machine Learning](#traditional-machine-learning)
- [Image Processing](#image-processing)
- [Depth Camera](#depth-camera)
- [Stereo](#stereo)

## GANS
#### Training Technique  
- Improved techniques for training GANs, NeurIPS2014 [[Paper](http://papers.nips.cc/paper/6124-improved-techniques-for-training-gans) | [Github](https://github.com/openai/improved-gan)]
#### Structure  
- **PGGAN**: Progressive growing of GANs for improved quality, stability, and variation, ICLR2018 [[Paper](https://arxiv.org/abs/1710.10196) | [Github](https://github.com/tkarras/progressive_growing_of_gans)]
- **StyleGANv2**: Analyzing and improving the image quality of StyleGAN, CVPR2020 [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Karras_Analyzing_and_Improving_the_Image_Quality_of_StyleGAN_CVPR_2020_paper.html) | [Github](https://github.com/NVlabs/stylegan2)]  
#### Compression
- GAN compression: efficient architectures for interactive conditional GANs, CVPR2020 [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Li_GAN_Compression_Efficient_Architectures_for_Interactive_Conditional_GANs_CVPR_2020_paper.html) | [Github](https://github.com/mit-han-lab/gan-compression)]
#### VAE
- Autoencoding beyond pixels using a learned similarity metric, ICML2016, [[Paper](http://proceedings.mlr.press/v48/larsen16.html) | [Github](https://github.com/andersbll/autoencoding_beyond_pixels)]  
#### Layer Swapping  
- Resolution dependent GAN interpolation for controllable image synthesis between domains, arXiv2020 [[Paper](https://arxiv.org/abs/2010.05334) | [Github](https://github.com/justinpinkney/stylegan2)]  
- Unsupervised image-to-image translation via pre-trained StyleGAN2 Network, arXiv2020 [[Paper](Unsupervised Image-to-Image Translation via Pre-trained StyleGAN2 Network) | [Github](https://github.com/HideUnderBush/UI2I_via_StyleGAN2)]  
#### Supervised Image-to-Image Translation  
- **pix2pixHD**: High-resolution image synthesis and semantic manipulation with conditional GANs, CVPR2018 [[Paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Wang_High-Resolution_Image_Synthesis_CVPR_2018_paper.html) | [Github](https://github.com/NVIDIA/pix2pixHD)]
- **pSp**: Encoding in StyleGAN: a StyleGAN encoder for image-to-image translation, ICLR2021 [[Paper](https://arxiv.org/abs/2008.00951) | [Github](https://github.com/eladrich/pixel2style2pixel)]
#### Unsupervised Image-to-Image Translation  
- **StarGANv1**: StarGAN: unified generative adversarial networks for multi-domain image-to-image translation, CVPR2018 [[Paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Choi_StarGAN_Unified_Generative_CVPR_2018_paper.html) | [Github](https://github.com/yunjey/stargan)]  
- **U-GAT-IT**: U-GAT-IT: unsupervised generative attentional networks with adaptive layer-instance normalization for image-to-image translation, ICLR2020 [[Paper](https://arxiv.org/abs/1907.10830) | [Github](https://github.com/znxlwm/UGATIT-pytorch)]  
- Breaking the cycle - colleagues are all you need, CVPR2020 [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Nizan_Breaking_the_Cycle_-_Colleagues_Are_All_You_Need_CVPR_2020_paper.html) | [Github](https://github.com/Onr/Council-GAN)]
#### Diverse Image-to-Image Translation  
- **SPADE**: Semantic image synthesis with spatially adaptive normalization, CVPR2019 [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Park_Semantic_Image_Synthesis_With_Spatially-Adaptive_Normalization_CVPR_2019_paper.html) | [Github](https://github.com/NVlabs/SPADE)]  
- **SEAN**: SEAN: image synthesis with semantic region-adaptive normalization, CVPR2020 [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhu_SEAN_Image_Synthesis_With_Semantic_Region-Adaptive_Normalization_CVPR_2020_paper.html) | [Github](https://github.com/ZPdesu/SEAN)]  
- **StarGANv2**: StarGAN v2: diverse image synthesis for multiple domains, CVPR2020 [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Choi_StarGAN_v2_Diverse_Image_Synthesis_for_Multiple_Domains_CVPR_2020_paper.html) | [Github](https://github.com/clovaai/stargan-v2)]    
- **pSp**: Encoding in StyleGAN: a StyleGAN encoder for image-to-image translation, ICLR2021 [[Paper](https://arxiv.org/abs/2008.00951) | [Github](https://github.com/eladrich/pixel2style2pixel)]
#### Supervised Interpretable GAN Control  
- StyleGAN2 distillation of feed-forward image manipulation, arXvi2020 [[Paper](https://arxiv.org/abs/2003.03581) | [Github](https://github.com/EvgenyKashin/stylegan2-distillation)]  
- **StyleFlow**: StyleFlow: attribute-conditioned exploration of StyleGAN-generated image using conditional continuous normalizing flows, arXiv2020 [[Paper](https://ui.adsabs.harvard.edu/abs/2020arXiv200802401A/abstract)]   
- Config: controllable neural face image generation, ECCV2020 [[Paper](https://arxiv.org/abs/2005.02671) | [Github](https://github.com/microsoft/ConfigNet)]
- **InterFaceGAN**: Interpreting the latent space of GANs for semantic face editing, CVPR2020 [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Shen_Interpreting_the_Latent_Space_of_GANs_for_Semantic_Face_Editing_CVPR_2020_paper.html) | [Github](https://github.com/genforce/interfacegan)]  
- **InterFaceGAN++**: InterFaceGAN: interpreting the disentangled face representation learend by GANs, arXiv2020 [[Paper](https://arxiv.org/abs/2005.09635) | [Github](https://github.com/genforce/interfacegan)]
#### Unsupervised Interpretable GAN Control  
- **SeFa**: Closed-form factorization of latent semantics in GANs, arXiv2020 [[Paper](https://arxiv.org/abs/2007.06600) | [Github](https://github.com/genforce/sefa)]
- **GANSpace**: GANSpace: discovering interpretable GAN controls, arXiv2020 [[Paper](https://arxiv.org/abs/2004.02546) | [Github](https://github.com/harskish/ganspace)]  
- Unsupervised discovery of interpretable directions in the GAN latent space, ICML2020 [[Paper](https://arxiv.org/abs/2002.03754) | [Github](https://github.com/anvoynov/GANLatentDiscovery)]
- The hessian penalty: a weak prior for unsupervised disentanglement, ECCV2020 [[Paper](https://arxiv.org/abs/2008.10599) | [Github](https://github.com/wpeebles/hessian_penalty)]
#### GAN Inversion  
- **Image2StyleGAN**: Image2StyleGAN: how to embed images into the StyleGAN latent space? ICCV2019 [[Paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Abdal_Image2StyleGAN_How_to_Embed_Images_Into_the_StyleGAN_Latent_Space_ICCV_2019_paper.html)]
- **IdInvert**: In-domain GAN inversion for real image editing, ECCV2020 [[Paper](https://arxiv.org/abs/2004.00049) | [Github](https://github.com/genforce/idinvert_pytorch)]
- **Image2StyleGAN++**: Image2StyleGAN++: how to edit the embedded images?, CVPR2020 [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Abdal_Image2StyleGAN_How_to_Edit_the_Embedded_Images_CVPR_2020_paper.html)]
- Image processing using multi-code GAN prior, CVPR2020 [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Gu_Image_Processing_Using_Multi-Code_GAN_Prior_CVPR_2020_paper.html) | [Github](https://github.com/genforce/mganprior)]  
#### Anime  
- Crypoko white paper, 2019 [[Paper](https://crypko.ai/static/files/crypko-whitepaper.pdf)]
#### Cartoon  
- **CartoonGAN**: CartoonGAN: generative adversarial networks for photo cartoonization, CVPR2018 [[Paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Chen_CartoonGAN_Generative_Adversarial_CVPR_2018_paper.html) | [Github](https://github.com/FlyingGoblin/CartoonGAN)]  
- **Whie-box**: Learning to cartoonize using white-box cartoon representations supplementary materials, CVPR2020 [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Learning_to_Cartoonize_Using_White-Box_Cartoon_Representations_CVPR_2020_paper.html) | [Github](https://github.com/SystemErrorWang/White-box-Cartoonization)]
#### Face Attribute Transfer
- **ATTGAN**: AttGAN: facial attribute editing by only changing what you want, TIP2019, [[Paper](https://arxiv.org/pdf/1711.10678.pdf) | [Github](https://github.com/LynnHo/AttGAN-Tensorflow)]  
- **STGAN**: STGAN: a unified selective transfer network for arbitrary image attribute editing, CVPR2019 [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Liu_STGAN_A_Unified_Selective_Transfer_Network_for_Arbitrary_Image_Attribute_CVPR_2019_paper.html) | [Github](https://github.com/csmliu/STGAN)]  
- **PA-GAN**: PA-GAN: progressive attention generative adversarial network for facial attribute editing, arXiv2020 [[Paper](https://arxiv.org/abs/2007.05892) | [Github](https://github.com/LynnHo/PA-GAN-Tensorflow)]
#### Face Change
- FaceShifter towards high fidelity and occlusion aware face swapping, CVPR2020 [[Paper](https://arxiv.org/abs/1912.13457) | [Github](https://github.com/mindslab-ai/faceshifter)]
#### Face Restoration
- Learning warped guidance for blind face restoration, ECCV2018 [[Ppaer](https://openaccess.thecvf.com/content_ECCV_2018/html/Xiaoming_Li_Learning_Warped_Guidance_ECCV_2018_paper.html) | [Github](https://github.com/csxmli2016/GFRNet)]

## Style Transfer
- Image style transfer using convolutional neural networks, CVPR2016 [[Paper](https://openaccess.thecvf.com/content_cvpr_2016/html/Gatys_Image_Style_Transfer_CVPR_2016_paper.html) | [Github](https://github.com/cysmith/neural-style-tf)]
- **AdaIN**: Arbitrary style transfer in real-time with adaptive instance normalization, ICCV2017 [[Paper](https://openaccess.thecvf.com/content_iccv_2017/html/Huang_Arbitrary_Style_Transfer_ICCV_2017_paper.html) | [Github](https://github.com/xunhuang1995/AdaIN-style)]

## Deep Learning
- **AlexNet**: ImageNet classification with deep convolutional neural networks, NeurIPS2011 [[Paper](https://dl.acm.org/doi/abs/10.1145/3065386)]
- **PReLU & He initialization**: Delving deep into rectifiers: surpassing human-level performance on ImageNet classification, ICCV2015 [[Paper](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html)]  
- **InstanceNorm**: Instance normalization: the missing ingredient for fast stylization, arXiv2016 [[Paper](https://arxiv.org/abs/1607.08022)]
- **MobileNetv1**: MobileNets: efficient convolutional neural networks for mobile vision applications, arXiv2017 [[Paper](https://arxiv.org/abs/1704.04861) | [Github](https://github.com/Zehaos/MobileNet)]  
- **MobileNetv2**: MobileNetV2: inverted residuals and linear bottlenecks, CVPR2018 [[Paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.html) | [Github](https://github.com/d-li14/mobilenetv2.pytorch)]  
- **MobileNetv3**: Searching for MobileNetV3, ICCV2019 [[Paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.html) | [Github](https://github.com/leaderj1001/MobileNetV3-Pytorch)]  

## Image Measurement
- **LPIPS**: The unreasonable effectiveness of deep features as a perceptual metric, CVPR2018 [[Paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_The_Unreasonable_Effectiveness_CVPR_2018_paper.html) | [Github](https://github.com/richzhang/PerceptualSimilarity)]

## Activity Analysis
- The recognition of human movement using temporal templates, PAMI2001 [[Paper](https://ieeexplore.ieee.org/abstract/document/910878)]
- Manifold learning for ToF-based human body tracking and activity recognition, BMVC2010 [[Paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.451.3302&rep=rep1&type=pdf)]  
- A discriminative key pose sequence model for recognizing human interactions, ICCV2011 [[Paper](https://ieeexplore.ieee.org/abstract/document/6130458)]
- A large-scale benchmark dataset for event recognition in surveillance video, CVPR2011 [[Paper](https://ieeexplore.ieee.org/abstract/document/5995586)]  
- Discriminative latent models for recognizing contextural group activities, PAMI2012 [[Paper](https://ieeexplore.ieee.org/abstract/document/6095563)]
- A unified tree-based framework for joint action localization, recognition and segmentation, CVIU2013 [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S1077314212001749)]  
- 3D convolutional neural networks for human action recognition, PAMI2013 [[Paper](https://ieeexplore.ieee.org/abstract/document/6165309)]  
- Real-time human pose recognition in parts from a single depth image, CACM2013 [[Paper](https://dl.acm.org/doi/abs/10.1145/2398356.2398381)]  
- Efficient human pose estimation from single depth images, PAMI2013 [[Paper](https://ieeexplore.ieee.org/abstract/document/6341759)]
- 시각장애인 보조를 위한 영상기반 휴먼 행동 인식 시스템, 한국정보과학회논문지2015 [[Paper](http://kiise.or.kr/e_journal/2015/1/JOK/pdf/17.pdf)]

## Attribute Prediction
- Deep learning face attributes in the wild, ICCV2015 [[Paper](https://openaccess.thecvf.com/content_iccv_2015/html/Liu_Deep_Learning_Face_ICCV_2015_paper.html)]

##  Fingerprint
- Segmentation of fingerprint images using the directional image, PR1987 [[Paper](https://www.sciencedirect.com/science/article/abs/pii/0031320387900690)]
- Segmentation of fingerprint images - a composite method, PR1989, [[Paper](https://www.sciencedirect.com/science/article/abs/pii/0031320389900472)] 
- Assessing the difficulty level of fingerprint datasets based on relative quality measures, ICHBB2011 [[Paper](https://ieeexplore.ieee.org/abstract/document/6094295)]  
- Type-independent pixel-level alignment point detection for fingerprints, ICHB2011, [[Paper](https://ieeexplore.ieee.org/abstract/document/6094351)]  
- Assessing the level of difficulty of fingerprint datasets based on relative quality measures, IS2014, [[Paper](https://www.sciencedirect.com/science/article/pii/S0020025513004003)]  

## Iris
- Robust iris segmentation via simple circular and linear filter, JEI2008, [[Paper](https://www.spiedigitallibrary.org/journals/Journal-of-Electronic-Imaging/volume-17/issue-4/043027/Robust-iris-segmentation-via-simple-circular-and-linear-filters/10.1117/1.3050067.short?SSO=1)]  
- A new texture analysis appraoch for iris recognition, CCSP2014, [[Paper](https://www.sciencedirect.com/science/article/pii/S2212671614001024)]  

## Detection
- **HOG**: Histograms of oriented gradients for human detection, CVPR2005 [[Paper](https://ieeexplore.ieee.org/abstract/document/1467360)]  
- Human detection using oriented histograms of oriented gradients, ECCV2006 [[Paper](https://link.springer.com/chapter/10.1007/11744047_33)]  
- Hybrid cascade boosting machine using variant scale blocks based HOG feature for pedestrain detection, Neurocomputing2014 [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S0925231214000277)]  
- 다양한 사람 방향을 고려한 파트 영역 기반 사람 영역 검출, JKIISE2013 [[Paper](http://www.dbpia.co.kr/Journal/articleDetail?nodeId=NODE02217486)]  
- **R-CNN**: Rich feature hierarchies for accurate object detection and semantic segmentation, CVPR2014 [[Paper](https://openaccess.thecvf.com/content_cvpr_2014/html/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.html)]  
- Feature pyramid networks for object detection, CVPR2017 [[Paper](https://openaccess.thecvf.com/content_cvpr_2017/html/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.html) | [Github](https://github.com/facebookresearch/detectron)]

## Adversarial Attack  
- Intriguing properties of neural networks, arXiv2013 [[Paper](https://arxiv.org/abs/1312.6199)]

## Traditional Machine Learning
- **SVM**: A tutorial on support vector machines for pattern recognition, DMKD1998 [[Paper](https://link.springer.com/article/10.1023/A:1009715923555)]

## Image Processing
- A threshold selection method from gray-level histograms, SMC1979 [[Paper](https://cw.fel.cvut.cz/wiki/_media/courses/a6m33bio/otsu.pdf)]  
- Dynamic histogram warping of image pairs for constant image brightness, ICIP1995 [[Paper](https://ieeexplore.ieee.org/abstract/document/537491)]  

## Depth Camera
- Time-of-flight sensors in computer graphics, CGF2010, [[Paper](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-8659.2009.01583.x)]  

## Stereo
- Displets-resolving stereo ambiguities using object knowledge, CVPR2015, [[Paper](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Guney_Displets_Resolving_Stereo_2015_CVPR_paper.html) | [Github](https://github.com/edz-o/displet)]  
