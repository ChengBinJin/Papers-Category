# Papers-Category
It's the repository for collecting papers that I read and category them according to the different objectives.

- [GANS](#gans)
  - [Training Technique](#training-technique)
  - [Structure](#structure)
  - [Style-based Image-to-Image Translation](#style-based-image-to-image-translation)
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
- [Traditional Machine Learning](#traditional-machine-learning)
- [Depth Camera](#depth-camera)

## GANS
#### Training Technique  
- Improved techniques for training GANs, NeurIPS2014 [[Paper](http://papers.nips.cc/paper/6124-improved-techniques-for-training-gans) | [Github](https://github.com/openai/improved-gan)]
#### Structure
- **StyleGANv2**: Analyzing and improving the image quality of StyleGAN, CVPR2020 [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Karras_Analyzing_and_Improving_the_Image_Quality_of_StyleGAN_CVPR_2020_paper.html) | [Github](https://github.com/NVlabs/stylegan2)]  
#### VAE
- Autoencoding beyond pixels using a learned similarity metric, ICML2016, [[Paper](http://proceedings.mlr.press/v48/larsen16.html) | [Github](https://github.com/andersbll/autoencoding_beyond_pixels)]  
#### Unsupervised Image-to-Image Translation  
- **StarGANv1**: StarGAN: unified generative adversarial networks for multi-domain image-to-image translation, CVPR2018 [[Paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Choi_StarGAN_Unified_Generative_CVPR_2018_paper.html) | [Github](https://github.com/yunjey/stargan)]  
#### Style-based Image-to-Image Translation  
- **SPADE**: Semantic image synthesis with spatially adaptive normalization, CVPR2019 [[Paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Park_Semantic_Image_Synthesis_With_Spatially-Adaptive_Normalization_CVPR_2019_paper.html) | [Github](https://github.com/NVlabs/SPADE)]  
- **SEAN**: SEAN: image synthesis with semantic region-adaptive normalization, CVPR2020 [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Zhu_SEAN_Image_Synthesis_With_Semantic_Region-Adaptive_Normalization_CVPR_2020_paper.html) | [Github](https://github.com/ZPdesu/SEAN)]  
- **StarGANv2**: StarGAN v2: diverse image synthesis for multiple domains, CVPR2020 [[Paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Choi_StarGAN_v2_Diverse_Image_Synthesis_for_Multiple_Domains_CVPR_2020_paper.html) | [Github](https://github.com/clovaai/stargan-v2)]  
#### Cartoon  
- **CartoonGAN**: CartoonGAN: generative adversarial networks for photo cartoonization, CVPR2018 [[Paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Chen_CartoonGAN_Generative_Adversarial_CVPR_2018_paper.html) | [Github](https://github.com/FlyingGoblin/CartoonGAN)]
#### Face Attribute Transfer
- **ATTGAN**: AttGAN: facial attribute editing by only changing what you want, TIP2019, [[Paper](https://arxiv.org/pdf/1711.10678.pdf) | [Github](https://github.com/LynnHo/AttGAN-Tensorflow)]
#### Face Change
- FaceShifter towards high fidelity and occlusion aware face swapping, CVPR2020 [[Paper](https://arxiv.org/abs/1912.13457) | [Github](https://github.com/mindslab-ai/faceshifter)]
#### Face Restoration
- Learning warped guidance for blind face restoration, ECCV2018 [[Ppaer](https://openaccess.thecvf.com/content_ECCV_2018/html/Xiaoming_Li_Learning_Warped_Guidance_ECCV_2018_paper.html) | [Github](https://github.com/csxmli2016/GFRNet)]

## Style Transfer
- **AdaIN**: Arbitrary style transfer in real-time with adaptive instance normalization, ICCV2017 [[Paper](https://openaccess.thecvf.com/content_iccv_2017/html/Huang_Arbitrary_Style_Transfer_ICCV_2017_paper.html) | [Github](https://github.com/xunhuang1995/AdaIN-style)]

## Deep Learning
- **PReLU & He initialization**: Delving deep into rectifiers: surpassing human-level performance on ImageNet classification, ICCV2015 [[Paper](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html)]  
- **MobileNetv1**: MobileNets: efficient convolutional neural networks for mobile vision applications, arXiv2017 [[Paper](https://arxiv.org/abs/1704.04861) | [Github](https://github.com/Zehaos/MobileNet)]  
- **MobileNetv2**: MobileNetV2: inverted residuals and linear bottlenecks, CVPR2018 [[Paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.html) | [Github](https://github.com/d-li14/mobilenetv2.pytorch)]  
- **MobileNetv3**: Searching for MobileNetV3, ICCV2019 [[Paper](https://openaccess.thecvf.com/content_ICCV_2019/html/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.html) | [Github](https://github.com/leaderj1001/MobileNetV3-Pytorch)]  

## Image Measurement
- **LPIPS**: The unreasonable effectiveness of deep features as a perceptual metric, CVPR2018 [[Paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_The_Unreasonable_Effectiveness_CVPR_2018_paper.html) | [Github](https://github.com/richzhang/PerceptualSimilarity)]

## Activity Analysis
- Manifold learning for ToF-based human body tracking and activity recognition, BMVC2010 [[Paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.451.3302&rep=rep1&type=pdf)]  
- A unified tree-based framework for joint action localization, recognition and segmentation, CVIU2013 [[Paper](https://www.sciencedirect.com/science/article/abs/pii/S1077314212001749)]  
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

## Traditional Machine Learning
- **SVM**: A tutorial on support vector machines for pattern recognition, DMKD1998 [[Paper](https://link.springer.com/article/10.1023/A:1009715923555)]

## Depth Camera
- Time-of-flight sensors in computer graphics, CGF2010, [[Paper](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-8659.2009.01583.x)]
