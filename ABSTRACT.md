# Project Abstract: Explainable AI-Based Automated Nuclear Cataract Detection & Grading System

## Revised Abstract (Single Paragraph for Upload / Presentation)

This project proposes an Explainable AI (XAI)-based framework for automated nuclear cataract detection and grading from slit-lamp images using LOCS III clinical standards. The system first employs Attention U-Net to segment the lens nucleus, removing irrelevant background noise and reflections. The segmented nucleus is classified into severity grades using GoogLeNet, with performance compared against DenseNet121 and InceptionV3 architectures using standard evaluation metrics. To enhance clinical transparency, Grad-CAM is integrated to visualize regions influencing model predictions. An interactive web interface displays the original slit-lamp image, segmented nucleus, predicted cataract grade, confidence score, and XAI heatmaps in real time, supporting accurate and interpretable clinical decision-making.
