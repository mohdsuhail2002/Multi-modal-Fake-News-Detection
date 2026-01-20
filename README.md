# Multi-modal-Fake-News-Detection
Overview

This project presents a human-centric multimodal fake news detection system that analyses both textual content and associated images to classify news as real or fake. Traditional fake news detection approaches rely mainly on text, which often fails when misleading or manipulated images are used to distort context. This work addresses that limitation by fusing semantic text representations with visual features to enable more reliable misinformation detection.

The project was developed as part of the M.Tech (AI & ML) Minor Project at Jamia Millia Islamia, New Delhi.

Motivation

Fake news spreads rapidly on social media and often combines sensational headlines with deceptive visuals to appear credible. Text-only models are insufficient in such scenarios. By jointly analysing text and images, this project aims to capture cross-modal inconsistencies and improve classification performance in real-world misinformation settings.

Datasets

Two benchmark datasets were used:

GossipCop: Entertainment and celebrity news dataset containing 22,155 articles (5,336 fake and 16,819 real).

PolitiFact: Political news dataset with expert-verified labels, containing 1,272 articles (474 fake and 798 real).

Both datasets include textual content and linked images, making them suitable for multimodal learning.

Methodology

Text data is cleaned and transformed into dense semantic embeddings using a Sentence Transformer (BERT-based) model. Images are resized and processed using a ResNet50 CNN to extract high-level visual features. The text embeddings (384-dimensional) and image embeddings (2048-dimensional) are concatenated to form a unified multimodal feature vector.

The fused features are normalized, passed through activation layers, and finally classified using a Logistic Regression model with Sigmoid activation to predict whether a news item is real or fake.

Results

Experimental results show that the multimodal approach consistently outperforms text-only and image-only models. The fused representation improves detection of misleading image–text pairs, reduces false classifications, and provides more stable performance across both datasets. The findings confirm that multimodal learning is more effective for fake news detection than unimodal approaches.

Technologies Used

Python, Sentence Transformers, ResNet50, Scikit-learn, NumPy, Pandas, Matplotlib, PyTorch/TensorFlow.

Conclusion

This project demonstrates that combining textual and visual information significantly enhances fake news detection performance. Multimodal fusion enables better understanding of deceptive content where images and text are intentionally misaligned. The work highlights the necessity of multimodal deep learning systems for building reliable and scalable misinformation detection tools.
