# NEURAL-STYLE-TRANSFER

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: MOHAMMED IBRAHIM SH

*INTERN ID*: CT04DG2681

*DOMAIN*: ARTIFICIAL INTELLIGENCE

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTOSH

**DESCRIPTION:**

This Python script performs neural style transfer, a deep learning technique that fuses the visual content of one image with the artistic style of another. The result is a new image that resembles the original photo but painted in the artistic style of the chosen artwork. The implementation leverages PyTorch, torchvision, and a pre-trained VGG19 neural network.

Libraries and Modules

The script begins by importing several essential libraries:

torch, torch.nn, torch.optim — for building and optimizing deep learning models.

torchvision — for image transformations and pre-trained models.

PIL (Python Imaging Library) — for handling image files.

matplotlib.pyplot — for displaying images.

These libraries collectively enable reading images, transforming them, performing neural computations, and visualizing the results.

Image Loading and Preprocessing

The function load_image handles the reading and preprocessing of images. It performs the following steps:

Opens an image file and converts it to RGB format.

Resizes the image so that its largest dimension is at most 400 pixels, preserving details without excessive memory use.

Transforms the image into a tensor suitable for PyTorch models.

Normalizes pixel values using the mean and standard deviation from ImageNet, matching the training conditions of VGG19.

This function is used to load both the content image (e.g. “your_photo.jpg”) and the style image (e.g. “starry_night.jpg”).

Loading the Pre-trained VGG19 Model

The code uses the feature extraction layers of a pre-trained VGG19 network, a deep convolutional neural network known for its excellent performance in capturing visual features. Importantly:

Gradients for the VGG19 model’s parameters are disabled because the model itself isn’t being trained—only used for feature extraction.

The model and images are moved to a computation device (GPU if available) for efficient processing.

Feature Extraction

The function get_features passes an image through VGG19 and collects feature maps from selected layers. These intermediate outputs capture:

Low-level features like edges and textures in earlier layers.

High-level abstract features in deeper layers.

The collected features are stored in a dictionary for easy retrieval during the loss calculations.

Gram Matrix Calculation

The function gram_matrix computes the Gram matrix of a feature map, which captures how different filters correlate with each other. This representation:

Encodes the texture and style of an image.

Is crucial for measuring style similarity between the target and the style image.

Style Transfer Process

A clone of the content image, called target, serves as the starting point for optimization. The process aims to gradually modify this target image so that:

Its content features resemble those of the original content image.

Its style features (as measured by Gram matrices) resemble those of the style image.

Two losses drive this optimization:

Content Loss

Compares the feature maps of the target and content images in layer conv4_2.

Encourages the target image to retain the structural details of the content image.

Style Loss

Compares Gram matrices of feature maps between the target and style images across several layers.

Encourages the target to adopt textures, colors, and brushstroke-like qualities from the style image.

Uses layer-specific weights to control the influence of each layer on the style transfer.

The total loss is computed as a weighted sum of content and style losses.

Optimization

An Adam optimizer iteratively adjusts the pixel values of the target image to minimize the total loss over 300 iterations. The script prints progress every 50 steps, showing how the total loss decreases as the target image evolves.

Visualization

After optimization, the function im_convert denormalizes the tensor back into a valid RGB image. Finally, Matplotlib displays the resulting stylized image.

Conclusion

This code beautifully demonstrates how deep learning can blend art and technology, transforming ordinary photos into images that mimic the style of renowned paintings—an example of AI’s creative capabilities in modern computer vision.

**INPUT:**
