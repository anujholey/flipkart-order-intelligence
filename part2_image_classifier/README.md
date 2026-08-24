
# Part 2 - Product Image Categoriser

This part implements a Fashion-MNIST product-image classifier using transfer learning with a pretrained ResNet-18 backbone.

## Dataset

Fashion-MNIST is used with the standard official splits.

- Original training split: 60,000 images
- Training subset: 55,000 images
- Validation subset: 5,000 images
- Test split: 10,000 images

The official test split is kept untouched until final evaluation.

## Preprocessing

Fashion-MNIST images are grayscale 28x28 images.

For compatibility with pretrained ResNet-18:

- The grayscale channel is replicated to 3 channels.
- Images are resized to 224x224.
- ImageNet normalization is applied.

## Model

A pretrained ResNet-18 backbone is used.

During the initial feature-extraction stage, the pretrained backbone is frozen and used to extract 512-dimensional image features.

A new 10-class classifier head is trained using:

- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 256 for feature extraction
- Head batch size: 512
- Epochs: 12

## Saved Model

The trained model is saved as:

`models/product_classifier.pt`

Model metadata is saved as:

`models/product_classifier_metadata.json`

## Single Image Prediction

The function:

`classify_product_image(image_path)`

is implemented in:

`part2_image_classifier/predict_product.py`

It loads the saved model and returns:

- predicted product category
- prediction confidence

## Sample Images

Real Fashion-MNIST test images are exported as PNG files under:

`data/sample_images/`

These are used later by the Part 3 support agent.
