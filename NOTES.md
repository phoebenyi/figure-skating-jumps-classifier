# Personal Learning Notes:

# Deep Neural Networks (DNN) vs Convolutional Neural Networks (CNN)

Deep Neural Networks (DNN) and Convolutional Neural Networks (CNN) are both types of neural networks used in deep learning, but they have different architectures and are suited for different types of tasks.

Initially, I decided to build a DNN machine learning model for classifying the 6 different types of figure skating jumps because DNN are oftenly used for classification. However, I got a low test accuracy of 10-20% because DNNs lack the ability to capture the spatial and temporal structure inherent in figure skating jumps, as they treat all input features independently.

Therefore, I decided to build a CNN machine learning model because CNNs work better with handling spatial and sequential data, as they capture local patterns and relationships that DNNs cannot effectively manage without an enormous amount of data. The CNN’s architecture—especially its convolutional and pooling layers—allows it to pick up essential spatial features, making it particularly well-suited for visual and temporal data, like movements in figure skating jumps. In the end, I got a test accuracy of 86.96%.

## Deep Neural Networks (DNN)

### Architecture:
A DNN consists of multiple layers of nodes (neurons), including input, hidden, and output layers. Each neuron in a layer is connected to every neuron in the next layer, making it a fully connected network.

### Use Cases:
DNNs are versatile and can be used for a variety of tasks, including classification, regression, and function approximation. They are often employed for structured data (like tabular data) and simpler tasks.

### Characteristics:
May require more training data to generalize well.
Can easily overfit if not properly regularized.
Training can be computationally expensive, especially with many layers.


## Convolutional Neural Networks (CNN)

### Architecture:
A CNN is designed specifically for processing structured grid data, such as images. It includes convolutional layers that apply filters (kernels) to the input data, pooling layers to reduce dimensionality, and often fully connected layers at the end.

### Use Cases:
CNNs are primarily used in image recognition, object detection, and other tasks involving spatial data. They excel at capturing spatial hierarchies and patterns.

### Characteristics:
More efficient than DNNs for image data due to the local connections and weight sharing.
Require less training data compared to DNNs for similar tasks, thanks to their ability to generalize well from features.
Typically incorporate techniques like dropout and data augmentation to combat overfitting.

## Summary
DNNs are general-purpose networks that can handle various types of data but may struggle with high-dimensional data like images.
CNNs are specialized for image and spatial data processing, leveraging their architecture to extract spatial features efficiently.
In conclusion, the choice between a DNN and a CNN depends on the specific application and the nature of the data you are working with. For image-related tasks, CNNs are generally preferred, while DNNs are suitable for more general tasks.

# Filters, Kernels, Layers

## Filters/Kernels:
Definition: A filter, or kernel, is a small matrix (e.g., 3x3, 5x5) that slides, or "convolves," across the input data to detect patterns.
Purpose: Filters identify specific features within the data, such as edges, textures, or shapes in images. Each filter is trained to detect a unique feature, which is then used in later layers to understand more complex structures.
How It Works: When a filter passes over an area of the image, it performs element-wise multiplication with the pixel values it covers, summing the result into a single value. This process produces a new matrix (feature map) showing where the filter "activates," or detects the feature.

## Convolutional Layers:
Definition: These apply filters to the input data, generating feature maps that highlight specific aspects of the data.
Purpose: Capture and process local patterns in the data, such as edges or textures in images.
Output: Feature maps representing detected features across the image.

## Pooling Layers:
Definition: These reduce the spatial dimensions (height and width) of feature maps, usually with max pooling or average pooling.
Purpose: Downsample feature maps to make the model less sensitive to slight variations and reduce computational requirements.
Example: A max-pooling layer with a 2x2 filter and a stride of 2 will pick the maximum value in each 2x2 area, halving the dimensions.

## Fully Connected Layers:
Definition: Located toward the end of a CNN, these layers connect every neuron to the next layer's neurons, similar to a DNN.
Purpose: Integrate the spatially extracted features from previous layers for final classification or prediction tasks.

## Activation Layers:
Definition: Apply non-linear transformations (like ReLU) to introduce non-linearity, which helps the model learn complex patterns.

## Summary:
Filters/Kernels: Small matrices that detect features in data.
Convolutional Layers: Apply filters to generate feature maps, capturing local patterns.
Pooling Layers: Reduce the size of feature maps, making the network more efficient and translation-invariant.
Fully Connected Layers: Combine and interpret features for the final output.

# Downsampling:
Reducing the size or resolution of data by decreasing the number of samples or pixels while retaining the essential information, which can help improve computational efficiency and performance in various applications.

# Question: Do we Upsample after Downsampling?
## Yes - Upsample:
Upsampling is essential in tasks that require the output to match the spatial dimensions of the input, such as image segmentation, where each pixel needs to be classified.

In architectures like U-Net, which is designed for semantic segmentation, downsampling layers reduce the spatial resolution to capture high-level features, while upsampling layers restore the original dimensions to produce pixel-wise predictions.

This process enables the model to leverage both global context and local details, ensuring that the output accurately reflects the input image's structure.

## No - Do Not Upsample:
Upsampling may not be necessary for tasks focused solely on classification, such as image recognition, where the goal is to assign a label to the entire image rather than generate a spatially detailed output.

In these cases, the model can effectively extract high-level features through downsampling layers without needing to reconstruct the original input dimensions.

The final output, typically a classification score or a set of bounding boxes, can be derived directly from the downsampled feature maps, making upsampling redundant and potentially adding unnecessary complexity to the model.


# Stride:
Definition: Number of pixels by which the filter (or kernel) moves across the input image during the convolution operation. A stride of 2 means that the filter moves two pixels at a time both horizontally and vertically. (but usually in one direction)

## Example: Effects of Stride of 2:
### Downsampling:
Using a stride of 2 effectively reduces the spatial dimensions (height and width) of the output feature map. This downsampling can help in reducing the computational load and controlling overfitting by providing a lower-resolution representation of the input.
### Feature Extraction:
A larger stride can help capture more abstract features as it emphasizes the most significant features in the image while losing some spatial detail.
###  Output Size Calculation:
When calculating the output size of a convolutional layer with a given stride, the formula can be represented as:
Output Size=⌊(Input Size−Kernel Size+2×Padding)/Stride⌋+1
Input Size: The height or width of the input feature map.
Kernel Size: The height or width of the convolutional filter.
Padding: The number of pixels added to the input image's border.
Stride: The step size of the filter movement (in this case, 2).

## Considerations
### Trade-off:
While using a stride of 2 can speed up training and reduce the feature map size, it may also lead to a loss of spatial information, making it potentially less effective for tasks that require precise localization of features (like object detection).
### Usage:
Stride values of 1 are typically used in earlier layers to capture fine details, while larger strides (like 2 or more) may be more common in deeper layers or when pooling operations are applied to reduce dimensionality.
### Summary:
A stride of 2 is an important hyperparameter in CNNs that affects how the model learns features from the input data while balancing computational efficiency and spatial resolution.
