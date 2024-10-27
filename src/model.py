"""
CNN Model with Residual Blocks

This script defines a Convolutional Neural Network (CNN) architecture utilizing residual blocks to 
enhance feature learning and enable deeper networks. This model is designed to integrate with a 
training script for effective end-to-end training and evaluation.

Key Components:

- Imports: Essential Keras layers, models, and regularizers are included for constructing and 
  regularizing the network.

- Residual Block: 
  The `residual_block` function implements two convolutional layers with batch normalization and 
  ReLU activation, along with a shortcut connection to improve gradient flow.

- Model Architecture:
  - Input Layer: Accepts input data with a specified shape.
  - Initial Convolutional Block: Consists of a convolutional layer, batch normalization, ReLU 
    activation, and max pooling for downsampling.
  - Residual Blocks: Three sequential blocks are included, with filters increasing from 64 to 256, 
    allowing for the capture of complex features.
  - Global Average Pooling: Reduces dimensionality while preserving spatial information.
  - Fully Connected Layer: A dense layer with 256 units, batch normalization, and dropout for 
    regularization.
  - Output Layer: A single unit with sigmoid activation for binary classification tasks.

"""

## Import necessary layers and modules from Keras for building the model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, BatchNormalization, Dense, Input, Add, GlobalAveragePooling2D, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

def residual_block(x, filters, kernel_size=(3, 3)):
    """Residual block with two convolutional layers."""
    shortcut = x  # Save the input tensor

    # First convolutional layer
    x = Conv2D(filters, kernel_size, padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x) # Apply batch normalization to improve training stability
    x = Activation('relu')(x) # Use ReLU activation for non-linearity

    # Second convolutional layer
    x = Conv2D(filters, kernel_size, padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)  # Apply batch normalization again

    # Adjust shortcut if needed
    if shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), padding='same')(shortcut)  # Match the dimensions

    # Add the shortcut to the output
    x = Add()([x, shortcut])
    x = Activation('relu')(x) # Apply ReLU activation to the combined output
    
    return x # Return the output of the residual block

def create_cnn_model(input_shape):
    """Create a CNN model with residual blocks."""
    inputs = Input(shape=input_shape)

    # Initial convolutional block
    x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(0.001))(inputs)  # Initial convolution with 64 filters
    x = BatchNormalization()(x) #Normalize the output
    x = Activation('relu')(x) #apply ReLu activation
    x = MaxPooling2D(pool_size=(2, 2))(x) # Downsample the feature maps

    # Residual block 1
    x = residual_block(x, 64) # Apply the first residual block with 64 filters
    x = MaxPooling2D(pool_size=(2, 2))(x) # Downsample again

    # Residual block 2
    x = residual_block(x, 128) # Apply the second residual block with 128 filters
    x = MaxPooling2D(pool_size=(2, 2))(x) # Downsample again

    # Residual block 3
    x = residual_block(x, 256)  # Apply the third residual block with 256 filters
    x = MaxPooling2D(pool_size=(2, 2))(x)  # Downsample again

    # Add more residual blocks if the dataset is large and training time allows it
    # x = residual_block(x, 512)
    # x = MaxPooling2D(pool_size=(2, 2))(x)

    # Global Average Pooling (reduces parameter count while keeping spatial info)
    x = GlobalAveragePooling2D()(x)

    # Fully connected layer
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x) # Normalize the output
    x = Dropout(0.7)(x) # Apply dropout for regularization to prevent overfitting

    # Output layer for binary classification
    outputs = Dense(1, activation='sigmoid')(x)

    # Define model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model
