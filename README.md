# Anime Face Generator using GANs

This project utilizes Generative Adversarial Networks (GANs) to generate anime face images. GANs have revolutionized various fields by enabling the creation of realistic synthetic data. The goal of this project is to generate high-quality anime faces that can be used in video games or anime productions.

## Problem Definition

Creating art for video games can be challenging for those who are not particularly artistic. This project aims to automate the creation of anime face images using machine learning, specifically GANs, to alleviate the need for manual artistic creation.

## Solution Specification

The project involves training a GAN on a dataset of anime faces. The GAN consists of two models:
- **Generator**: Creates new images from random noise.
- **Discriminator**: Evaluates the authenticity of images, distinguishing between real and generated images.

## Requirements

- Python 3.10
- TensorFlow
- NumPy
- Matplotlib
- Pillow

## Installation

### Option 1: Create a New Virtual Environment

1. **Clone the repository:**

    ```sh
    git clone <repository-url>
    cd <repository-name>
    ```

2. **Create a virtual environment:**

    ```sh
    python3.10 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required libraries:**

    ```sh
    pip install tensorflow numpy matplotlib pillow
    ```

### Option 2: Use the Pre-Created Virtual Environment

1. **Activate the pre-created virtual environment:**

    ```sh
    source 10ver/bin/activate  # On Windows use `10ver\Scripts\activate`
    ```

## Dataset

The dataset used in this project is the [Anime Face Dataset](https://www.kaggle.com/datasets/splcher/animefacedataset?resource=download). Download the dataset and place the images in a folder named `images` within the project directory.

## Data Preprocessing

Images are resized to 28x28 pixels and normalized to prepare them for the GAN model.

## Running the Code

1. **Check GPU Availability:**

    The project leverages GPU acceleration to speed up the training process. Make sure your setup supports GPU execution.

    ```python
    import tensorflow as tf
    num_gpus = len(tf.config.list_physical_devices('GPU'))
    print("Number of GPUs Available: ", num_gpus)
    ```

2. **Run the Jupyter Notebook:**

    Open the `Code.ipynb` notebook using Jupyter Notebook or JupyterLab and run all cells to execute the training process.

3. **Monitor the Training:**

    Generated images will be saved in the `Generated` folder at the end of each epoch to visualize the progress.

## Creating the Models

- **Generator Model**: Takes random noise as input and generates images.
- **Discriminator Model**: Evaluates images and classifies them as real or fake.

## Training Loop

The training process alternates between training the discriminator and the generator:
1. Generate fake images using the generator.
2. Train the discriminator with both real and fake images.
3. Train the generator to fool the discriminator.

## Visualizing Data

Visualize some sample images from the dataset before and after training the models.

## Testing and Analysis

Plot the learning process and evaluate the generator model by visualizing generated images. Compare the results with an autoencoder to highlight the advantages of GANs in generating high-quality images.

## Recommendations for Future Work

1. Extended Training
2. Hyperparameter Tuning
3. Balanced Training Steps
4. Color Image Generation

## Conclusion

GANs demonstrate great potential in generating high-quality images, including anime faces. Despite training challenges, the results are promising for creative applications in game development and art generation.

## References

- Dataset: [Anime Face Dataset](https://www.kaggle.com/datasets/splcher/animefacedataset?resource=download)
