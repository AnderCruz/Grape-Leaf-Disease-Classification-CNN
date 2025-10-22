# Grape Leaf Disease Classification with CNN

## Project Overview

This project develops a **Convolutional Neural Network (CNN)** model to automatically classify vine leaf images as healthy or diseased. The system was created to help **Grape Valley Winery** improve grape quality by enabling early detection of leaf diseases, reducing agricultural losses, and promoting sustainable vineyard monitoring practices.

### Problem Statement

Grape Valley Winery has been experiencing quality variations in their grape harvest due to diseases and anomalies in vine leaves. The current manual inspection process by field workers is:
- **Time-consuming**
- **Subjective** 
- **Prone to human error**

This AI-powered solution provides an automated, efficient, and accurate approach to identify diseased leaves from field-captured images.

## Dataset

The dataset contains **1,600 JPG images** of grape leaves categorized into four classes:

- **Leaf Blight** 🍂
- **Black Measles** 🔴  
- **Healthy Grapes** ✅
- **Black Rot** ⚫

### Dataset Structure
```
grapes/
├── LeafBlight/
├── BlackMeasles/
├── HealthyGrapes/
└── BlackRot/
```

## Technical Implementation

### Environment Setup

The project uses **TensorFlow** with optimized GPU configuration:

```python
import tensorflow as tf
from tensorflow.keras import mixed_precision

def setup_tensorflow(gpu_growth=True, mixed_precision_enabled=True):
    """
    Configures TensorFlow for optimal GPU memory usage and mixed precision.
    """
    tf.keras.backend.clear_session()
    
    if mixed_precision_enabled:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
    
    if gpu_growth:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
```

### Key Features

- **Mixed Precision Training**: Uses float16 for improved performance
- **Dynamic GPU Memory Growth**: Efficient GPU resource utilization
- **Image Data Processing**: Leverages `pathlib` for dataset management
- **Data Visualization**: Tools for exploring and understanding the dataset

## Installation & Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd grape-leaf-disease-classification
```

2. **Install dependencies**
```bash
pip install tensorflow pillow pathlib
```

3. **Prepare dataset**
   - Organize images in the `grapes/` directory with subfolders for each class
   - Ensure images are in JPG format

## Model Architecture

The project implements a **Convolutional Neural Network** with:

- **Input**: Grape leaf images
- **Architecture**: Custom CNN designed for image classification
- **Output**: Multi-class classification (4 disease categories + healthy)
- **Optimization**: Mixed precision training for speed and efficiency

## Usage

1. **Data Preparation**
```python
data_dir = pathlib.Path('grapes')
leafblight = list(data_dir.glob('LeafBlight/*'))
```

2. **Model Training**
   - Configure TensorFlow settings
   - Load and preprocess dataset
   - Train CNN model
   - Evaluate performance

3. **Prediction**
   - Use trained model to classify new leaf images
   - Get predictions for disease detection

## Results

The model provides:
- **Automated disease detection**
- **Early warning system** for vineyard management
- **Reduced manual inspection time**
- **Improved accuracy** in leaf health assessment

## Impact

This solution helps Grape Valley Winery:
- **Improve grape quality** through early disease detection
- **Reduce agricultural losses** by timely intervention
- **Optimize production processes** with automated monitoring
- **Promote sustainable practices** through precision agriculture

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for suggestions and improvements.

## 📄 License

This project is developed for educational and agricultural improvement purposes.


**Tags**: `CNN` `TensorFlow` `Agriculture` `Disease-Classification` `Computer-Vision` `Deep-Learning` `Precision-Farming`
