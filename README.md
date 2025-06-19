# FloodNet Semantic Segmentation

### Project Summary:

This project creates a machine learning model to operate a semantic segmentation on FloodNet images, using TensorFlow. The number of segmentation classes is **10**.

---

### Dataset Description:

This dataset consists of `1843` images for training and `500` images for final evaluation/testing. in **12 classes** (including background). Below are sample images from some of the classes present in the dataset:

TODO - UPDATE: ![](./visuals/aerial_drone_segmentation_dataset_image.jpg?raw=true)

---

### Machine Learning Model Architecture:

* Combines VGG16 encoder with a U-Net decoder

* Supports multi-class segmentation via softmax

* Uses skip connections to preserve spatial details

* Trained in two phases: feature extraction, then fine-tuning

* Designed for input shape (224, 224, 3) and configurable number of classes

---

### Data Augmentation:

For the training dataset, I applied the following data augmentation to avoid overfitting:

```
self.transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.CLAHE(clip_limit=4.0, p=0.5),
    A.CoarseDropout(max_holes=8, max_height=16, max_width=16, min_holes=1, fill_value=0, p=0.3),
])
```
---

### Training Hyperparameters:

* Epochs: `35` for feature extraction stage and `10` for fine-tuning stage
  
* `EarlyStopping` with patience of `10` for feature extraction stage and `4` for fine-tuning stage
  
* Optimizer: `Adam`

* Learning Rate: `1e-4` for feature extraction stage and `1e-6` for fine-tuning stage

* Batch size: `8`

---

### Inference for Validation:

TODO - UPDATE: ![](./visuals/inference_for_validation_aerial_drone1.png?raw=true)
TODO - UPDATE: ![](./visuals/inference_for_validation_aerial_drone2.png?raw=true)
TODO - UPDATE: ![](./visuals/inference_for_validation_aerial_drone3.png?raw=true)

---

### Inference for Test:

TODO - UPDATE: ![](./visuals/inference_for_test_aerial_drone1.png?raw=true)
TODO - UPDATE: ![](./visuals/inference_for_test_aerial_drone2.png?raw=true)
TODO - UPDATE: ![](./visuals/inference_for_test_aerial_drone3.png?raw=true)

---

### Accuracy on Test Dataset for Kaggle Submission

The configurations discussed above, yielded a score of **0.85274** on the Kaggle's Leaderboard.

TODO - UPDATE: ![](./visuals/aerial_drone_kaggle_title.png?raw=true)
TODO - UPDATE: ![](./visuals/aerial_drone_segment_kaggle.png?raw=true)
