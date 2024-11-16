Here's a cleaned-up and formatted version of your analysis for a README file.

---

# Model Performance and Analysis

## Overview

Our group presents a comparison of different models trained on a classification task. The performance metrics include training time, training loss, training accuracy, testing accuracy, and the number of model parameters. We also discuss the impact of data augmentation, fine-tuning, and analyze model performance.

## Results Summary

| Model            | Training Time | Training Loss | Training Accuracy | Testing Accuracy | Number of Model Parameters |
|------------------|---------------|---------------|-------------------|------------------|----------------------------|
| VGG 1            | 1.45 minutes  | 12.5          | 88.13%            | 85%              | 25,691,137                 |
| VGG 3            | 1.47 minutes  | 0.058         | 93.75%            | 95%              | 12,938,561                 |
| VGG 3-Augmented  | 1.82 minutes  | 0.23          | 91.25%            | 92.5%            | 103,132,289                |
| VGG 16 (Unfrozen) | 7.39 minutes  | 0.701         | 55%               | 62.5%            | 136,358,721                |
| VGG 16 (Frozen)  | 1.44 minutes  | 0             | 100%              | 100%             | 121,644,033                |
| MLP Model        | 1.44 minutes  | 0.17          | 87.5%             | 92.5%            | 38,638,273                 |

## Analysis

### 1. **Are the Results as Expected?**

Yes, the results met expectations for most models, except for VGG 16 with trainable (unfrozen) parameters. While the other models were able to capture the features effectively, the VGG 16 (unfrozen) model performed worse. This is likely because the model has a large number of parameters, especially those of sensitive filters, making it prone to overfitting without sufficient fine-tuning and regularization.

### 2. **Does Data Augmentation Help?**

Yes, data augmentation helps in reducing overfitting. This can be observed with the `VGG 3-Augmented` model, which has a higher training loss compared to the simple `VGG 3` model but shows a more balanced performance with a comparable testing accuracy. The increased training loss indicates that the model was trained over more diverse samples, which helps it generalize better to unseen data.

### 3. **Does the Number of Fine-Tuning Epochs Matter?**

Yes, the number of epochs for fine-tuning is crucial. Training for too few epochs may result in underfitting, where the model does not capture enough features. Conversely, training for too many epochs can lead to overfitting, especially for models with a large number of parameters like `VGG 16`. We can see that earlier stages see less training and testing accuracy, while later stages seem to have a significant imcrease in both.

### 4. **Are There Any Particular Images That the Model is Confused About?**

Yes, there are certain images that the model tend to remain confused about, such as those having only facial structure from the front.
