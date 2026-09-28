# Assignment 2: Self-Supervised Rotation Prediction and Transfer Learning

This folder contains the two experiment notebooks and the writeup:

- `ssl_pretrain.ipynb` it pretrains ResNet-18 on STL10 unlabeled images by predicting rotations of 0°, 90°, 180°, and 270°.
- `intel_transfer.ipynb` it transfers that backbone to the Intel Image Classification dataset and compares it with a from-scratch baseline and ImageNet initialization.
- `experiments.pdf` is the experiment report. 

## Conceptual questions

### Question 1: Rotation dataset construction and chance performance

- Total rotation examples: 4 * 100,000 = 400,000.
- Examples in each rotation class:  it will be {0: 100,000, 1: 100,000, 2: 100,000, 3: 100,000}.
- Random-chance accuracy: one correct class out of four, so 1/4 = 25\%.
- Balanced labels matter because if one rotaion occupies the majority in the training set, then the training loss would be pulled toward the majority label. Also, a model could achieve deceptively high accuracy simply by predicting the most common rotation.


### Question 2: Cross-entropy loss and model confidence

- -ln(0.8) = 0.2231
- -ln(0.25) = 1.386. So the loss would be higher. A smaller probability on the correct class would make the loss higher.
- Cross-entropy penalizes the model for assigning smaller probability to the correct class.

### Question 3: Transfer learning improvement

- SSL-FT improves on S0 by (81.0% - 76.5% = 4.5%).
- SSL-FT is below I-FT by (89.5% - 81.0% = 8.5%).
- It improves by 4.5%. So a gain of 4.5 percentage points meets that criterion.
- If it is percent improvement, we would use (81.0% - 76.5%) / 76.5% = 5.88%. The percentage point improvement is clearer because it directly describe the difference between those two accuracy numbers.

### Question 4: Frozen features versus full fine-tuning

- The weight matrix is 512 * 6. And there are 6 bias terms. So in total there are 512 * 6 + 6 = 3078 trainable parameters.
- In the SSL-Frozen, the CNN backbone is frozen, so only the 3,078 parameters in the classifier are trained. In SSL-FT, both the CNN backbone and the classifier are trained. So there are more trainable parameters in SSL-FT.
- This would suggest that the SSL backbone learned useful information, but the representation is not directly well aligned with the Intel classification task.
- Because using the same backbone makes the experiment a fair comparison. If different architectures were used, differences in accuracy could come from differences in the networks themselves.