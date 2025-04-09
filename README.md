# flower-classificaiton-recognization

🌸 Flower Classification with CNN - Project Overview
This project performs image classification on five flower types using a Convolutional Neural Network (CNN) in TensorFlow and Keras. The dataset consists of labeled images of daisy, dandelion, sunflower, tulip, and rose, and the goal is to classify them using deep learning techniques accurately.

📦 1. Modules and Libraries
The project begins by importing all necessary libraries:

General Setup: Warnings are suppressed for cleaner output.

Data Handling: numpy, pandas for numerical and tabular data.

Visualization: matplotlib.pyplot and seaborn for plotting images and training metrics.

Machine Learning Utilities: From sklearn — train-test split, performance metrics, label encoding.

Deep Learning: Keras and TensorFlow modules are used to build and train the CNN model.

Image Processing: cv2, os, PIL, and tqdm are used to read, preprocess, and iterate over image files.

🧹 2. Data Preparation
The dataset is stored in five directories — one for each flower category:

flowers/daisy

flowers/dandelion

flowers/sunflower

flowers/tulip

flowers/rose

A function (make_train_data) is used to:

Read images from each directory.

Resize them to 150×150 pixels.

Append image arrays to list X and their labels to list Z.

🖼️ 3. Image Visualization
A sample of 10 randomly selected flower images is plotted using matplotlib to give a visual sense of the dataset.

🔠 4. Label Encoding & Normalization
The string labels (e.g., "Daisy", "Rose") are encoded as integers using LabelEncoder.

One-hot encoding is then applied to represent the labels for categorical classification.

Image data is normalized by dividing pixel values by 255.

🔀 5. Train-Test Split
The dataset is split into training and validation sets with a 75-25 ratio using train_test_split.

⚙️ 6. Model Definition (CNN)
A Convolutional Neural Network (CNN) is built using Keras’ Sequential API:

Convolutional Layers: 4 Conv2D layers with ReLU activation and max pooling.

Flattening Layer: Converts image matrix to a 1D array.

Dense Layers: A fully connected hidden layer followed by an output layer with softmax activation (5 classes).

📉 7. Learning Rate Scheduling
A ReduceLROnPlateau callback is implemented to reduce the learning rate if the model's validation accuracy plateaus, helping with model convergence.

🔄 8. Data Augmentation
To prevent overfitting and improve generalization, ImageDataGenerator is used to perform real-time image augmentation:

Rotation

Zoom

Width/height shifting

Horizontal flipping

🛠️ 9. Model Compilation
The model is compiled using:

Optimizer: Adam with a learning rate of 0.001

Loss Function: Categorical cross-entropy

Metric: Accuracy

🏋️ 10. Model Training
The model is trained using the fit_generator() method, which applies the data augmentation on the fly during training.

Batch size: 128

Epochs: 50

Validation data: Used to monitor performance during training

📈 11. Training Performance Visualization
Loss and accuracy metrics for both training and validation sets are plotted over each epoch to visualize how well the model is learning.

🔍 12. Model Evaluation on Test Data
After training, the model's predictions on the validation set are analyzed:

Predicted labels are compared with actual labels.

Images that were correctly classified and misclassified are displayed, showing both predicted and actual labels.

✅ 13. Correct & ❌ Incorrect Predictions
8 correctly classified and 8 misclassified images are visualized to qualitatively evaluate model performance.

✅ Conclusion
This project demonstrates a complete image classification pipeline using Convolutional Neural Networks in Keras. It includes data preparation, augmentation, model design, training, evaluation, and visualization. The CNN architecture achieves reasonable accuracy and provides insights into both its strengths and limitations through visual analysis of predictions.

