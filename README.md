## INTRODUCTION
The "Brain Age Estimation" project aims to predict an individual's brain age based on neuroimaging data using advanced machine learning techniques.
By analyzing brain structure and function, this project seeks to estimate the biological age of the brain, which can differ from chronological age due to various factors like health conditions or cognitive decline. 
Accurately estimating brain age can provide valuable insights into neurological health, helping in early detection of brain disorders as "Alzimer's Disease" and offering potential markers for aging-related research.

## OBJECTIVE
Develop an Accurate Brain Age Estimation Model: Create a robust model using neuroimaging data or biomarkers to accurately predict biological brain age, helping to identify discrepancies from chronological age as early indicators of cognitive decline or neurological conditions.
Validate Model Performance in Neurodegenerative Diseases: Investigate and validate the model's ability to predict brain age in individuals with neurodegenerative diseases, expanding its clinical utility and aiding in early diagnosis and intervention strategies.

## Model Architecture
ResNet-18 (Residual Network-18) is a convolutional neural network (CNN) architecture designed to address the vanishing gradient problem in deep networks using residual connections (skip connections). It is a lightweight variant of the ResNet family and consists of 18 layers, making it suitable for tasks requiring a balance between accuracy and computational efficiency.

Architecture Overview
Total Layers: 18

Number of Convolutional Layers: 17

Number of Fully Connected (FC) Layers: 1

Residual Blocks: 8 (each containing two convolutional layers)

Activation Function: ReLU

Normalization: Batch Normalization

Pooling: Max Pooling and Global Average Pooling

Final Layer: Fully connected layer with Softmax for classification

## Output

![Screenshot 2025-04-04 103104](https://github.com/user-attachments/assets/04c07bee-8fb4-47a2-bb07-46cc072f7310)

![Screenshot 2025-04-04 103053](https://github.com/user-attachments/assets/e414bb33-a3a2-418c-a97f-f2a565078912)

![Screenshot 2025-04-04 103119](https://github.com/user-attachments/assets/a766dab5-fd38-4a6d-92d2-1529eba6f959)
