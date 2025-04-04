import gradio as gr
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import traceback
import numpy as np
from PIL import Image

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Model loading function
def load_model(model_path, num_classes):
    """
    Load a pre-trained ResNet18 model with state dict compatibility

    Args:
    - model_path (str): Path to saved model weights
    - num_classes (int): Number of output classes

    Returns:
    - Loaded and evaluated model
    """
    try:
        # Create the base ResNet18 model
        model = models.resnet18(weights=None)

        # Modify the final fully connected layer
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        # Load the state dictionary
        state_dict = torch.load(model_path, map_location=device)

        # Remove 'model.' prefix and 'num_batches_tracked' keys if present
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            # Remove 'model.' prefix
            if key.startswith('model.'):
                new_key = key.replace('model.', '')
            else:
                new_key = key

            # Remove keys with num_batches_tracked
            if 'num_batches_tracked' not in new_key:
                cleaned_state_dict[new_key] = value

        # Load cleaned state dictionary
        model.load_state_dict(cleaned_state_dict)

        # Set to evaluation mode and move to device
        model.eval()
        print("Model loaded successfully!")
        return model.to(device)

    except Exception as e:
        print(f"Error loading model: {e}")
        print(traceback.format_exc())
        raise

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match ResNet18 input size
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # Standard ImageNet normalization
        std=[0.229, 0.224, 0.225]
    )
])

# Updated Model Path
MODEL_PATH = r"C:\Users\harsh\Desktop\bn (2)\bn\resnet18_model.pth"
CLASSES = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']

# Global model variable to avoid reloading
global_model = None

def predict_alzheimers_category(image):
    """
    Predict Alzheimer's category from an MRI scan with extensive debugging

    Args:
    - image (PIL.Image): Input MRI scan image

    Returns:
    - Predicted category with confidence and debug information
    """
    global global_model

    try:
        # Ensure model is loaded
        if global_model is None:
            global_model = load_model(MODEL_PATH, num_classes=len(CLASSES))

        # Convert to RGB if image is not already
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Debug: Save input image for inspection
        debug_input_path = 'debug_input_image.png'
        image.save(debug_input_path)
        print(f"Saved input image to {debug_input_path}")

        # Preprocess the image
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Debug: Print input tensor details
        print("Input tensor details:")
        print(f"Shape: {input_tensor.shape}")
        print(f"Data type: {input_tensor.dtype}")
        print(f"Min value: {input_tensor.min().item()}")
        print(f"Max value: {input_tensor.max().item()}")

        # Make prediction
        with torch.no_grad():
            outputs = global_model(input_tensor)

            # Debug: Print outputs
            print("Model outputs:")
            print(outputs)

            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()

        # Get the predicted class and probability
        category = CLASSES[predicted_class]
        confidence = probabilities[0][predicted_class].item() * 100

        # Debug: Print full probability distribution
        print("Probability distribution:")
        for cls, prob in zip(CLASSES, probabilities[0]):
            print(f"{cls}: {prob.item() * 100:.2f}%")

        return f"Predicted Category: {category} (Confidence: {confidence:.2f}%)"

    except Exception as e:
        error_details = f"Error in prediction: {e}\n{traceback.format_exc()}"
        print(error_details)
        return error_details

# Create Gradio interface
iface = gr.Interface(
    fn=predict_alzheimers_category,
    inputs=gr.Image(type="pil", label="Upload MRI Scan"),
    outputs=gr.Textbox(label="Alzheimer's Category Assessment"),
    title="Alzheimer's MRI Classification",
    description="Upload an MRI scan to determine the Alzheimer's category",
)

# Launch the interface
if __name__ == "__main__":
    iface.launch()

# Print system and library versions for additional context
import sys

print("\nSystem and Library Versions:")
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
import torchvision
print(f"Torchvision version: {torchvision.__version__}")
