import os
import torch
import torchvision.models as models
import torch.nn as nn
import traceback


def recover_model(model_path):
    """
    Attempt to recover and diagnose the saved model file
    """
    print("Model Recovery and Diagnostic Script")
    print("=" * 50)

    # Verify file exists and is not empty
    if not os.path.exists(model_path):
        print(f"Error: File {model_path} does not exist!")
        return None

    file_size = os.path.getsize(model_path)
    print(f"Model file size: {file_size} bytes")

    if file_size == 0:
        print("Error: Model file is empty!")
        return None

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    try:
        # Try loading with different methods
        print("\n1. Attempting to load with default method:")
        try:
            state_dict = torch.load(model_path, map_location=device)
            print("Default load successful!")
        except Exception as e:
            print(f"Default load failed: {e}")
            traceback.print_exc()
            return None

        # Create base model
        print("\n2. Attempting to create base model:")
        try:
            # Create base ResNet18 model
            model = models.resnet18(weights=None)

            # Modify the final layer (4 classes for Alzheimer's categories)
            num_classes = 4
            model.fc = nn.Linear(model.fc.in_features, num_classes)

            print("\n3. Attempting to load state dict:")
            try:
                model.load_state_dict(state_dict)
                print("Model loaded successfully!")
                return model
            except Exception as load_error:
                print(f"Failed to load state dictionary: {load_error}")

                # Print out keys for debugging
                print("\nLoaded state dict keys:")
                for key in state_dict.keys():
                    print(key)

                print("\nModel's expected state dict keys:")
                for key in model.state_dict().keys():
                    print(key)

                return None

        except Exception as model_error:
            print(f"Error creating model: {model_error}")
            return None

    except Exception as final_error:
        print(f"Unexpected error: {final_error}")
        traceback.print_exc()
        return None


# Path to your model file
MODEL_PATH = r"C:\Users\acer\OneDrive\Desktop\bn\trained_resnet18_model.pth"

# Run the recovery script
recovered_model = recover_model(MODEL_PATH)

# Additional diagnostics
if recovered_model:
    print("\nModel Recovery Successful!")
    print("Model is ready for inference.")
else:
    print("\nModel Recovery Failed.")
    print("Possible solutions:")
    print("1. Re-train the model")
    print("2. Check the original model saving process")
    print("3. Verify the integrity of the saved model file")