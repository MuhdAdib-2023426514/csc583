import torch 
from torchvision import transforms
from train_model import CustomCNN
from PIL import Image
import matplotlib.pyplot as plt  # Import matplotlib for visualization

IMG_SIZE = 150
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the checkpoint
checkpoint = torch.load('/home/muhammad_adib/imgclass_cnn/intel_cnn_model.pth', map_location=device)

# Re-initialize the model architecture (must match the saved model)
model = CustomCNN(num_classes=len(checkpoint['class_names'])).to(device)

# Load the saved weights
model.load_state_dict(checkpoint['model_state_dict'])

# Set to evaluation mode
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),  # Same size as training!
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

img = Image.open("653.jpg").convert('RGB')
img_tensor = preprocess(img)
img_tensor = img_tensor.unsqueeze(0)
img_tensor = img_tensor.to(device)
# 5. Predict!
with torch.no_grad():
    output = model(img_tensor)

# 6. Get the predicted class index
probabilities = torch.nn.functional.softmax(output[0], dim=0)
percentiles = probabilities.cpu().numpy() * 100  # Convert probabilities to percentiles

# Visualize confidence levels as a bar chart
class_names = checkpoint['class_names']  # Get class names from the checkpoint
plt.bar(class_names, percentiles)  # Use percentiles for plotting

# Add confidence level values on top of the bars
for i, value in enumerate(percentiles):
    plt.text(i, value + 1, f"{value:.2f}%", ha='center', fontsize=8)  # Adjust position and font size

plt.xlabel('Classes')
plt.ylabel('Confidence Level (%)')  # Update label to show percentage
plt.title('Confidence Levels for Each Class')
plt.xticks(rotation=45, ha='right')  # Rotate class names for better readability
plt.tight_layout()  # Adjust layout to prevent clipping
plt.show()

predicted_idx = torch.argmax(probabilities).item()
confidence = percentiles[predicted_idx]  # Get the confidence level as a percentage

print(f"Predicted Class Index: {predicted_idx}")

# Get the class name using the predicted index
class_names = checkpoint['class_names']
predicted_class = class_names[predicted_idx]

print(f"Predicted Class: {predicted_class}")
print(f"Confidence Level: {confidence:.2f}%")  # Display confidence as a percentage