import torch
import torch.nn.functional as F
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchvision import transforms
from train_model import CustomCNN

# Page configuration
st.set_page_config(
    page_title="Intel Image Classifier",
    page_icon="🖼️",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f8ff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
    }
    .confidence-label {
        font-weight: bold;
        margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load('/home/muhammad_adib/imgclass_cnn/intel_cnn_model.pth', map_location=device)
    model = CustomCNN(num_classes=len(checkpoint['class_names'])).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint['class_names'], device


def preprocess_image(image):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


class GradCAM:
    """Grad-CAM implementation for CNN visualization"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handles = []
        
        # Register hooks
        self.handles.append(target_layer.register_forward_hook(self.save_activation))
        self.handles.append(target_layer.register_full_backward_hook(self.save_gradient))
    
    def save_activation(self, module, input, output):
        # Clone the output to avoid in-place modification issues
        self.activations = output.clone().detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        # Clone the gradient to avoid in-place modification issues
        self.gradients = grad_output[0].clone().detach()
    
    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
    
    def generate_cam(self, input_tensor, target_class=None):
        """Generate Grad-CAM heatmap"""
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot)
        
        # Compute Grad-CAM
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0].clone()
        
        # Weight activations by gradients
        for i in range(activations.shape[0]):
            activations[i, :, :] = activations[i, :, :] * pooled_gradients[i]
        
        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= (heatmap.max() + 1e-8)
        
        # Remove hooks after use
        self.remove_hooks()
        
        return heatmap


def apply_gradcam_to_image(image, heatmap, alpha=0.4):
    """Overlay Grad-CAM heatmap on original image"""
    # Resize heatmap to match image size
    heatmap_resized = np.array(Image.fromarray(np.uint8(255 * heatmap)).resize(image.size, Image.BILINEAR))
    
    # Apply colormap
    colormap = cm.jet(heatmap_resized / 255.0)[:, :, :3]
    colormap = np.uint8(255 * colormap)
    
    # Convert original image to numpy
    img_array = np.array(image)
    
    # Blend images
    superimposed = (alpha * colormap + (1 - alpha) * img_array).astype(np.uint8)
    
    return superimposed, heatmap_resized


def predict_image(model, image, class_names, device):
    """Make prediction on an image"""
    input_tensor = preprocess_image(image).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
    
    # Get predictions and confidences
    confidences = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
    predicted_class = max(confidences, key=confidences.get)
    
    return predicted_class, confidences


def generate_gradcam(model, image, device):
    """Generate Grad-CAM for the image"""
    import copy
    
    # Create a copy of the model for Grad-CAM to avoid in-place operation issues
    model_copy = copy.deepcopy(model)
    model_copy.eval()
    
    # Use the last convolutional layer (conv4) - target the Conv2d before the last ReLU
    grad_cam = GradCAM(model_copy, model_copy.conv4[4])  # Target Conv2d in conv4
    
    input_tensor = preprocess_image(image).to(device)
    input_tensor.requires_grad = True
    
    heatmap = grad_cam.generate_cam(input_tensor)
    
    # Clean up
    del model_copy
    
    return heatmap


def plot_confidence_bars(confidences, predicted_class):
    """Create confidence bar chart"""
    fig, ax = plt.subplots(figsize=(10, 4))
    
    classes = list(confidences.keys())
    values = [confidences[c] * 100 for c in classes]
    
    # Color bars: highlight predicted class
    colors = ['#1E88E5' if c == predicted_class else '#90CAF9' for c in classes]
    
    bars = ax.barh(classes, values, color=colors, edgecolor='white', height=0.6)
    
    # Add percentage labels
    for bar, value in zip(bars, values):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f'{value:.1f}%', va='center', fontsize=11, fontweight='bold')
    
    ax.set_xlim(0, 110)
    ax.set_xlabel('Confidence (%)', fontsize=12)
    ax.set_title('Classification Confidence Levels', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig


# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🖼️ Intel Image Classification</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload an image to classify it into one of 6 categories: Buildings, Forest, Glacier, Mountain, Sea, or Street</p>', unsafe_allow_html=True)
    
    # Load model
    try:
        model, class_names, device = load_model()
        st.sidebar.success(f"✅ Model loaded successfully!")
        st.sidebar.info(f"📱 Device: {device}")
        st.sidebar.write("**Classes:**", ", ".join(class_names))
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return
    
    # File uploader
    st.markdown("---")
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image of a natural scene or building"
    )
    
    if uploaded_file is not None:
        # Load and display original image
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 Original Image")
            st.image(image, use_column_width=True)
        
        # Make prediction
        with st.spinner("🔍 Classifying image..."):
            predicted_class, confidences = predict_image(model, image, class_names, device)
        
        # Generate Grad-CAM
        with st.spinner("🎨 Generating Grad-CAM visualization..."):
            heatmap = generate_gradcam(model, image, device)
            gradcam_overlay, heatmap_resized = apply_gradcam_to_image(image, heatmap)
        
        with col2:
            st.subheader("🔥 Grad-CAM Visualization")
            st.image(gradcam_overlay, use_column_width=True)
            st.caption("Highlighted regions show areas the model focused on for classification")
        
        # Display prediction result
        st.markdown("---")
        st.subheader("🎯 Prediction Result")
        
        confidence_value = confidences[predicted_class] * 100
        
        # Prediction card
        col_pred1, col_pred2, col_pred3 = st.columns([1, 2, 1])
        with col_pred2:
            st.markdown(f"""
            <div class="prediction-box">
                <h2 style="text-align: center; color: #1E88E5; margin: 0;">
                    {predicted_class.upper()}
                </h2>
                <p style="text-align: center; font-size: 1.5rem; margin: 10px 0;">
                    Confidence: <strong>{confidence_value:.2f}%</strong>
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Confidence bars
        st.markdown("---")
        st.subheader("📊 Confidence Levels for All Classes")
        
        fig = plot_confidence_bars(confidences, predicted_class)
        st.pyplot(fig)
        
        # Additional details in expander
        with st.expander("📋 Detailed Confidence Scores"):
            sorted_confidences = sorted(confidences.items(), key=lambda x: x[1], reverse=True)
            for class_name, conf in sorted_confidences:
                emoji = "🥇" if class_name == predicted_class else "  "
                st.write(f"{emoji} **{class_name}**: {conf*100:.4f}%")
    
    else:
        # Show sample usage
        st.info("👆 Please upload an image to get started!")
        
        with st.expander("ℹ️ About this app"):
            st.markdown("""
            This application uses a **Custom CNN model** trained on the Intel Image Classification dataset.
            
            **Features:**
            - 🖼️ **Image Classification**: Upload any natural scene image to classify it
            - 📊 **Confidence Bars**: See the model's confidence for each class
            - 🔥 **Grad-CAM Visualization**: Understand which parts of the image influenced the prediction
            
            **Supported Classes:**
            - 🏢 Buildings
            - 🌲 Forest
            - 🧊 Glacier
            - ⛰️ Mountain
            - 🌊 Sea
            - 🛣️ Street
            """)


if __name__ == "__main__":
    main()