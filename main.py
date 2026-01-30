import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from train_model import CustomCNN
import streamlit as st

# Ensure page config is set before any other Streamlit API usage
st.set_page_config(page_title="Grad-CAM Visualizer", layout="wide")

class GradCAM:
    """
    Improved Grad-CAM implementation with proper hook management,
    support for Grad-CAM++ and guided backpropagation.
    """
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        
        # Use register_forward_hook and store activations
        self.hooks.append(
            target_layer.register_forward_hook(self._save_activation)
        )
        # Use tensor hook on activations for gradients (avoids inplace issues)

    def _save_activation(self, module, input, output):
        """Save forward activations (clone to avoid inplace modification issues)."""
        self.activations = output.clone()
        # Register hook on the cloned activation to capture gradients
        if output.requires_grad:
            output.register_hook(self._save_gradient_hook)

    def _save_gradient_hook(self, grad):
        """Hook function to save gradients from activation tensor."""
        self.gradients = grad.clone()

    def remove_hooks(self):
        """Clean up hooks to prevent memory leaks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def generate_cam(self, input_tensor, class_idx=None, method='gradcam'):
        """
        Generate CAM heatmap.
        
        Args:
            input_tensor: Preprocessed input image tensor
            class_idx: Target class index (None for predicted class)
            method: 'gradcam' or 'gradcam++' for different weighting schemes
        
        Returns:
            Normalized CAM heatmap
        """
        # Enable gradients for input
        input_tensor = input_tensor.clone().requires_grad_(True)
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Use predicted class if not specified
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Backward pass for the target class
        self.model.zero_grad()
        target = output[0, class_idx]
        target.backward(retain_graph=True)

        # Get gradients and activations
        gradients = self.gradients.cpu().numpy()
        activations = self.activations.cpu().detach().numpy()
        
        if method == 'gradcam++':
            # Grad-CAM++ weighting (better localization for multiple instances)
            # Compute alpha weights using second and third order gradients
            grad_2 = gradients ** 2
            grad_3 = gradients ** 3
            
            # Sum of activations
            sum_activations = np.sum(activations, axis=(2, 3), keepdims=True)
            
            # Compute alpha (importance weights)
            alpha_num = grad_2
            alpha_denom = 2 * grad_2 + sum_activations * grad_3 + 1e-8
            alpha = alpha_num / alpha_denom
            
            # Apply ReLU to gradients
            weights = np.sum(alpha * np.maximum(gradients, 0), axis=(2, 3))
        else:
            # Standard Grad-CAM: Global average pooling of gradients
            weights = np.mean(gradients, axis=(2, 3))

        # Compute weighted combination of activation maps
        cam = np.zeros(activations.shape[2:], dtype=np.float32)
        for i, w in enumerate(weights[0]):
            cam += w * activations[0, i]

        # Apply ReLU and normalize
        cam = np.maximum(cam, 0)
        
        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
            
        return cam
    
    def __del__(self):
        """Cleanup hooks on deletion."""
        self.remove_hooks()

def apply_colormap(cam, colormap=cv2.COLORMAP_JET):
    """Apply colormap to CAM heatmap."""
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap / 255.0


def overlay_cam(image, cam, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Overlay CAM heatmap on the original image with improved blending.
    
    Args:
        image: Original image as numpy array (H, W, 3) in [0, 1]
        cam: CAM heatmap (H, W) in [0, 1]
        alpha: Blending factor (0 = only image, 1 = only heatmap)
        colormap: OpenCV colormap to use
    
    Returns:
        Blended overlay image
    """
    heatmap = apply_colormap(cam, colormap)
    
    # Weighted overlay with intensity-based blending
    # Areas with higher activation get more heatmap, lower get more original image
    cam_3d = np.expand_dims(cam, axis=2)
    blended = (1 - alpha * cam_3d) * image + alpha * cam_3d * heatmap
    
    # Normalize to valid range
    blended = np.clip(blended, 0, 1)
    return blended


def visualize_gradcam(image, cam, class_name, confidence, method_name='Grad-CAM'):
    """Create a comprehensive visualization with multiple views."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')
    
    # Heatmap only
    heatmap = apply_colormap(cam)
    axes[1].imshow(heatmap)
    axes[1].set_title(f'{method_name} Heatmap', fontsize=12)
    axes[1].axis('off')
    
    # Standard overlay
    overlay = overlay_cam(image, cam, alpha=0.6)
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay (α=0.6)', fontsize=12)
    axes[2].axis('off')
    
    # Highlighted regions (threshold-based)
    threshold = 0.5
    mask = cam > threshold
    highlighted = image.copy()
    highlighted[~mask] = highlighted[~mask] * 0.3  # Dim non-important regions
    axes[3].imshow(highlighted)
    axes[3].set_title('Focus Regions (>50%)', fontsize=12)
    axes[3].axis('off')
    
    # Add overall title with prediction info
    fig.suptitle(
        f'Predicted: {class_name} (Confidence: {confidence:.1%})', 
        fontsize=14, fontweight='bold', y=1.02
    )
    
    plt.tight_layout()
    return fig


def compare_gradcam_methods(image, grad_cam_obj, input_tensor, class_idx, class_name, confidence):
    """Compare Grad-CAM and Grad-CAM++ side by side."""
    # Generate both CAMs
    cam_standard = grad_cam_obj.generate_cam(input_tensor.clone(), class_idx, method='gradcam')
    cam_plusplus = grad_cam_obj.generate_cam(input_tensor.clone(), class_idx, method='gradcam++')
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    
    # Row 1: Standard Grad-CAM
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original', fontsize=11)
    axes[0, 0].axis('off')
    
    cam_resized = cv2.resize(cam_standard, (image.shape[1], image.shape[0]))
    overlay_std = overlay_cam(image, cam_resized, alpha=0.6)
    axes[0, 1].imshow(overlay_std)
    axes[0, 1].set_title('Grad-CAM', fontsize=11)
    axes[0, 1].axis('off')
    
    heatmap_std = apply_colormap(cam_resized)
    axes[0, 2].imshow(heatmap_std)
    axes[0, 2].set_title('Grad-CAM Heatmap', fontsize=11)
    axes[0, 2].axis('off')
    
    # Row 2: Grad-CAM++
    axes[1, 0].imshow(image)
    axes[1, 0].set_title('Original', fontsize=11)
    axes[1, 0].axis('off')
    
    cam_pp_resized = cv2.resize(cam_plusplus, (image.shape[1], image.shape[0]))
    overlay_pp = overlay_cam(image, cam_pp_resized, alpha=0.6)
    axes[1, 1].imshow(overlay_pp)
    axes[1, 1].set_title('Grad-CAM++', fontsize=11)
    axes[1, 1].axis('off')
    
    heatmap_pp = apply_colormap(cam_pp_resized)
    axes[1, 2].imshow(heatmap_pp)
    axes[1, 2].set_title('Grad-CAM++ Heatmap', fontsize=11)
    axes[1, 2].axis('off')
    
    fig.suptitle(
        f'Predicted: {class_name} (Confidence: {confidence:.1%})', 
        fontsize=14, fontweight='bold'
    )
    
    plt.tight_layout()
    return fig

# --- add: bar chart helper ---
def plot_confidence_bars(class_labels, class_probs, title="Confidence"):
    """
    class_labels: list[str]
    class_probs: 1D array-like in [0,1]
    """
    probs = np.asarray(class_probs, dtype=np.float32)

    # Slightly wider figure for readability when there are multiple classes
    fig_w = max(10, min(18, 1.2 * len(class_labels)))
    fig, ax = plt.subplots(figsize=(fig_w, 4))

    bars = ax.bar(range(len(class_labels)), probs, color="#4C78A8")
    ax.set_title(title)
    ax.set_ylabel("Probability")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(range(len(class_labels)))
    ax.set_xticklabels(class_labels, rotation=35, ha="right")

    # value labels on top of each bar
    for b, p in zip(bars, probs):
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            min(1.0, p + 0.02),
            f"{p*100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    return fig

# Load the model and checkpoint (cache to avoid reload)
@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load('/home/muhammad_adib/imgclass_cnn/intel_cnn_model.pth', map_location=device)
    model = CustomCNN(num_classes=len(checkpoint['class_names'])).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint, device

model, checkpoint, device = load_model()
target_layer = model.conv4[8]  # Last ReLU in conv4 block

# Preprocessing pipeline
IMG_SIZE = 150
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def process_image(image_file):
    img = Image.open(image_file).convert('RGB')
    input_tensor = preprocess(img).unsqueeze(0).to(device)
    img_vis = np.array(img.resize((IMG_SIZE, IMG_SIZE))) / 255.0
    return img, input_tensor, img_vis

# Streamlit UI setup
st.title("Grad-CAM Visualizer for CustomCNN")
st.write(
    "Upload an image to visualize Grad-CAM and Grad-CAM++ explanations for the model's predictions."
)

# Sidebar for options
st.sidebar.header("Options")
method = st.sidebar.selectbox("CAM Method", ["Grad-CAM", "Grad-CAM++"])
show_compare = st.sidebar.checkbox("Compare Grad-CAM vs Grad-CAM++", value=True)
uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img, input_tensor, img_vis = process_image(uploaded_file)
    grad_cam = GradCAM(model, target_layer)
    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output[0], dim=0)
        predicted_idx = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_idx].item()
    class_name = checkpoint['class_names'][predicted_idx]

    st.subheader("Confidence Bar Chart")

    all_probs = probabilities.detach().cpu().numpy()
    all_labels = list(checkpoint["class_names"])

    # Sort bars by confidence (descending) for readability
    order = np.argsort(-all_probs)
    sorted_probs = all_probs[order]
    sorted_labels = [all_labels[i] for i in order]

    fig_bar = plot_confidence_bars(sorted_labels, sorted_probs, title="Class Confidences")
    st.pyplot(fig_bar)
    plt.close(fig_bar)

    # Generate CAM
    cam_method = 'gradcam' if method == "Grad-CAM" else 'gradcam++'
    cam = grad_cam.generate_cam(input_tensor.clone(), predicted_idx, method=cam_method)
    cam_resized = cv2.resize(cam, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

    # Visualize
    st.subheader(f"{method} Visualization")
    fig1 = visualize_gradcam(img_vis, cam_resized, class_name, confidence, method)
    st.pyplot(fig1)
    plt.close(fig1)

    if show_compare:
        st.subheader("Grad-CAM vs Grad-CAM++ Comparison")
        fig2 = compare_gradcam_methods(img_vis, grad_cam, input_tensor, predicted_idx, class_name, confidence)
        st.pyplot(fig2)
        plt.close(fig2)

    grad_cam.remove_hooks()
else:
    st.info("Please upload an image to begin.")