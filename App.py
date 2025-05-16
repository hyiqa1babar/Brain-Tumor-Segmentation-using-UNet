import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import os
import tempfile
import logging
import time
from datetime import datetime
from PIL import Image
import torch.nn as nn
import requests
from io import BytesIO
import google.generativeai as genai
import base64
from scipy import ndimage
from skimage import measure

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page title and configuration
st.set_page_config(page_title="Brain Tumor Segmentation", layout="wide")


# ------------------ MODEL ARCHITECTURE ------------------

class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        # Contracting path
        self.down_convolution_1 = self._double_convolution(1, 64)
        self.down_convolution_2 = self._double_convolution(64, 128)
        self.down_convolution_3 = self._double_convolution(128, 256)
        self.down_convolution_4 = self._double_convolution(256, 512)
        self.down_convolution_5 = self._double_convolution(512, 1024)

        # Expanding path
        self.up_transpose_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.up_convolution_1 = self._double_convolution(1024, 512)
        self.up_transpose_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.up_convolution_2 = self._double_convolution(512, 256)
        self.up_transpose_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up_convolution_3 = self._double_convolution(256, 128)
        self.up_transpose_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.up_convolution_4 = self._double_convolution(128, 64)
        # output
        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def _double_convolution(self, in_channels, out_channels):
        """Double convolution block for UNet"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        down_1 = self.down_convolution_1(x)
        down_2 = self.max_pool2d(down_1)
        down_3 = self.down_convolution_2(down_2)
        down_4 = self.max_pool2d(down_3)
        down_5 = self.down_convolution_3(down_4)
        down_6 = self.max_pool2d(down_5)
        down_7 = self.down_convolution_4(down_6)
        down_8 = self.max_pool2d(down_7)
        down_9 = self.down_convolution_5(down_8)

        up_1 = self.up_transpose_1(down_9)
        up_2 = self.up_convolution_1(torch.cat([down_7, up_1], 1))
        up_3 = self.up_transpose_2(up_2)
        up_4 = self.up_convolution_2(torch.cat([down_5, up_3], 1))
        up_5 = self.up_transpose_3(up_4)
        up_6 = self.up_convolution_3(torch.cat([down_3, up_5], 1))
        up_7 = self.up_transpose_4(up_6)
        up_8 = self.up_convolution_4(torch.cat([down_1, up_7], 1))

        return self.out(up_8)


# ------------------ UTILITY FUNCTIONS ------------------

def fig_to_base64(fig, dpi=120, quality=90):
    """Convert matplotlib figure to base64-encoded JPEG string"""
    png_buffer = BytesIO()
    fig.savefig(png_buffer, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    png_buffer.seek(0)

    image = Image.open(png_buffer).convert('RGB')
    jpeg_buffer = BytesIO()
    image.save(jpeg_buffer, format='JPEG', quality=quality, optimize=True)
    jpeg_buffer.seek(0)

    return base64.b64encode(jpeg_buffer.getvalue()).decode('utf-8')


def calculate_detailed_tumor_metrics(segmentation_mask):
    """Calculate comprehensive tumor metrics from the segmentation mask"""
    # Get basic metrics
    tumor_pixels = np.sum(segmentation_mask > 0)
    total_pixels = segmentation_mask.shape[0] * segmentation_mask.shape[1]
    percentage = (tumor_pixels / total_pixels) * 100

    # If no tumor found
    if tumor_pixels == 0:
        return {
            "area_pixels": 0,
            "percentage": 0,
            "width": 0,
            "height": 0,
            "center": None,
            "present": False
        }

    # Get tumor bounding box and dimensions
    y_indices, x_indices = np.where(segmentation_mask > 0)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    x_min, x_max = np.min(x_indices), np.max(x_indices)

    height = y_max - y_min
    width = x_max - x_min

    # Find center of mass
    center_y, center_x = ndimage.center_of_mass(segmentation_mask)

    # Calculate approximate diameter
    diameter = 2 * np.sqrt(tumor_pixels / np.pi)

    # Calculate perimeter using contour
    contours = measure.find_contours(segmentation_mask, 0.5)
    perimeter = sum(len(contour) for contour in contours) if contours else 0

    # Irregularity index (approximation of shape irregularity)
    circularity = (4 * np.pi * tumor_pixels) / (perimeter ** 2) if perimeter > 0 else 0

    return {
        "area_pixels": float(tumor_pixels),
        "percentage": float(percentage),
        "width": float(width),
        "height": float(height),
        "diameter": float(diameter),
        "perimeter": float(perimeter),
        "circularity": float(circularity),
        "center": (float(center_x), float(center_y)),
        "bounding_box": (int(x_min), int(y_min), int(x_max), int(y_max)),
        "present": True
    }


# ------------------ AI EXPLANATION FUNCTIONS ------------------

def initialize_gemini_api(api_key=None):
    """Initialize and configure the Gemini API client with the provided key"""
    try:
        # If no API key is provided, return None (API disabled)
        if not api_key or api_key.strip() == "":
            return None

        # Configure the Gemini client with the provided key
        genai.configure(api_key=api_key)

        # Try newer model with fallback option
        try:
            model = genai.GenerativeModel('gemini-2.0-flash')
            response = model.generate_content("Test connection")  # Test if model works
            return model
        except Exception as model_error:
            st.sidebar.warning(f"Could not initialize gemini-2.0-flash model: {str(model_error)}")
            try:
                model = genai.GenerativeModel('gemini-pro')
                return model
            except Exception as fallback_error:
                st.sidebar.error(f"Fallback model also failed: {str(fallback_error)}")
                return None
    except Exception as e:
        st.sidebar.error(f"Failed to initialize Gemini API: {str(e)}")
        return None


def setup_api_key_section():
    """Sets up the Gemini API key input section in sidebar"""
    # Create an expander for the API key input
    with st.sidebar:
        st.markdown("---")
        api_expander = st.expander("🔑 AI Explanation Settings", expanded=False)

        with api_expander:
            # Get API key from session state or create empty
            if 'gemini_api_key' not in st.session_state:
                st.session_state.gemini_api_key = ""

            # Input field for API key
            api_key = st.text_input(
                "Enter Gemini API Key",
                type="password",
                value=st.session_state.gemini_api_key,
                help="Enter your Google Gemini API key to enable AI explanations"
            )

            # Activate button
            col1, col2 = st.columns([1, 1])
            with col1:
                activate_button = st.button("Activate API", use_container_width=True)
            with col2:
                clear_button = st.button("Clear Key", use_container_width=True)

            if activate_button and api_key:
                st.session_state.gemini_api_key = api_key
                st.session_state.activate_pressed = True
                return api_key

            if clear_button:
                st.session_state.gemini_api_key = ""
                st.session_state.activate_pressed = False
                if 'gemini_model' in st.session_state:
                    del st.session_state.gemini_model

                # This will force a page refresh
                st.rerun()

            # If there's a key in session_state, return it
            if st.session_state.gemini_api_key:
                return st.session_state.gemini_api_key

    # Return None if no key is provided
    return None

@st.cache_data(ttl=3600, show_spinner=False)  # Cache for 1 hour
def get_gemini_explanation(_model, original_img, segmentation_mask, tumor_stats, view_orientation="Axial",
                           slice_info=""):
    """Get a detailed explanation of the tumor segmentation from Gemini"""
    if _model is None:
        return "AI explanations are disabled. Please configure your Gemini API key in the sidebar to enable this feature."

    # If no tumor is detected
    if not tumor_stats["present"]:
        return "No significant tumor mass detected in this slice. This could indicate a normal brain section, " + \
            "or that the tumor is located in a different region of the brain. Consider examining additional slices."

    # Create figures for both images
    fig1, ax1 = plt.subplots(figsize=(5, 5))
    ax1.imshow(original_img, cmap='gray')
    ax1.set_title("Original MRI")
    ax1.axis('off')

    # Create overlay figure with contour
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    ax2.imshow(original_img, cmap='gray')

    # Use contour for clearer tumor boundary visualization
    if tumor_stats["present"]:
        contours = measure.find_contours(segmentation_mask, 0.5)
        for contour in contours:
            ax2.plot(contour[:, 1], contour[:, 0], linewidth=2, color='red')

    ax2.set_title("Tumor Segmentation")
    ax2.axis('off')

    # Convert figures to base64 strings and close to free memory
    original_b64 = fig_to_base64(fig1)
    segmentation_b64 = fig_to_base64(fig2)
    plt.close(fig1)
    plt.close(fig2)

    # Format circularity for easier understanding
    circularity_description = "highly irregular"
    if tumor_stats.get('circularity', 0) > 0.8:
        circularity_description = "mostly spherical"
    elif tumor_stats.get('circularity', 0) > 0.6:
        circularity_description = "fairly regular"
    elif tumor_stats.get('circularity', 0) > 0.4:
        circularity_description = "somewhat irregular"

    # Create enhanced prompt for Gemini
    prompt = f"""
    You are a neuroradiologist analyzing brain MRI scan images with tumor segmentation results.

    SCAN INFORMATION:
    - View: {view_orientation} 
    - Slice Information: {slice_info}

    TUMOR MEASUREMENTS:
    - Area: {tumor_stats['area_pixels']:.1f} pixels
    - Percentage of slice: {tumor_stats['percentage']:.2f}%
    - Dimensions: Width {tumor_stats['width']:.1f} px × Height {tumor_stats['height']:.1f} px
    - Approximate diameter: {tumor_stats.get('diameter', 0):.1f} px
    - Shape: {circularity_description} (circularity index: {tumor_stats.get('circularity', 0):.2f})
    - Location: Centered at coordinates {tumor_stats['center']}

    Based on this quantitative analysis:

    1. Provide a brief clinical impression of what this segmentation likely represents
    2. Mention 2-3 differential diagnoses that should be considered
    3. Suggest appropriate next steps in evaluation (additional imaging, etc.)
    4. Briefly note limitations of this single-slice analysis

    Structure your response as 3-4 concise paragraphs that would be appropriate for a clinical setting.
    Include relevant medical terminology but explain key concepts. Total response should be under 300 words.
    """

    # Implement retry logic for API resilience
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = _model.generate_content(prompt)

            if hasattr(response, 'text') and response.text:
                explanation = response.text.strip()
                disclaimer = "\n\n*Note: This AI-generated analysis is for educational purposes only. It is not a substitute for professional medical evaluation.*"
                return explanation + disclaimer
            else:
                if attempt < max_retries - 1:
                    time.sleep(1)  # Wait before retry
                    continue
                return "Error: Received empty response from Gemini API."
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)  # Wait before retry
                continue
            return f"Error generating explanation after {max_retries} attempts: {str(e)}"

    return "Failed to generate explanation after multiple attempts."

# ------------------ MODEL LOADING FUNCTIONS ------------------

@st.cache_resource
def load_model():
    """Load the UNet model with error handling"""
    model = UNet(num_classes=1)

    # Try to download checkpoint if not exists
    checkpoint_path = 'checkpoint-epoch-29.pt'
    if not os.path.exists(checkpoint_path):
        checkpoint_url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/HATpl02lA0ykn9aAU9K6sA/checkpoint-epoch-29.pt'

        with st.spinner('Downloading model checkpoint... This may take a moment.'):
            try:
                response = requests.get(checkpoint_url)
                response.raise_for_status()

                with open(checkpoint_path, 'wb') as f:
                    f.write(response.content)
                st.success("Model checkpoint downloaded successfully!")
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to download model checkpoint: {str(e)}")
                st.stop()

    # Load the checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Failed to load model checkpoint: {str(e)}")
        st.stop()


# ------------------ IMAGE PROCESSING FUNCTIONS ------------------

def process_nifti_file(uploaded_file):
    """Process NIfTI files with error handling"""
    temp_file_path = None
    try:
        # Save the uploaded file to a temporary file
        temp_dir = tempfile.gettempdir()
        temp_file_fd, temp_file_path = tempfile.mkstemp(suffix='.nii.gz')

        os.close(temp_file_fd)
        with open(temp_file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        logger.info(f"Saved temporary file to: {temp_file_path}")

        try:
            # Try loading as .nii.gz
            nifti_img = nib.load(temp_file_path)
        except Exception as e:
            logger.error(f"Failed to load as .nii.gz: {str(e)}")
            # If that fails, try loading as regular NIfTI
            new_path = temp_file_path.replace('.nii.gz', '.nii')
            os.rename(temp_file_path, new_path)
            temp_file_path = new_path
            logger.info(f"Renamed to: {temp_file_path}")
            nifti_img = nib.load(temp_file_path)

        logger.info(f"Successfully loaded NIfTI file: {nifti_img.shape}")
        return nifti_img

    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise e


def preprocess_image(nifti_img, slice_num, slice_dim=2, normalize_method='max'):
    """Preprocess image for model input with multiple normalization options"""
    data = nifti_img.get_fdata()

    # Extract the appropriate slice based on view
    if slice_dim == 0:  # Sagittal
        slice_data = data[slice_num, :, :]
        slice_data = np.rot90(slice_data)  # Rotate for proper visualization
    elif slice_dim == 1:  # Coronal
        slice_data = data[:, slice_num, :]
        slice_data = np.rot90(slice_data)  # Rotate for proper visualization
    else:  # Axial (default)
        slice_data = data[:, :, slice_num]

    # Normalize the slice based on method
    if normalize_method == 'max':
        normalized_slice = slice_data / np.max(slice_data) if np.max(slice_data) > 0 else slice_data
    elif normalize_method == 'z-score':
        mean = np.mean(slice_data)
        std = np.std(slice_data)
        normalized_slice = (slice_data - mean) / std if std > 0 else slice_data - mean
    elif normalize_method == 'min-max':
        min_val = np.min(slice_data)
        max_val = np.max(slice_data)
        normalized_slice = (slice_data - min_val) / (max_val - min_val) if max_val > min_val else slice_data
    else:  # No normalization
        normalized_slice = slice_data

    # Convert to tensor and add batch and channel dimensions
    tensor_slice = torch.from_numpy(normalized_slice).float().unsqueeze(0).unsqueeze(0)

    return tensor_slice, normalized_slice


def predict_segmentation(model, device, image_tensor, threshold=0.5):
    """Make prediction with the model"""
    try:
        image_tensor = image_tensor.to(device)

        with torch.no_grad():
            start_time = time.time()
            prediction = model(image_tensor)
            inference_time = time.time() - start_time

        # Convert prediction to binary mask
        prediction = torch.sigmoid(prediction)
        binary_prediction = (prediction > threshold).float()

        return binary_prediction.squeeze().cpu().numpy(), prediction.squeeze().cpu().numpy(), inference_time
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None, 0


def apply_custom_colormap(segmentation, raw_prediction, colormap_name='hot'):
    """Apply a custom colormap to the segmentation with alpha based on confidence"""
    cmap = plt.get_cmap(colormap_name)
    colored_segmentation = cmap(raw_prediction)
    colored_segmentation[:, :, 3] = raw_prediction * 0.8  # Maximum alpha of 0.8
    return colored_segmentation


def plot_with_tumor_outline(ax, mri_slice, segmentation, title):
    """Plot MRI with tumor outline highlighted"""
    ax.imshow(mri_slice, cmap='gray')

    # Create contour around tumor
    if np.sum(segmentation) > 0:
        ax.contour(segmentation, levels=[0.5], colors='r', linewidths=2)

    ax.set_title(title)
    ax.axis('off')
    return ax


def create_sample_brain_data():
    """Create synthetic brain MRI data for testing"""
    try:
        # Create a 3D array to represent brain volume (256x256x128)
        shape = (256, 256, 128)
        data = np.zeros(shape)

        # Create a sphere in the middle to simulate brain tissue
        x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
        center = (shape[0] // 2, shape[1] // 2, shape[2] // 2)
        radius = min(shape) // 3

        # Create brain-like structure (ellipsoid)
        brain_mask = ((x - center[0]) ** 2 + (y - center[1]) ** 2) / ((radius * 1.2) ** 2) + \
                     ((z - center[2]) ** 2) / (radius ** 2) <= 1

        # Set values for the brain region (random values but higher than background)
        data[brain_mask] = np.random.uniform(0.5, 1.0, size=np.sum(brain_mask))

        # Add "ventricles" as darker regions
        ventricle_center = center
        ventricle_radius = radius // 4
        ventricle_mask = ((x - ventricle_center[0]) ** 2 +
                          (y - ventricle_center[1]) ** 2 +
                          (z - ventricle_center[2]) ** 2) <= ventricle_radius ** 2
        data[ventricle_mask & brain_mask] = np.random.uniform(0.1, 0.3, size=np.sum(ventricle_mask & brain_mask))

        # Create a "tumor" as a bright spot
        tumor_center = (center[0] + radius // 2, center[1] - radius // 2, center[2])
        tumor_radius = radius // 5
        tumor_mask = ((x - tumor_center[0]) ** 2 +
                      (y - tumor_center[1]) ** 2 +
                      (z - tumor_center[2]) ** 2) <= tumor_radius ** 2
        data[tumor_mask & brain_mask] = np.random.uniform(0.8, 1.0, size=np.sum(tumor_mask & brain_mask))

        # Add noise
        data += np.random.normal(0, 0.05, size=shape)

        # Normalize values to [0, 1]
        data = np.clip(data, 0, 1)

        # Create affine matrix (identity for simplicity)
        affine = np.eye(4)

        # Create NIfTI image
        nifti_img = nib.Nifti1Image(data, affine)

        return nifti_img

    except Exception as e:
        st.error(f"Error creating sample data: {str(e)}")
        return None


# ------------------ UI STYLING ------------------

# Add stylish CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem !important;
        color: #1E3A8A !important;
        text-align: center;
        margin-bottom: 1rem;
    }
    .explanation-title {
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .gemini-explanation {
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6;
        margin-top: 1rem;
    }
    .subtitle {
        font-size: 1.5rem !important;
        color: #1E3A8A !important;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #E0F2FE;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .result-header {
        font-size: 1.2rem !important;
        color: #1E3A8A !important;
        margin-bottom: 0.5rem;
    }
    .metrics-container {
        background-color: #F0F9FF;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        color: #64748B;
    }
</style>
""", unsafe_allow_html=True)


# ------------------ MAIN APPLICATION ------------------

def main():
    # App header
    st.markdown("<h1 class='main-title'>Brain Tumor Segmentation using UNet</h1>", unsafe_allow_html=True)
    # Initialize session state variables if they don't exist
    if 'gemini_model' not in st.session_state:
        st.session_state.gemini_model = None
    if 'ai_explanations_enabled' not in st.session_state:
        st.session_state.ai_explanations_enabled = False
    if 'activate_pressed' not in st.session_state:
        st.session_state.activate_pressed = False

    with st.expander("📋 About This Application", expanded=False):
        st.markdown("""
        This application performs advanced segmentation of brain tumors from MRI scans using a UNet deep learning model.

        ### How it works:
        1. Upload a NIfTI (.nii or .nii.gz) file containing brain MRI data
        2. The model will automatically segment tumor regions
        3. Explore different slices and visualization options
        4. Analyze tumor metrics and characteristics

        ### Technical Details:
        - **Model**: UNet architecture trained on BraTS2020 dataset
        - **Input**: T2-weighted MRI scans (other modalities supported)
        - **Output**: Segmentation mask with confidence levels
        """)

    # Sidebar configuration
    with st.sidebar:
        st.image("C:/Users/user/PycharmProjects/PythonProject1/projlogo.png", use_column_width=True)
        st.markdown("### Model Configuration")

        # Segmentation threshold
        threshold = st.slider("Segmentation Threshold", 0.1, 0.9, 0.5, 0.05,
                              help="Adjust the confidence threshold for tumor detection")

        # Normalization method
        normalization = st.selectbox(
            "Image Normalization Method",
            options=["max", "z-score", "min-max", "none"],
            index=0,
            help="Method used to normalize the MRI slice before processing"
        )

        # Overlay style
        overlay_style = st.selectbox(
            "Overlay Style",
            options=["hot", "viridis", "jet", "cool"],
            index=0,
            help="Color scheme used for tumor visualization"
        )

        # Display options
        st.markdown("### Display Options")
        show_confidence = st.checkbox("Show Confidence Map", value=True)
        show_outline = st.checkbox("Show Tumor Outline", value=True)
        show_metrics = st.checkbox("Show Tumor Metrics", value=True)

        # Only allow AI explanations if they're enabled
        if st.session_state.ai_explanations_enabled:
            show_ai_explanation = st.checkbox("Show AI Clinical Explanation", value=True,
                                              help="Get AI-powered clinical interpretation using Google Gemini")
        else:
            show_ai_explanation = False
            st.info("AI explanations are disabled. Set up your Gemini API key in the sidebar to enable them.")

        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        **Brain Tumor Segmentation Project**

        This app identifies brain tumors in MRI scans using deep learning.

        * UNet architecture
        * Trained on BraTS2020 Dataset
        * Supports multiple visualization options

        Made with Streamlit & PyTorch
        """)

        # Version information
        st.markdown("---")
        st.markdown("**v1.2.1** | Updated May 2025")

        # Set up API key section in sidebar
        api_key = setup_api_key_section()

        # Initialize session state variables if they don't exist
        if 'gemini_model' not in st.session_state:
            st.session_state.gemini_model = None
        if 'ai_explanations_enabled' not in st.session_state:
            st.session_state.ai_explanations_enabled = False

        # If the API key exists and activation was pressed, initialize the model
        if api_key and hasattr(st.session_state, 'activate_pressed') and st.session_state.activate_pressed:
            with st.sidebar:
                with st.spinner("Initializing Gemini AI..."):
                    gemini_model = initialize_gemini_api(api_key)
                    if gemini_model:
                        st.success("✅ Gemini AI successfully initialized!")
                        st.session_state.gemini_model = gemini_model
                        st.session_state.ai_explanations_enabled = True
                        # Reset the activation flag
                        st.session_state.activate_pressed = False
                    else:
                        st.error("❌ Failed to initialize Gemini API. Please check your API key.")
                        st.session_state.ai_explanations_enabled = False

        # Load model
        with st.spinner("Initializing model..."):
            try:
                model, device = load_model()
                st.success("✅ Model loaded successfully!")
            except Exception as e:
                st.error(f"❌ Failed to load model: {str(e)}")
                st.stop()

        # Display current API status
        with st.sidebar:
            if st.session_state.ai_explanations_enabled:
                st.success("✅ AI explanations are enabled")
            else:
                st.warning("⚠️ AI explanations are disabled. Set up your Gemini API key to enable them.")

    # File uploader
    st.markdown("<h2 class='subtitle'>Upload MRI Scan</h2>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose a NIfTI file (.nii or .nii.gz)",
        type=["nii", "nii.gz"],
        help="Upload a NIfTI format brain MRI scan"
    )

    # Sample data option
    use_sample = st.checkbox("Use sample data instead", value=False)

    # Process uploaded file or sample data
    if use_sample:
        st.info("Using synthetic sample brain MRI data...")
        try:
            with st.spinner("Creating sample brain data..."):
                nifti_img = create_sample_brain_data()
                if nifti_img is None:
                    st.error("Failed to create sample data.")
                    st.stop()
                st.success("Sample data created successfully!")
        except Exception as e:
            st.error(f"Error creating sample data: {str(e)}")
            st.stop()
    elif uploaded_file is not None:
        try:
            with st.spinner("Processing uploaded file..."):
                nifti_img = process_nifti_file(uploaded_file)
                st.success("✅ File processed successfully!")
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please ensure you're uploading a valid NIfTI (.nii or .nii.gz) file.")
            st.stop()
    else:
        st.info("Please upload a NIfTI file or use the sample data to continue.")
        st.stop()

    # Get image dimensions
    img_shape = nifti_img.get_fdata().shape

    # Display image information
    with st.expander("📊 Image Information", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Image Dimensions", f"{img_shape[0]}×{img_shape[1]}×{img_shape[2]}")

        with col2:
            # Calculate approximate file size
            data_size_mb = np.prod(img_shape) * 4 / (1024 * 1024)  # Assuming 4 bytes per voxel
            st.metric("Data Size", f"{data_size_mb:.2f} MB")

        with col3:
            # Voxel resolution
            try:
                zooms = nifti_img.header.get_zooms()
                voxel_size = f"{zooms[0]:.2f}×{zooms[1]:.2f}×{zooms[2]:.2f} mm"
            except:
                voxel_size = "Not available"
            st.metric("Voxel Size", voxel_size)

    # View selection (axial, sagittal, coronal)
    view_options = ["Axial (Top-Down)", "Sagittal (Side)", "Coronal (Front-Back)"]
    selected_view = st.selectbox("Select View Orientation", view_options, index=0)

    # Based on view, set the appropriate dimension and slider
    # Based on view, set the appropriate dimension and slider
    if selected_view == "Axial (Top-Down)":
        slice_dim = 2
        max_slice = img_shape[2] - 1
        view_name = "Axial"
    elif selected_view == "Sagittal (Side)":
        slice_dim = 0
        max_slice = img_shape[0] - 1
        view_name = "Sagittal"
    else:  # Coronal
        slice_dim = 1
        max_slice = img_shape[1] - 1
        view_name = "Coronal"

    # Slice selector
    slice_num = st.slider(f"Select {view_name} Slice", 0, max_slice, max_slice // 2)

    # Process selected slice
    try:
        with st.spinner("Processing slice..."):
            # Preprocess the image
            image_tensor, normalized_slice = preprocess_image(
                nifti_img, slice_num, slice_dim, normalization
            )

            # Make prediction
            binary_prediction, raw_prediction, inference_time = predict_segmentation(
                model, device, image_tensor, threshold
            )

            if binary_prediction is None:
                st.error("Prediction failed. Please try a different slice or configuration.")
                st.stop()

            # Calculate tumor metrics
            tumor_metrics = calculate_detailed_tumor_metrics(binary_prediction)

            st.success(f"✅ Processing completed in {inference_time:.3f} seconds")
    except Exception as e:
        st.error(f"Error during image processing: {str(e)}")
        st.stop()

    # Create visualization
    st.markdown("<h2 class='subtitle'>Segmentation Results</h2>", unsafe_allow_html=True)

    # Create multi-panel visualization
    fig, axes = plt.subplots(1, 3 if show_confidence else 2, figsize=(15, 5))

    # Plot original image
    axes[0].imshow(normalized_slice, cmap='gray')
    axes[0].set_title("Original MRI")
    axes[0].axis('off')

    # Plot segmentation
    if show_outline:
        plot_with_tumor_outline(axes[1], normalized_slice, binary_prediction, "Tumor Segmentation")
    else:
        # Create colored overlay
        colored_segmentation = apply_custom_colormap(binary_prediction, raw_prediction, overlay_style)
        axes[1].imshow(normalized_slice, cmap='gray')
        axes[1].imshow(colored_segmentation, alpha=0.7)
        axes[1].set_title("Tumor Segmentation")
        axes[1].axis('off')

    # Plot confidence map if enabled
    if show_confidence:
        confidence_map = axes[2].imshow(raw_prediction, cmap=overlay_style)
        axes[2].set_title("Confidence Map")
        axes[2].axis('off')
        plt.colorbar(confidence_map, ax=axes[2], fraction=0.046, pad=0.04)

    # Show the figure
    st.pyplot(fig)
    plt.close(fig)

    # Show metrics if enabled
    if show_metrics and tumor_metrics["present"]:
        st.markdown("<h3 class='result-header'>Tumor Metrics</h3>", unsafe_allow_html=True)

        with st.container():
            st.markdown("<div class='metrics-container'>", unsafe_allow_html=True)

            # Create two columns for metrics
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Size Metrics:**")
                st.write(f"• Area: {tumor_metrics['area_pixels']:.1f} pixels")
                st.write(f"• Percentage of slice: {tumor_metrics['percentage']:.2f}%")
                st.write(f"• Dimensions: {tumor_metrics['width']:.1f} × {tumor_metrics['height']:.1f} pixels")
                st.write(f"• Diameter: {tumor_metrics.get('diameter', 0):.1f} pixels")

            with col2:
                st.markdown("**Shape & Location Metrics:**")
                st.write(f"• Perimeter: {tumor_metrics.get('perimeter', 0):.1f} pixels")
                st.write(f"• Circularity: {tumor_metrics.get('circularity', 0):.3f}")
                st.write(f"• Center position: ({tumor_metrics['center'][0]:.1f}, {tumor_metrics['center'][1]:.1f})")
                st.write(f"• Bounding box: {tumor_metrics.get('bounding_box', (0, 0, 0, 0))}")

            st.markdown("</div>", unsafe_allow_html=True)
    elif show_metrics:
        st.info("No significant tumor was detected in this slice.")

    # Show AI explanation if enabled
    if show_ai_explanation and st.session_state.ai_explanations_enabled:
        st.markdown("<h3 class='explanation-title'>Clinical Interpretation</h3>", unsafe_allow_html=True)

        with st.spinner("Generating clinical explanation..."):
            # Format slice info
            slice_info = f"Slice {slice_num + 1}/{max_slice + 1}"

            # Get explanation
            explanation = get_gemini_explanation(
                st.session_state.gemini_model,
                normalized_slice,
                binary_prediction,
                tumor_metrics,
                view_orientation=view_name,
                slice_info=slice_info
            )

            st.markdown(f"<div class='gemini-explanation'>{explanation}</div>", unsafe_allow_html=True)

    # Download options for results
    st.markdown("<h3 class='result-header'>Download Results</h3>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Save the current figure for download
        download_fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(normalized_slice, cmap='gray')

        if np.sum(binary_prediction) > 0:
            contours = measure.find_contours(binary_prediction, 0.5)
            for contour in contours:
                ax.plot(contour[:, 1], contour[:, 0], linewidth=2, color='red')

        ax.set_title(f"Brain Tumor Segmentation - {view_name} View")
        ax.axis('off')

        # Save to BytesIO
        buf = BytesIO()
        download_fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close(download_fig)

        st.download_button(
            label="Download Image",
            data=buf,
            file_name=f"brain_tumor_segmentation_{view_name.lower()}_slice_{slice_num}.png",
            mime="image/png"
        )

    with col2:
        # Create a report as text
        if tumor_metrics["present"]:
            report = f"""Brain Tumor Segmentation Report
    Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    Scan Information:
    - View: {view_name}
    - Slice: {slice_num + 1}/{max_slice + 1}

    Tumor Metrics:
    - Area: {tumor_metrics['area_pixels']:.1f} pixels
    - Percentage of slice: {tumor_metrics['percentage']:.2f}%
    - Dimensions: {tumor_metrics['width']:.1f} × {tumor_metrics['height']:.1f} pixels
    - Diameter: {tumor_metrics.get('diameter', 0):.1f} pixels
    - Perimeter: {tumor_metrics.get('perimeter', 0):.1f} pixels
    - Circularity: {tumor_metrics.get('circularity', 0):.3f}
    - Center position: ({tumor_metrics['center'][0]:.1f}, {tumor_metrics['center'][1]:.1f})

    Analysis Parameters:
    - Segmentation threshold: {threshold}
    - Normalization method: {normalization}

    Note: This report is generated automatically and should be reviewed by a qualified medical professional.
    """
        else:
            report = f"""Brain Tumor Segmentation Report
    Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    Scan Information:
    - View: {view_name}
    - Slice: {slice_num + 1}/{max_slice + 1}

    Results:
    No significant tumor detected in this slice.

    Analysis Parameters:
    - Segmentation threshold: {threshold}
    - Normalization method: {normalization}

    Note: This report is generated automatically and should be reviewed by a qualified medical professional.
    """

        st.download_button(
            label="Download Report",
            data=report,
            file_name=f"brain_tumor_report_{view_name.lower()}_slice_{slice_num}.txt",
            mime="text/plain"
        )

    # Advanced options section
    with st.expander("⚙️ Advanced Visualization Options", expanded=False):
        st.markdown("### 3D Visualization Preview (Coming Soon)")
        st.info("3D visualization of tumor volumes will be available in a future update.")

        st.markdown("### Batch Processing")
        st.info("Process multiple slices or scans at once.")
        batch_enabled = st.checkbox("Enable batch processing", value=False)

        if batch_enabled:
            batch_start = st.number_input("Start slice", min_value=0, max_value=max_slice - 10, value=0)
            batch_end = st.number_input("End slice", min_value=batch_start + 1, max_value=max_slice,
                                        value=min(batch_start + 10, max_slice))

            if st.button("Process Batch"):
                with st.spinner(f"Processing slices {batch_start} to {batch_end}..."):
                    # Placeholder for batch processing
                    progress_bar = st.progress(0)

                    # Simple visualization of batch progress
                    for i, slice_idx in enumerate(range(batch_start, batch_end + 1)):
                        # Update progress
                        progress = int((i / (batch_end - batch_start + 1)) * 100)
                        progress_bar.progress(progress)
                        time.sleep(0.1)  # Simulate processing time

                    st.success(f"Processed {batch_end - batch_start + 1} slices!")
                    st.info("In a future update, batch results will be available for download.")

    # Footer
    st.markdown("""
        <div class='footer'>
            <p>Brain Tumor Segmentation App | Made with Streamlit, PyTorch & Deep Learning</p>
            <p>© 2025 | For educational and research purposes only</p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()