import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import io
import os
import tempfile
from PIL import Image
import torch.nn as nn
import requests
from io import BytesIO
import logging
import time
from datetime import datetime

import google.generativeai as genai
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import streamlit as st
import time
import os


# Enhanced Gemini API initialization with better error handling and API key management
def initialize_gemini_api():
    """Initialize and configure the Gemini API client with robust error handling"""
    try:

        api_key = "AIzaSyCAE_UGFvjlMHdwpHgXWGkufPNOP4xR5O8"
        st.sidebar.warning("⚠️ Using default API key. For production, set your own key.", icon="⚠️")

        # Configure the Gemini client
        genai.configure(api_key=api_key)

        # Use newer model with better capabilities and fallback options
        try:
            model = genai.GenerativeModel('gemini-2.0-flash')
            # Test if model works with a simple prompt
            response = model.generate_content("Hello")
            return model
        except Exception as model_error:
            st.sidebar.warning(f"Could not initialize gemini-2.0-flash model: {str(model_error)}")
            # Fall back to gemini-pro if gemini-2.0-flash isn't available
            try:
                model = genai.GenerativeModel('gemini-pro')
                return model
            except Exception as fallback_error:
                st.sidebar.error(f"Fallback model also failed: {str(fallback_error)}")
                return None

    except Exception as e:
        st.sidebar.error(f"Failed to initialize Gemini API: {str(e)}")
        return None


def fig_to_base64(fig, dpi=120, quality=90):
    """
    Convert a matplotlib figure to a base64-encoded JPEG image string with quality control.

    Parameters:
    - fig: matplotlib figure
    - dpi: resolution
    - quality: JPEG quality (0-100)

    Returns:
    - Base64 encoded JPEG image string
    """
    # Save figure to PNG in memory
    png_buffer = BytesIO()
    fig.savefig(png_buffer, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    png_buffer.seek(0)

    # Open with PIL and convert to JPEG with quality settings
    image = Image.open(png_buffer).convert('RGB')
    jpeg_buffer = BytesIO()
    image.save(jpeg_buffer, format='JPEG', quality=quality, optimize=True)
    jpeg_buffer.seek(0)

    # Encode to base64
    image_data = base64.b64encode(jpeg_buffer.getvalue()).decode('utf-8')
    return image_data


# Enhanced function to calculate detailed tumor statistics
def calculate_detailed_tumor_metrics(segmentation_mask):
    """Calculate comprehensive tumor metrics from the segmentation mask"""
    import numpy as np
    from scipy import ndimage

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
    from skimage import measure
    contours = measure.find_contours(segmentation_mask, 0.5)
    perimeter = 0
    if contours:
        perimeter = sum(len(contour) for contour in contours)

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


# Improved function to get explanations from Gemini with retry logic and cached responses
@st.cache_data(ttl=3600, show_spinner=False)  # Cache for 1 hour
def get_gemini_explanation(_model, original_img, segmentation_mask, tumor_stats, view_orientation="Axial",
                           slice_info=""):
    """
    Get a detailed explanation of the tumor segmentation from Gemini with enhanced prompting

    Parameters:
    model: Gemini model instance
    original_img: Original MRI image
    segmentation_mask: Predicted segmentation mask
    tumor_stats: Dictionary containing detailed tumor statistics
    view_orientation: Orientation of the MRI slice (Axial, Sagittal, Coronal)
    slice_info: Additional information about the slice location

    Returns:
    str: Explanation from Gemini
    """
    if model is None:
        return "Gemini AI explanation unavailable. Please check API key configuration."

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
        from skimage import measure
        contours = measure.find_contours(segmentation_mask, 0.5)
        for contour in contours:
            ax2.plot(contour[:, 1], contour[:, 0], linewidth=2, color='red')

    ax2.set_title("Tumor Segmentation")
    ax2.axis('off')

    # Convert figures to base64 strings
    original_b64 = fig_to_base64(fig1)
    segmentation_b64 = fig_to_base64(fig2)

    # Close figures to free memory
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

    # Create enhanced prompt for Gemini with more relevant medical context
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
            # Add timeout handling
            start_time = time.time()
            response = _model.generate_content(prompt)  # 15 second timeout

            # Check if we have a valid response
            if hasattr(response, 'text') and response.text:
                # Add disclaimer for medical context
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page title and configuration
st.set_page_config(
    page_title="Brain Tumor Segmentation",
    layout="wide"
)


# Define UNet architecture
def double_convolution(in_channels, out_channels):
    """
    Double convolution block for UNet
    """
    conv_op = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
    return conv_op


class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        # Contracting path
        self.down_convolution_1 = double_convolution(1, 64)
        self.down_convolution_2 = double_convolution(64, 128)
        self.down_convolution_3 = double_convolution(128, 256)
        self.down_convolution_4 = double_convolution(256, 512)
        self.down_convolution_5 = double_convolution(512, 1024)

        # Expanding path
        self.up_transpose_1 = nn.ConvTranspose2d(
            in_channels=1024, out_channels=512,
            kernel_size=2,
            stride=2)
        self.up_convolution_1 = double_convolution(1024, 512)
        self.up_transpose_2 = nn.ConvTranspose2d(
            in_channels=512, out_channels=256,
            kernel_size=2,
            stride=2)
        self.up_convolution_2 = double_convolution(512, 256)
        self.up_transpose_3 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128,
            kernel_size=2,
            stride=2)
        self.up_convolution_3 = double_convolution(256, 128)
        self.up_transpose_4 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64,
            kernel_size=2,
            stride=2)
        self.up_convolution_4 = double_convolution(128, 64)
        # output
        self.out = nn.Conv2d(
            in_channels=64, out_channels=num_classes,
            kernel_size=1
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

        out = self.out(up_8)

        return out


# Load model with better error handling
@st.cache_resource
def load_model():
    model = UNet(num_classes=1)

    # Try to download checkpoint if not exists
    checkpoint_path = 'checkpoint-epoch-29.pt'
    if not os.path.exists(checkpoint_path):
        checkpoint_url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/HATpl02lA0ykn9aAU9K6sA/checkpoint-epoch-29.pt'

        with st.spinner('Downloading model checkpoint... This may take a moment.'):
            try:
                response = requests.get(checkpoint_url)
                response.raise_for_status()  # Raise an exception for bad responses

                # Save the downloaded file locally
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


# Improved NIfTI file processing
def process_nifti_file(uploaded_file):
    """Process NIfTI files with better error handling"""
    temp_file_path = None
    try:
        # Save the uploaded file content to a temporary file
        temp_dir = tempfile.gettempdir()
        temp_file_fd, temp_file_path = tempfile.mkstemp(suffix='.nii.gz')

        # Close the file descriptor and write the content
        os.close(temp_file_fd)
        with open(temp_file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        # Log the path for debugging
        logger.info(f"Saved temporary file to: {temp_file_path}")

        try:
            # Try loading as .nii.gz
            nifti_img = nib.load(temp_file_path)
        except Exception as e:
            logger.error(f"Failed to load as .nii.gz: {str(e)}")
            # If that fails, try loading as regular NIfTI
            # Rename and try again
            new_path = temp_file_path.replace('.nii.gz', '.nii')
            os.rename(temp_file_path, new_path)
            temp_file_path = new_path
            logger.info(f"Renamed to: {temp_file_path}")
            nifti_img = nib.load(temp_file_path)

        # Log success
        logger.info(f"Successfully loaded NIfTI file: {nifti_img.shape}")

        # Get data ready for return but don't remove the file yet
        return nifti_img

    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise e
    # finally:
    #     # Clean up in finally block to ensure it happens even if there's an exception
    #     try:
    #         if temp_file_path and os.path.exists(temp_file_path):
    #             os.remove(temp_file_path)
    #             logger.info(f"Removed temporary file: {temp_file_path}")
    #     except Exception as cleanup_error:
    #         logger.error(f"Failed to clean up temporary file: {str(cleanup_error)}")


def preprocess_image(nifti_img, slice_num, normalize_method='max'):
    """
    Preprocess image for model input with multiple normalization options
    """
    data = nifti_img.get_fdata()

    # Get the slice data
    slice_data = data[:, :, slice_num]

    # Normalize the slice based on method
    if normalize_method == 'max':
        if np.max(slice_data) > 0:
            normalized_slice = slice_data / np.max(slice_data)
        else:
            normalized_slice = slice_data
    elif normalize_method == 'z-score':
        mean = np.mean(slice_data)
        std = np.std(slice_data)
        if std > 0:
            normalized_slice = (slice_data - mean) / std
        else:
            normalized_slice = slice_data - mean
    elif normalize_method == 'min-max':
        min_val = np.min(slice_data)
        max_val = np.max(slice_data)
        if max_val > min_val:
            normalized_slice = (slice_data - min_val) / (max_val - min_val)
        else:
            normalized_slice = slice_data
    else:
        normalized_slice = slice_data  # No normalization

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


def calculate_tumor_metrics(segmentation):
    """Calculate tumor size and other metrics"""
    pixel_count = np.sum(segmentation > 0)
    total_pixels = segmentation.shape[0] * segmentation.shape[1]
    percentage = (pixel_count / total_pixels) * 100

    # Find center of mass if tumor is present
    if pixel_count > 0:
        y_indices, x_indices = np.where(segmentation > 0)
        center_y = np.mean(y_indices)
        center_x = np.mean(x_indices)
        center = (int(center_x), int(center_y))
    else:
        center = None

    return {
        "pixel_count": pixel_count,
        "percentage": percentage,
        "center": center
    }


def apply_custom_colormap(segmentation, raw_prediction, colormap_name='hot'):
    """Apply a custom colormap to the segmentation with alpha based on confidence"""
    # Create RGBA data using the raw prediction values as alpha
    cmap = plt.get_cmap(colormap_name)
    colored_segmentation = cmap(raw_prediction)

    # Adjust the alpha channel based on prediction strength
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


# Function to create sample brain data for testing
def create_sample_brain_data():
    """Create synthetic brain MRI data for testing"""
    try:
        # Create a 3D array to represent brain volume (256x256x128)
        # This is just for demonstration - not realistic brain data
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
        background-color: #f0f7ff;
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

# App header
st.markdown("<h1 class='main-title'>Brain Tumor Segmentation using UNet</h1>", unsafe_allow_html=True)

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
    st.image("https://i.imgur.com/wKYuNtM.png", use_column_width=True)
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
    # In the sidebar Display Options section
    show_ai_explanation = st.checkbox("Show AI Clinical Explanation",
                                      value=True,
                                      help="Get AI-powered clinical interpretation using Google Gemini")
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

# Load model
with st.spinner("Initializing model..."):
    try:
        model, device = load_model()
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        st.stop()
# Initialize Gemini API
with st.spinner("Initializing AI explanation capabilities..."):
    gemini_model = initialize_gemini_api()
    if gemini_model:
        st.success("✅ Gemini AI successfully initialized!")
    else:
        st.warning("⚠️ Gemini AI could not be initialized. Explanations will be unavailable.")
# File uploader
st.markdown("<h2 class='subtitle'>Upload MRI Scan</h2>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose a NIfTI file (.nii or .nii.gz)",
    type=["nii", "nii.gz"],
    help="Upload a NIfTI format brain MRI scan"
)

# Sample data option
use_sample = st.checkbox("Use sample data instead", value=False)

if use_sample:
    st.info("Using synthetic sample brain MRI data...")

    try:
        with st.spinner("Creating sample brain data..."):
            # Create synthetic brain data instead of downloading
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
        # Process the uploaded NIfTI file
        with st.spinner("Processing uploaded file..."):
            nifti_img = process_nifti_file(uploaded_file)
            st.success("✅ File processed successfully!")
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.info("Please ensure you're uploading a valid NIfTI (.nii or .nii.gz) file.")
        st.stop()
else:
    # No file uploaded yet
    st.info("Please upload a NIfTI file or use the sample data to continue.")
    st.stop()

# Get image dimensions
img_shape = nifti_img.get_fdata().shape
orientation = nifti_img.affine

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
if selected_view == "Axial (Top-Down)":
    max_slice = img_shape[2] - 1
    slice_dim = 2
    slice_name = "Z-axis"
elif selected_view == "Sagittal (Side)":
    max_slice = img_shape[0] - 1
    slice_dim = 0
    slice_name = "X-axis"
else:  # Coronal
    max_slice = img_shape[1] - 1
    slice_dim = 1
    slice_name = "Y-axis"

# Allow user to select slice number
slice_num = st.slider(f"Select slice number ({slice_name})",
                      0, max_slice, max_slice // 2,
                      help="Move the slider to navigate through different slices of the brain")

# Process the selected slice
with st.spinner("Analyzing slice..."):
    try:
        # Extract the appropriate slice based on view
        data = nifti_img.get_fdata()

        if slice_dim == 0:  # Sagittal
            slice_data = data[slice_num, :, :]
            slice_data = np.rot90(slice_data)  # Rotate for proper visualization
        elif slice_dim == 1:  # Coronal
            slice_data = data[:, slice_num, :]
            slice_data = np.rot90(slice_data)  # Rotate for proper visualization
        else:  # Axial (default)
            slice_data = data[:, :, slice_num]

        # Normalize the slice based on selected method
        if normalization == "max":
            normalized_slice = slice_data / np.max(slice_data) if np.max(slice_data) > 0 else slice_data
        elif normalization == "z-score":
            mean = np.mean(slice_data)
            std = np.std(slice_data)
            normalized_slice = (slice_data - mean) / std if std > 0 else slice_data - mean
        elif normalization == "min-max":
            min_val = np.min(slice_data)
            max_val = np.max(slice_data)
            normalized_slice = (slice_data - min_val) / (max_val - min_val) if max_val > min_val else slice_data
        else:  # No normalization
            normalized_slice = slice_data

        # Convert to tensor for model
        tensor_slice = torch.from_numpy(normalized_slice).float().unsqueeze(0).unsqueeze(0)

        # Make prediction
        binary_segmentation, raw_prediction, inference_time = predict_segmentation(model, device, tensor_slice,
                                                                                   threshold)

        if binary_segmentation is None:
            st.error("Failed to generate segmentation for this slice.")
            st.stop()

        # Calculate tumor metrics
        if show_metrics:
            # Replace your existing metrics calculation with this enhanced version
            detailed_metrics = calculate_detailed_tumor_metrics(binary_segmentation)
    except Exception as e:
        st.error(f"Error processing slice: {str(e)}")
        st.stop()

# Display results
st.markdown("<h2 class='subtitle'>Segmentation Results</h2>", unsafe_allow_html=True)

# Create colorful segmentation overlay
colored_segmentation = apply_custom_colormap(binary_segmentation, raw_prediction, overlay_style)
#metrics = detailed_metrics
# Add tumor metrics if requested
# Add AI explanation section
if show_metrics and show_ai_explanation:
    st.markdown("<h3>🧠 AI Clinical Interpretation</h3>", unsafe_allow_html=True)

    # Create a container with custom styling
    explanation_container = st.container()
    explanation_container.markdown("""
    <div style="background-color:#f8f9fa; padding:15px; border-radius:5px; border-left:4px solid #1E88E5;">
    """, unsafe_allow_html=True)

    with explanation_container:
        with st.spinner("Generating clinical interpretation..."):
            # Get the slice information
            slice_info = f"Slice {slice_num}/{max_slice} ({slice_num / max_slice * 100:.0f}% from bottom)"

            # Get explanation from Gemini
            explanation = get_gemini_explanation(
                gemini_model,
                normalized_slice,
                binary_segmentation,
                detailed_metrics,
                view_orientation=selected_view,
                slice_info=slice_info
            )

            # Display the explanation
            st.markdown(explanation)

    explanation_container.markdown("</div>", unsafe_allow_html=True)
elif show_metrics:
    st.info("No tumor detected in this slice.")

# Display visualizations
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<p class='result-header'>Original MRI Slice</p>", unsafe_allow_html=True)
    fig1, ax1 = plt.subplots(figsize=(5, 5))
    ax1.imshow(normalized_slice, cmap='gray')
    ax1.set_title("Original MRI")
    ax1.axis('off')
    st.pyplot(fig1)

with col2:
    if show_confidence:
        st.markdown("<p class='result-header'>Tumor Confidence Map</p>", unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        confidence_map = ax2.imshow(raw_prediction, cmap=overlay_style, vmin=0, vmax=1)
        ax2.set_title("Segmentation Confidence")
        ax2.axis('off')
        cbar = plt.colorbar(confidence_map, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label('Confidence')
        st.pyplot(fig2)
    else:
        st.markdown("<p class='result-header'>Tumor Segmentation</p>", unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        ax2.imshow(binary_segmentation, cmap=overlay_style)
        ax2.set_title("Segmentation Mask")
        ax2.axis('off')
        st.pyplot(fig2)

with col3:
    st.markdown("<p class='result-header'>Segmentation Overlay</p>", unsafe_allow_html=True)
    fig3, ax3 = plt.subplots(figsize=(5, 5))

    if show_outline:
        plot_with_tumor_outline(ax3, normalized_slice, binary_segmentation, "Tumor Outline")
    else:
        ax3.imshow(normalized_slice, cmap='gray')
        ax3.imshow(colored_segmentation)
        ax3.set_title("Segmentation Overlay")
        ax3.axis('off')

    st.pyplot(fig3)

# Multi-slice visualization
if st.checkbox("Show multi-slice visualization", value=False):
    st.subheader("Multi-Slice Visualization")

    # Get number of slices to show
    num_slices = st.slider("Number of slices to visualize", 3, 9, 5)

    # Calculate slice indices to show
    current_slice_index = slice_num
    max_slice_index = max_slice

    # Calculate offsets
    half_range = (num_slices - 1) // 2
    start_slice = max(0, current_slice_index - half_range)
    end_slice = min(max_slice_index, current_slice_index + half_range)

    # Adjust start and end if needed
    if start_slice == 0:
        end_slice = min(max_slice_index, num_slices - 1)
    elif end_slice == max_slice_index:
        start_slice = max(0, max_slice_index - num_slices + 1)

    # Get slices to display
    slices_to_show = range(start_slice, end_slice + 1)

    # Create multi-slice figure
    fig, axs = plt.subplots(1, len(slices_to_show), figsize=(15, 4))

    # If only one slice, convert axs to a list
    if len(slices_to_show) == 1:
        axs = [axs]

    # Process each slice
    for i, s in enumerate(slices_to_show):
        # Get data for the slice
        if slice_dim == 0:  # Sagittal
            slice_data = data[s, :, :]
            slice_data = np.rot90(slice_data)  # Rotate for proper visualization
        elif slice_dim == 1:  # Coronal
            slice_data = data[:, s, :]
            slice_data = np.rot90(slice_data)  # Rotate for proper visualization
        else:  # Axial (default)
            slice_data = data[:, :, s]

        # Normalize the slice
        if np.max(slice_data) > 0:
            normalized = slice_data / np.max(slice_data)
        else:
            normalized = slice_data

        # Convert to tensor for model
        tensor = torch.from_numpy(normalized).float().unsqueeze(0).unsqueeze(0)

        # Predict segmentation
        binary_seg, raw_pred, _ = predict_segmentation(model, device, tensor, threshold)

        if binary_seg is None:
            axs[i].imshow(normalized, cmap='gray')
            axs[i].set_title(f"Slice {s}")
            axs[i].axis('off')
            continue

        # Show the result
        axs[i].imshow(normalized, cmap='gray')

        if show_outline:
            # Draw contour around tumor
            if np.sum(binary_seg) > 0:
                axs[i].contour(binary_seg, levels=[0.5], colors='r', linewidths=1)
        else:
            # Apply overlay with colored segmentation
            colored_seg = apply_custom_colormap(binary_seg, raw_pred, overlay_style)
            axs[i].imshow(colored_seg)

        # Highlight current slice
        if s == current_slice_index:
            axs[i].set_title(f"Slice {s} (Selected)", color='red')
        else:
            axs[i].set_title(f"Slice {s}")

        axs[i].axis('off')

    plt.tight_layout()
    st.pyplot(fig)

# Download options
st.markdown("<h2 class='subtitle'>Export Results</h2>", unsafe_allow_html=True)

export_col1, export_col2 = st.columns(2)

with export_col1:
    if st.button("📥 Download Segmentation Mask"):
        try:
            # Convert segmentation to image format
            segmentation_img = Image.fromarray((binary_segmentation * 255).astype(np.uint8))

            # Create a BytesIO object to store the image
            buf = io.BytesIO()
            segmentation_img.save(buf, format="PNG")

            # Create a download button
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label="Download Segmentation PNG",
                data=buf.getvalue(),
                file_name=f"segmentation_slice{slice_num}_{timestamp}.png",
                mime="image/png",
            )
        except Exception as e:
            st.error(f"Error during download: {str(e)}")

with export_col2:
    if st.button("📥 Download Overlay Image"):
        try:
            # Create figure with overlay
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(normalized_slice, cmap='gray')

            if show_outline:
                if np.sum(binary_segmentation) > 0:
                    ax.contour(binary_segmentation, levels=[0.5], colors='r', linewidths=2)
            else:
                ax.imshow(colored_segmentation)

            ax.set_title("Tumor Segmentation Overlay")
            ax.axis('off')

            # Save figure to BytesIO
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
            plt.close(fig)

            # Create a download button
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label="Download Overlay PNG",
                data=buf.getvalue(),
                file_name=f"overlay_slice{slice_num}_{timestamp}.png",
                mime="image/png",
            )
        except Exception as e:
            st.error(f"Error during download: {str(e)}")

# Add advanced options expander
with st.expander("⚙️ Advanced Settings", expanded=False):
    st.subheader("Model Configuration")
    st.write("These settings are for advanced users who want to fine-tune the segmentation process.")

    # Additional parameters
    confidence_threshold = st.slider(
        "Minimum Confidence Display",
        0.0, 1.0, 0.2, 0.05,
        help="Only show predictions above this confidence level"
    )

    # Post-processing options
    st.subheader("Post-processing")
    apply_smoothing = st.checkbox("Apply Smoothing", value=False)

    if apply_smoothing:
        smoothing_sigma = st.slider(
            "Smoothing Sigma",
            0.5, 5.0, 1.0, 0.1,
            help="Higher values create smoother boundaries"
        )

        # Apply smoothing if requested
        try:
            from scipy import ndimage

            # Apply Gaussian filter for smoothing
            smoothed_prediction = ndimage.gaussian_filter(raw_prediction, sigma=smoothing_sigma)

            # Apply threshold to the smoothed prediction
            smoothed_binary = (smoothed_prediction > threshold).astype(float)

            # Show comparison
            smooth_col1, smooth_col2 = st.columns(2)

            with smooth_col1:
                fig_orig, ax_orig = plt.subplots(figsize=(5, 5))
                ax_orig.imshow(binary_segmentation, cmap='gray')
                ax_orig.set_title("Original Segmentation")
                ax_orig.axis('off')
                st.pyplot(fig_orig)

            with smooth_col2:
                fig_smooth, ax_smooth = plt.subplots(figsize=(5, 5))
                ax_smooth.imshow(smoothed_binary, cmap='gray')
                ax_smooth.set_title("Smoothed Segmentation")
                ax_smooth.axis('off')
                st.pyplot(fig_smooth)

        except ImportError:
            st.error("Scipy is required for smoothing operations.")

# Add footer with information
st.markdown("""
        <div class="footer">
            <p>Brain Tumor Segmentation Application v1.2.1</p>
            <p>Based on UNet architecture trained on medical imaging data</p>
            <p>© 2025 | Developed with Streamlit, PyTorch & NiBabel</p>
        </div>
        """, unsafe_allow_html=True)