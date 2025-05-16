# 🧠 Brain Tumor Segmentation using UNet (Streamlit App)

This project presents a web-based application built with Streamlit and PyTorch for the segmentation of brain tumors from MRI scans. It utilizes a UNet deep learning model to identify tumor regions and provides interactive visualizations, quantitative metrics, and AI-powered clinical interpretations using the Google Gemini API.

## ✨ Features

* **NIfTI File Upload**: Easily upload brain MRI scans in `.nii` or `.nii.gz` formats.
* **Interactive Slice Navigation**: Explore MRI volumes across Axial, Sagittal, and Coronal views with a dynamic slider.
* **UNet Deep Learning Model**: Automatic segmentation of tumor regions using a pre-trained UNet architecture.
* **Tumor Visualization**:
    * Original MRI slice display.
    * Tumor confidence maps to visualize prediction strength.
    * Overlay with tumor outline or custom colormaps (Hot, Viridis, Jet, Cool) for clear identification.
* **Quantitative Tumor Metrics**: Get instant statistics like tumor pixel count, percentage of slice, dimensions (width, height), approximate diameter, perimeter, and circularity.
* **AI Clinical Interpretation**: Leverage the Google Gemini API to generate concise, clinically-oriented explanations of the segmentation results, including potential diagnoses and next steps.
* **Multi-Slice Visualization**: View a series of adjacent slices to understand the 3D extent of the tumor.
* **Sample Data Option**: Test the application with synthetic brain MRI data if you don't have your own files.
* **Export Results**: Download the segmented tumor mask or the overlay image as PNG files.
* **Advanced Settings**: Adjust segmentation thresholds, image normalization methods, and apply post-processing like smoothing.

## 🚀 Getting Started

Follow these instructions to set up and run the application locally.

### Prerequisites

* Python 3.8+
* `pip` (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```
    (Replace `your-username/your-repo-name` with your actual GitHub repository path.)

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```
    If you don't have a `requirements.txt` file, you can create one with the following contents based on the provided `App.py`:
    ```
    streamlit>=1.0.0
    torch
    numpy
    matplotlib
    nibabel
    Pillow
    requests
    scipy # For smoothing, if used
    scikit-image # For contour finding
    google-generativeai
    ```

4.  **Set up Google Gemini API Key:**
    The provided code uses a placeholder API key. For actual use, it's highly recommended to use Streamlit's secrets management.But first you will need an API key
    
    ✅ How to Get Your Gemini (Generative AI) API Key
Step 1: Create or Select a Project
Click the Select a project button at the top. Then:
Either create a new project or select an existing one.
Give it a name like "GeminiApp".

Step 2: Enable the Gemini API
Once you're inside the project:

Click the top-left menu (☰) > APIs & Services > Library.
Search for:

Gemini API or

Generative Language API (under Vertex AI API or PaLM API)
Click it, then press Enable.

Step 3: Create Credentials (API Key)
Go to the left menu again:

APIs & Services > Credentials
Click + Create Credentials > API key

Your key will be generated. Copy and save it securely.
(Optional but recommended) Restrict the key to only the Gemini API for safety.

Step 4: Use the Key in Your App
Put it into your .env file like this:

env
Copy
Edit
GEMINI_API_KEY=your-generated-api-key
Or into your secrets.toml (if using Streamlit Secrets):

toml
Copy
Edit
[general]
GEMINI_API_KEY = "your-generated-api-key"

    * Create a `.streamlit` folder in your project's root directory.
    * Inside `.streamlit`, create a file named `secrets.toml`.
    * Add your Gemini API key to `secrets.toml`:
        ```toml
        GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"
        ```
    * Replace API key in `App.py` with `st.secrets["GEMINI_API_KEY"]` for the `api_key` variable within `initialize_gemini_api()`.

### Running the Application

Once you have everything set up, run the Streamlit app from your terminal:

```bash
streamlit run App.py
