# Image Forgery Detection Web Application

A web-based application for detecting whether images have been tampered with, using a deep learning model.

## Features

- **Image Forgery Detection**: Upload images to check if they've been tampered with
- **User Authentication**: Secure login and registration system
- **Analysis History**: View past image analysis results
- **Tampered Region Visualization**: For tampered images, see highlighted regions that have been modified

## Technologies Used

- **Backend**: Flask with PyMongo for MongoDB integration
- **Frontend**: HTML, CSS, JavaScript, Bootstrap 5
- **Database**: MongoDB for user management and history tracking
- **Deep Learning**: PyTorch for the forgery detection model
- **Image Processing**: OpenCV and PIL for image manipulation

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- MongoDB installed and running
- pip (Python package manager)

### Installation

1. Clone the repository or download the source code

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Make sure MongoDB is running on your system:
   ```
   # On Windows
   # Start MongoDB service if not already running
   net start MongoDB
   ```

4. Run the application:
   ```
   python app.py
   ```

5. Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

## Usage

1. Register a new account or login with existing credentials
2. On the home page, upload an image for forgery detection
3. View the analysis results showing whether the image is authentic or tampered
4. For tampered images, view the highlighted regions that have been modified
5. Access your analysis history from the History page

## Model Information

The application uses a deep learning model (`forgery_detection_model_120.pth`) that combines classification and segmentation to:

1. Determine if an image has been tampered with
2. Identify and highlight the specific regions that have been modified

The model architecture is a dual-task network with:
- A classification head to determine authenticity
- A segmentation head to localize tampered regions

## File Structure

```
image-app/
├── app.py                  # Main Flask application
├── model_architecture.py   # Neural network architecture
├── inference.py            # Functions for model inference
├── static/                 # Static files (CSS, JS)
├── templates/              # HTML templates
├── uploads/                # Uploaded images
├── results/                # Analysis results
└── forgery_detection_model_120.pth  # Pre-trained model
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
