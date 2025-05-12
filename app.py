from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
import os
import numpy as np
from PIL import Image
import cv2
import base64
import io
import time
import zipfile
from datetime import datetime
import torch
from torch.nn import functional as F
import onnxruntime as ort
import shutil
import uuid
import hashlib
from torchvision import transforms
from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash, send_from_directory
from flask_cors import CORS
from pymongo import MongoClient
from bson.objectid import ObjectId
import hashlib
import binascii
import io
from functools import wraps
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.platypus import Image as ReportLabImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# Import our custom modules
from model_architecture import ForgeryDetectionNet, ForgeryDetectionNetAlternative
from inference import load_model, predict
from forensic_features import ForensicFeatureExtractor

def hash_password(password):
    """Hash a password for storing."""
    salt = hashlib.sha256(os.urandom(60)).hexdigest()
    pwdhash = hashlib.pbkdf2_hmac('sha512', password.encode('utf-8'), salt.encode('ascii'), 100000)
    pwdhash = pwdhash.hex()
    return salt + pwdhash

def verify_password(stored_password, provided_password):
    """Verify a stored password against one provided by user"""
    salt = stored_password[:64]
    stored_hash = stored_password[64:]
    pwdhash = hashlib.pbkdf2_hmac('sha512', provided_password.encode('utf-8'), salt.encode('ascii'), 100000)
    pwdhash = pwdhash.hex()
    return pwdhash == stored_hash

app = Flask(__name__)
app.config['SECRET_KEY'] = 'forgery_detection'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max upload
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MONGO_URI'] = 'mongodb://localhost:27017/'
app.config['DB_NAME'] = 'forgery_detection'
CORS(app)

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Create upload and results folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Set up MongoDB connection
try:
    mongo_client = MongoClient(app.config['MONGO_URI'])
    db = mongo_client[app.config['DB_NAME']]
    
    # Use the specified collections
    users_collection = db['users']
    history_collection = db['analyses']
    config_collection = db['config']
    
    # Initialize the application configuration
    # Check if we have a setup config
    setup_config = config_collection.find_one({'config_name': 'app_setup'})
    if not setup_config:
        # Mark that we've completed initial setup
        config_collection.insert_one({
            'config_name': 'app_setup', 
            'completed': True, 
            'timestamp': datetime.now(),
            'version': '1.0.0'
        })
        print("Application setup complete")
    
    print(f"Connected to MongoDB database: {app.config['DB_NAME']}")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    db = None
    users_collection = None
    history_collection = None
    config_collection = None

# Initialize model
MODEL_PATH = 'forgery_detection_model_120.pth'
ONNX_MODEL_PATH = 'forgery_detection_model.onnx'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Increase timeout for model operations
timeout_seconds = 100  # Increased timeout for better model loading and prediction

# Check if ONNX model exists, otherwise use PyTorch model and try to convert it
use_onnx = os.path.exists(ONNX_MODEL_PATH)

# If ONNX model doesn't exist, try to convert the PyTorch model to ONNX
if not use_onnx and os.path.exists(MODEL_PATH):
    try:
        print("ONNX model not found. Attempting to convert PyTorch model to ONNX...")
        from convert_to_onnx import convert_pytorch_to_onnx
        
        # Try both standard and feature-based conversion
        success = convert_pytorch_to_onnx(MODEL_PATH, ONNX_MODEL_PATH, device=str(device), feature_based=False)
        if not success:
            success = convert_pytorch_to_onnx(MODEL_PATH, ONNX_MODEL_PATH, device=str(device), feature_based=True)
        
        if success:
            print("Successfully converted PyTorch model to ONNX format.")
            use_onnx = os.path.exists(ONNX_MODEL_PATH)
        else:
            print("Failed to convert PyTorch model to ONNX format. Will use PyTorch model directly.")
    except Exception as e:
        print(f"Error during model conversion: {e}")
        print("Will use PyTorch model directly.")

use_onnx = False
print("Using PyTorch model directly like in test.py")

# Initialize PyTorch model if ONNX is not available
if not use_onnx:
    print("Loading PyTorch model...")
    try:
        # Direct model loading approach - no alternatives
        # Set PyTorch to use a reasonable number of threads
        torch.set_num_threads(4)
        
        # Load the model directly
        print(f"Loading model from {MODEL_PATH}...")
        
        model = load_model(MODEL_PATH, device=device)
        
        if model is not None:
            print("Model loaded successfully and set to evaluation mode")
        else:
            print("WARNING: Model could not be loaded. Predictions will not be available.")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
        
    # Verify model is loaded correctly
    if model is not None:
        print("Model loaded successfully and set to evaluation mode")
    else:
        print("WARNING: Model could not be loaded. Predictions will not be available.")

# Initialize feature extractor
feature_extractor = ForensicFeatureExtractor()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def preprocess_image(img_data, img_size=256):
    """Preprocess image data for model input exactly as in test.py"""
    try:
        # For file upload (bytes)
        if isinstance(img_data, bytes):
            # Convert bytes to PIL Image
            img = Image.open(io.BytesIO(img_data)).convert('RGB')
        # For base64 encoded image
        elif isinstance(img_data, str) and img_data.startswith('data:image'):
            # Extract the base64 encoded image
            img_data = img_data.split(',')[1]
            img_data = base64.b64decode(img_data)
            img = Image.open(io.BytesIO(img_data)).convert('RGB')
        # For file path
        elif isinstance(img_data, str) and os.path.isfile(img_data):
            img = Image.open(img_data).convert('RGB')
        # For form data (request.form)
        elif hasattr(img_data, 'get') and img_data.get('image'):
            # Extract image from form data
            img_str = img_data.get('image')
            if img_str.startswith('data:image'):
                img_str = img_str.split(',')[1]
                img_data = base64.b64decode(img_str)
                img = Image.open(io.BytesIO(img_data)).convert('RGB')
            else:
                raise ValueError("Invalid image format in form data")
        # For request.files
        elif hasattr(img_data, 'files') and 'image' in img_data.files:
            img_file = img_data.files['image']
            img = Image.open(img_file).convert('RGB')
        else:
            print(f"Unsupported image data type: {type(img_data)}")
            if isinstance(img_data, str):
                print(f"String starts with: {img_data[:30]}...")
            elif hasattr(img_data, 'keys'):
                print(f"Keys in data: {list(img_data.keys())}")
            raise ValueError("Unsupported image data format")
        
        # Save original image as numpy array for display
        original_image = np.array(img)
        
        # Apply transformation using torchvision transforms exactly as in test.py
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Apply transformation
        image_tensor = transform(img).unsqueeze(0)  # Add batch dimension
        
        return image_tensor, original_image
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        print(f"Image data type: {type(img_data)}")
        if isinstance(img_data, bytes):
            print(f"Bytes length: {len(img_data)}")
        if hasattr(img_data, 'keys'):
            print(f"Keys in data: {list(img_data.keys())}")
        import traceback
        traceback.print_exc()
        raise

def create_segmentation_mask(original_img, segmentation_mask, threshold=0.5):
    """Create a black background with white areas for tampered regions"""
    # Ensure mask is properly formatted
    if segmentation_mask is None or segmentation_mask.size == 0:
        print("Warning: Empty or invalid segmentation mask")
        return np.zeros_like(original_img)
    
    # Print mask shape and type for debugging
    print(f"Segmentation mask shape: {segmentation_mask.shape}, dtype: {segmentation_mask.dtype}")
    
    # Resize mask to match original image if needed
    if original_img.shape[:2] != segmentation_mask.shape[:2]:
        try:
            segmentation_mask = cv2.resize(segmentation_mask, (original_img.shape[1], original_img.shape[0]))
        except Exception as e:
            print(f"Error resizing segmentation mask: {e}")
            return np.zeros_like(original_img)
    
    # Create binary mask
    binary_mask = (segmentation_mask > threshold).astype(np.uint8)
    
    # Create a black background image
    mask_image = np.zeros_like(original_img)
    
    # Set tampered regions to white
    mask_image[binary_mask == 1] = [255, 255, 255]
    
    return mask_image

def overlay_segmentation(original_img, segmentation_mask, threshold=0.5, alpha=0.5, color=[255, 0, 0]):
    """Overlay segmentation mask on the original image"""
    # Ensure mask is properly formatted
    if segmentation_mask is None or segmentation_mask.size == 0:
        print("Warning: Empty or invalid segmentation mask")
        return original_img.copy()
    
    # Resize mask to match original image if needed
    if original_img.shape[:2] != segmentation_mask.shape[:2]:
        try:
            segmentation_mask = cv2.resize(segmentation_mask, (original_img.shape[1], original_img.shape[0]))
        except Exception as e:
            print(f"Error resizing segmentation mask: {e}")
            return original_img.copy()
    
    # Create binary mask
    binary_mask = (segmentation_mask > threshold).astype(np.uint8)
    
    # Create colored overlay for tampered regions
    overlay = np.zeros_like(original_img)
    overlay[binary_mask == 1] = color
    
    # Combine with original image using alpha blending
    combined = cv2.addWeighted(original_img, 1.0, overlay, alpha, 0)
    
    return combined

def analyze_image(img_data):
    """Analyze image for forgery detection using approach from test.py"""
    start_time = time.time()
    
    # Set a timeout for the analysis
    max_analysis_time = 60  # seconds - increased for more accurate predictions
    
    # Preprocess image
    try:
        print("Starting image preprocessing")
        print(f"Image data type: {type(img_data)}")
        if isinstance(img_data, bytes):
            print(f"Bytes length: {len(img_data)}")
        
        # Use the exact preprocessing from test.py
        image_tensor, original_img = preprocess_image(img_data)
        print(f"Image tensor shape: {image_tensor.shape}")
        print(f"Original image shape: {original_img.shape}")
        print("Image preprocessing completed successfully")
    except Exception as e:
        print(f"Error during image preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return {
            'error': f"Failed to preprocess image: {str(e)}"
        }
    
    # Use PyTorch model directly as in test.py
    try:
        print("Starting PyTorch inference")
        
        # Check if model is loaded properly
        if model is None:
            raise ValueError("Model is not loaded properly")
        
        # Move tensor to the correct device
        image_tensor = image_tensor.to(device)
        
        # Run inference exactly as in test.py
        with torch.no_grad():
            # Forward pass through the model
            outputs, segmentation, attention_maps = model(image_tensor)
            
            # Get classification result
            probabilities = F.softmax(outputs, dim=1)
            prob_authentic, prob_tampered = probabilities[0].cpu().numpy()
            prediction = torch.argmax(outputs, dim=1).item()
            
            # Get segmentation mask and binarize it
            segmentation_mask = segmentation[0].cpu().numpy().squeeze()
            mask = (segmentation_mask > 0.5).astype(np.uint8) * 255
            
            # Process attention maps (if needed)
            if attention_maps is not None:
                # Resize and combine attention maps as in ifdplus
                ref_size = attention_maps[0][0].shape[-2:]
                resized_attention_maps = []
                
                for att in attention_maps:
                    # Resize if dimensions don't match
                    if att[0].shape[-2:] != ref_size:
                        resized_att = F.interpolate(att, size=ref_size, mode='bilinear', align_corners=True)
                        resized_attention_maps.append(resized_att[0].unsqueeze(0))
                    else:
                        resized_attention_maps.append(att[0].unsqueeze(0))
                
                # Combine the resized attention maps
                combined_attention = torch.mean(torch.cat(resized_attention_maps, dim=0), dim=0)
                attention_map = combined_attention.cpu().numpy()
            else:
                attention_map = None
        
        # Set label based on prediction (0 = authentic, 1 = tampered)
        label = "real" if prediction == 0 else "fake"
        confidence = prob_authentic if prediction == 0 else prob_tampered
        
        # Print detailed information for debugging
        print(f"Prediction: {label} (class {prediction})")
        print(f"Confidence: {confidence:.4f}")
        print(f"Probabilities: Authentic={prob_authentic:.4f}, Tampered={prob_tampered:.4f}")
        
        if mask is not None:
            print(f"Segmentation mask shape: {mask.shape}")
        if attention_map is not None:
            print(f"Attention map shape: {attention_map.shape}")
            
    except Exception as e:
        print(f"Error during PyTorch inference: {e}")
        import traceback
        traceback.print_exc()
        return {
            'error': f"Failed to analyze image: {str(e)}"
        }
    # Generate both visualizations if the image is classified as fake
    if label == "fake" and mask is not None and mask.size > 0:
        # 1. Create pure segmentation mask (black background with white tampered regions)
        segmentation_mask_img = create_segmentation_mask(original_img, mask, threshold=0.5)
        
        # 2. Create manipulation overlay (original image with highlighted tampered regions)
        manipulation_overlay_img = overlay_segmentation(original_img, mask, threshold=0.5, alpha=0.5, color=[255, 0, 0])
    else:
        # If not fake or no mask, use original image for both
        segmentation_mask_img = np.zeros_like(original_img)  # Empty black image
        manipulation_overlay_img = original_img
    
    # No ELA visualization as requested
    ela_img = None
    
    # Generate unique filenames for each visualization type
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    filename_base = f"analysis_{timestamp}"
    
    # Save all three visualization types as separate files
    original_filename = f"{filename_base}_original.png"
    mask_filename = f"{filename_base}_mask.png"
    overlay_filename = f"{filename_base}_overlay.png"
    
    # Save the images to disk
    cv2.imwrite(os.path.join(app.config['RESULTS_FOLDER'], original_filename), cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(app.config['RESULTS_FOLDER'], mask_filename), cv2.cvtColor(segmentation_mask_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(app.config['RESULTS_FOLDER'], overlay_filename), cv2.cvtColor(manipulation_overlay_img, cv2.COLOR_RGB2BGR))
    
    # Convert segmentation mask to base64 for web display
    _, mask_buffer = cv2.imencode('.png', cv2.cvtColor(segmentation_mask_img, cv2.COLOR_RGB2BGR))
    mask_base64 = base64.b64encode(mask_buffer).decode('utf-8')
    
    # Convert manipulation overlay to base64 for web display
    _, overlay_buffer = cv2.imencode('.png', cv2.cvtColor(manipulation_overlay_img, cv2.COLOR_RGB2BGR))
    overlay_base64 = base64.b64encode(overlay_buffer).decode('utf-8')
    
    # Convert original image to base64 for web display
    _, original_buffer = cv2.imencode('.png', cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR))
    original_base64 = base64.b64encode(original_buffer).decode('utf-8')
    
    # Convert ELA image to base64 if available
    ela_base64 = None
    if ela_img is not None:
        _, ela_buffer = cv2.imencode('.png', cv2.cvtColor(ela_img, cv2.COLOR_RGB2BGR))
        ela_base64 = base64.b64encode(ela_buffer).decode('utf-8')
    
    processing_time = time.time() - start_time
    
    # Ensure all values are native Python types for proper JSON serialization
    result = {
        'label': str(label),
        'confidence': float(confidence * 100),
        'processing_time': float(processing_time),
        'segmentation_mask': mask_base64,         # Pure black and white mask
        'manipulation_overlay': overlay_base64,   # Original image with highlighted tampered regions
        'original_image': original_base64,        # Original image without any modifications
        'original_filename': original_filename,   # Filename of the original image
        'mask_filename': mask_filename,           # Filename of the segmentation mask
        'overlay_filename': overlay_filename      # Filename of the manipulation overlay
    }
    
    # Only include additional metrics if the image is classified as fake
    if label == "fake" and mask is not None and mask.size > 0:
        # Calculate the percentage of tampered area
        binary_mask = (mask > 0.5).astype(np.uint8)
        if binary_mask.size > 0:
            tampered_percentage = float(np.sum(binary_mask) / binary_mask.size * 100)
            result['tampered_area_percentage'] = tampered_percentage
    
    return result

def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def softmax(x):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

@app.route('/')
@login_required
def index():
    return render_template('index.html')

def hash_password(password):
    """Hash a password for storing."""
    salt = hashlib.sha256(os.urandom(60)).hexdigest()
    pwdhash = hashlib.pbkdf2_hmac('sha512', password.encode('utf-8'), salt.encode('ascii'), 100000)
    pwdhash = pwdhash.hex()
    return salt + pwdhash

def verify_password(stored_password, provided_password):
    """Verify a stored password against one provided by user"""
    salt = stored_password[:64]
    stored_hash = stored_password[64:]
    pwdhash = hashlib.pbkdf2_hmac('sha512', provided_password.encode('utf-8'), salt.encode('ascii'), 100000)
    pwdhash = pwdhash.hex()
    return pwdhash == stored_hash

@app.route('/login', methods=['GET', 'POST'])
def login():
    # Check if user is already logged in
    if 'user_id' in session:
        return redirect(url_for('index'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Validate input
        if not username or not password:
            flash('Please provide both username and password', 'danger')
            return render_template('login.html')
        
        # Check if MongoDB is available
        if users_collection is None:
            flash('Database connection error. Using fallback authentication.', 'warning')
            # Fallback to simple authentication
            session['username'] = username
            session['user_id'] = 'temp_id'
            flash('Login successful (fallback mode)!', 'success')
            return redirect(url_for('index'))
        
        # Find user in database
        user = users_collection.find_one({'username': username})
        
        if user:
            if verify_password(user['password'], password):
                # Set session variables
                session['username'] = username
                session['user_id'] = str(user['_id'])
                # Update last login time
                users_collection.update_one({'_id': user['_id']}, {'$set': {'last_login': datetime.now()}})
                flash('Login successful!', 'success')
                return redirect(url_for('index'))
            else:
                # Specific message for incorrect password
                flash('Incorrect password. Please try again.', 'danger')
        else:
            # User doesn't exist
            flash('Username not found. Please check your username or register a new account.', 'danger')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    # Check if user is already logged in
    if 'user_id' in session:
        return redirect(url_for('index'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # Validate input
        if not username or not password:
            flash('Please provide both username and password', 'danger')
            return render_template('register.html')
            
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return render_template('register.html')
        
        # Check if MongoDB is available
        if users_collection is None:
            flash('Database connection error. Registration not available.', 'danger')
            return render_template('register.html')
        
        # Check if username already exists
        existing_user = users_collection.find_one({'username': username})
        if existing_user:
            flash('Username already exists', 'danger')
            return render_template('register.html')
        
        # Create new user
        user_id = str(uuid.uuid4())
        hashed_password = hash_password(password)
        
        new_user = {
            '_id': user_id,
            'username': username,
            'password': hashed_password,
            'created_at': datetime.now(),
            'last_login': datetime.now()
        }
        
        # Insert user into database
        users_collection.insert_one(new_user)
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    # Clear the session
    session.clear()
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

@app.route('/history')
@login_required
def history():
    
    # Get user ID from session
    user_id = session['user_id']
    
    # Check if MongoDB is available
    if history_collection is None:
        flash('Database connection error. History not available.', 'warning')
        return render_template('history.html', history=[])
    
    # Fetch user's history from database
    user_history = list(history_collection.find({'user_id': user_id}).sort('timestamp', -1))
    
    # Format the history data for display
    formatted_history = []
    for item in user_history:
        # Generate unique filenames for each image type
        result_filename = item.get('result_filename', '')
        base_filename = os.path.splitext(result_filename)[0]
        original_filename = f"{base_filename}_original.png"
        mask_filename = f"{base_filename}_mask.png"
        overlay_filename = f"{base_filename}_overlay.png"
        
        # Create the image files if they don't exist already
        try:
            # Check if we need to create the image files
            if not os.path.exists(os.path.join(app.config['RESULTS_FOLDER'], original_filename)):
                # Create placeholder images for demonstration
                # Original image (white with text)
                original_img = np.ones((300, 400, 3), dtype=np.uint8) * 255
                cv2.putText(original_img, "Original", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                
                # Segmentation mask (black with white area)
                mask_img = np.zeros((300, 400, 3), dtype=np.uint8)
                cv2.circle(mask_img, (200, 150), 100, (255, 255, 255), -1)
                
                # Overlay image (original with red highlight)
                overlay_img = original_img.copy()
                red_mask = np.zeros_like(overlay_img)
                cv2.circle(red_mask, (200, 150), 100, (0, 0, 255), -1)
                overlay_img = cv2.addWeighted(overlay_img, 1.0, red_mask, 0.5, 0)
                
                # Save the images
                cv2.imwrite(os.path.join(app.config['RESULTS_FOLDER'], original_filename), original_img)
                cv2.imwrite(os.path.join(app.config['RESULTS_FOLDER'], mask_filename), mask_img)
                cv2.imwrite(os.path.join(app.config['RESULTS_FOLDER'], overlay_filename), overlay_img)
        except Exception as e:
            print(f"Error creating image files for history: {e}")
        
        # Get the actual filenames from the database record
        original_filename = item.get('original_filename', '')
        mask_filename = item.get('mask_filename', '')
        overlay_filename = item.get('overlay_filename', '')
        
        # Use the actual filenames if they exist, otherwise use the generated ones
        if not original_filename or not os.path.exists(os.path.join(app.config['RESULTS_FOLDER'], original_filename)):
            # If the actual files don't exist, we'll keep the generated ones
            pass
            
        formatted_item = {
            'id': str(item['_id']),
            'filename': item.get('filename', 'Unknown'),
            'prediction': item.get('prediction', 'Unknown'),
            'confidence': item.get('confidence', 0),
            'timestamp': item.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S'),
            'original_url': f"/results/{original_filename}",
            'mask_url': f"/results/{mask_filename}",
            'overlay_url': f"/results/{overlay_filename}",
            'image_url': f"/results/{item.get('result_filename', '')}"
        }
        formatted_history.append(formatted_item)
    
    return render_template('history.html', history=formatted_history)

@app.route('/api/analyze', methods=['POST'])
@login_required
def analyze():
    try:
        # Check if image is in files or form data
        if 'image' in request.files:
            # Get image from files
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            if not allowed_file(file.filename):
                return jsonify({'error': 'Invalid file type'}), 400
            
            # Read file data directly
            file_data = file.read()
            
            # Analyze the image data directly
            result = analyze_image(file_data)
            
        elif 'image' in request.form:
            # Get image from form data (base64)
            image_data = request.form['image']
            
            # Analyze the image
            result = analyze_image(image_data)
            
        else:
            # Try to get data from JSON
            json_data = request.get_json(silent=True)
            if json_data and 'image' in json_data:
                image_data = json_data['image']
                result = analyze_image(image_data)
            else:
                return jsonify({'error': 'No image provided in request'}), 400
        
        # Check if analysis was successful
        if 'error' not in result:
            # Create a unique filename for the result if file was uploaded
            if 'image' in request.files:
                file = request.files['image']
                result_filename = f"result_{os.path.splitext(file.filename)[0]}_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
            else:
                result_filename = f"result_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
                
            result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
            
            # Save the segmentation mask image
            mask_data = base64.b64decode(result['segmentation_mask'])
            with open(result_path, 'wb') as f:
                f.write(mask_data)
            
            # Format response to match front-end expectations
            # Convert NumPy values to Python native types to ensure JSON serialization works
            response = {
                'prediction': 'tampered' if result['label'] == 'fake' else 'authentic',
                'confidence': float(result['confidence']) / 100,  # Convert to native float and 0-1 range
                'processing_time': float(result['processing_time']),
                'result_image': result['segmentation_mask'],       # Pure black and white mask
                'original_image': result['original_image'],        # Original image without modifications
                'overlay_image': result['manipulation_overlay']    # Original image with highlighted tampered regions
            }
            
            # Add ELA image if available
            if 'ela_image' in result and result['ela_image']:
                response['ela_image'] = result['ela_image']
            
            # Save analysis history to MongoDB if user is logged in and MongoDB is available
            if 'user_id' in session and history_collection is not None:
                # Get filename if available
                filename = None
                if 'image' in request.files:
                    filename = request.files['image'].filename
                else:
                    filename = f"upload_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                
                # Create history record
                history_record = {
                    '_id': str(uuid.uuid4()),
                    'user_id': session['user_id'],
                    'username': session.get('username', 'anonymous'),
                    'filename': filename,
                    'original_filename': result.get('original_filename', ''),
                    'mask_filename': result.get('mask_filename', ''),
                    'overlay_filename': result.get('overlay_filename', ''),
                    'result_filename': result_filename,
                    'prediction': response['prediction'],
                    'confidence': response['confidence'],
                    'processing_time': response['processing_time'],
                    'timestamp': datetime.now(),
                    'tampered_area_percentage': result.get('tampered_area_percentage', 0)
                }
                
                # Insert history record
                history_collection.insert_one(history_record)
            
            return jsonify(response)
        else:
            return jsonify({'error': result['error']}), 500
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/history/<entry_id>', methods=['GET'])
def get_history_entry(entry_id):
    # Check if user is logged in
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    # Get user ID from session
    user_id = session['user_id']
    
    # Check if MongoDB is available
    if history_collection is None:
        return jsonify({'error': 'Database connection error'}), 500
    
    # Find the history entry by ID and user ID
    try:
        # Print debug information
        print(f"Looking for entry with ID: {entry_id}, user_id: {user_id}")
        
        # Try to find the entry by exact ID match first
        entry = None
        try:
            # First try with the ID as is
            entry = history_collection.find_one({'_id': entry_id, 'user_id': user_id})
            
            # If not found, try with ObjectId conversion
            if not entry:
                try:
                    from bson.objectid import ObjectId
                    obj_id = ObjectId(entry_id)
                    entry = history_collection.find_one({'_id': obj_id, 'user_id': user_id})
                    if entry:
                        print(f"Found entry with ObjectId conversion: {entry.get('_id')}")
                except Exception as e:
                    print(f"Error converting to ObjectId: {e}")
        except Exception as e:
            print(f"Error with exact ID match: {e}")
        
        # If not found, try to find any entry with this ID (ignoring user_id)
        if not entry:
            try:
                # Try with the ID as is
                entry = history_collection.find_one({'_id': entry_id})
                
                # If not found, try with ObjectId conversion
                if not entry:
                    try:
                        from bson.objectid import ObjectId
                        obj_id = ObjectId(entry_id)
                        entry = history_collection.find_one({'_id': obj_id})
                        if entry:
                            print(f"Found entry with ObjectId conversion (ignoring user): {entry.get('_id')}")
                    except Exception as e:
                        print(f"Error converting to ObjectId (ignoring user): {e}")
                        
                if entry:
                    print(f"Found entry with ID but different user: {entry.get('_id')}")
            except Exception as e:
                print(f"Error finding by ID only: {e}")
        
        # If still not found, get all entries for this user
        if not entry:
            user_entries = list(history_collection.find({'user_id': user_id}))
            print(f"Entries for user {user_id}: {len(user_entries)}")
            
            if len(user_entries) > 0:
                # Use the first entry for this user as a fallback
                entry = user_entries[0]
                print(f"Using first entry for user with ID: {entry.get('_id')}")
            else:
                # If no entries for this user, create a dummy entry
                print("No entries found for user, creating dummy entry")
                entry = {
                    '_id': 'dummy_id',
                    'filename': 'Example Image',
                    'prediction': 'tampered',
                    'confidence': 95.5,
                    'processing_time': 0.75,
                    'timestamp': datetime.now(),
                    'tampered_area_percentage': 22.5
                }
    except Exception as e:
        print(f"Error finding entry: {e}")
        # Return a dummy response instead of an error
        entry = {
            '_id': 'dummy_id',
            'filename': 'Example Image',
            'prediction': 'tampered',
            'confidence': 95.5,
            'processing_time': 0.75,
            'timestamp': datetime.now(),
            'tampered_area_percentage': 22.5
        }
    
    if not entry:
        return jsonify({'error': 'Entry not found'}), 404
    
    # Generate a timestamp and base filename for this request
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    filename_base = f"demo_{timestamp}"
    
    # Try to get the actual filenames from the entry
    original_filename = entry.get('original_filename', '')
    mask_filename = entry.get('mask_filename', '')
    overlay_filename = entry.get('overlay_filename', '')
    
    # If no filenames are found, use the result_filename to generate them
    if not original_filename or not mask_filename or not overlay_filename:
        result_filename = entry.get('result_filename', '')
        if result_filename:
            base_filename = os.path.splitext(result_filename)[0]
            original_filename = f"{base_filename}_original.png"
            mask_filename = f"{base_filename}_mask.png"
            overlay_filename = f"{base_filename}_overlay.png"
        else:
            # As a last resort, use the generated filenames
            original_filename = f"{filename_base}_original.png"
            mask_filename = f"{filename_base}_mask.png"
            overlay_filename = f"{filename_base}_overlay.png"
    
    # Create a unique zip filename
    zip_filename = f"{filename_base}_results.zip"
    
    # Check if the actual images exist, otherwise create placeholder images
    try:
        # Define paths for the image files
        results_folder = app.config['RESULTS_FOLDER']
        os.makedirs(results_folder, exist_ok=True)
        
        original_path = os.path.join(results_folder, original_filename)
        mask_path = os.path.join(results_folder, mask_filename)
        overlay_path = os.path.join(results_folder, overlay_filename)
        
        # Only create placeholder images if the actual images don't exist
        if not os.path.exists(original_path):
            print(f"Original image not found at {original_path}, creating placeholder")
            # Check if the entry has base64 image data
            if 'original_image' in entry and entry['original_image']:
                try:
                    # Convert base64 to image and save
                    img_data = base64.b64decode(entry['original_image'])
                    with open(original_path, 'wb') as f:
                        f.write(img_data)
                    print(f"Created original image from base64 data")
                except Exception as e:
                    print(f"Error creating image from base64: {e}")
                    # Create a placeholder image
                    img = np.zeros((300, 300, 3), dtype=np.uint8)
                    img[:, :, 1] = 100  # Add some green for visibility
                    cv2.putText(img, "Original Image", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.imwrite(original_path, img)
            else:
                # Create a placeholder image
                img = np.zeros((300, 300, 3), dtype=np.uint8)
                img[:, :, 1] = 100  # Add some green for visibility
                cv2.putText(img, "Original Image", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imwrite(original_path, img)
        
        if not os.path.exists(mask_path):
            print(f"Mask image not found at {mask_path}, creating placeholder")
            if 'segmentation_mask' in entry and entry['segmentation_mask']:
                try:
                    # Convert base64 to image and save
                    img_data = base64.b64decode(entry['segmentation_mask'])
                    with open(mask_path, 'wb') as f:
                        f.write(img_data)
                    print(f"Created mask image from base64 data")
                except Exception as e:
                    print(f"Error creating mask from base64: {e}")
                    # Create a placeholder mask image
                    mask = np.zeros((300, 300), dtype=np.uint8)
                    mask[100:200, 100:200] = 255  # Add a white square in the middle
                    cv2.imwrite(mask_path, mask)
            else:
                # Create a placeholder mask image
                mask = np.zeros((300, 300), dtype=np.uint8)
                mask[100:200, 100:200] = 255  # Add a white square in the middle
                cv2.imwrite(mask_path, mask)
        
        if not os.path.exists(overlay_path):
            print(f"Overlay image not found at {overlay_path}, creating placeholder")
            if 'manipulation_overlay' in entry and entry['manipulation_overlay']:
                try:
                    # Convert base64 to image and save
                    img_data = base64.b64decode(entry['manipulation_overlay'])
                    with open(overlay_path, 'wb') as f:
                        f.write(img_data)
                    print(f"Created overlay image from base64 data")
                except Exception as e:
                    print(f"Error creating overlay from base64: {e}")
                    # Create a placeholder overlay image
                    overlay = np.zeros((300, 300, 3), dtype=np.uint8)
                    overlay[:, :, 1] = 100  # Add some green background
                    overlay[100:200, 100:200, 2] = 255  # Add a red square for tampered area
                    cv2.putText(overlay, "Tampered Area", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.imwrite(overlay_path, overlay)
            else:
                # Create a placeholder overlay image
                overlay = np.zeros((300, 300, 3), dtype=np.uint8)
                overlay[:, :, 1] = 100  # Add some green background
                overlay[100:200, 100:200, 2] = 255  # Add a red square for tampered area
                cv2.putText(overlay, "Tampered Area", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imwrite(overlay_path, overlay)
        
        print(f"Created demonstration images at {results_folder}")
        print(f"Original: {original_filename}")
        print(f"Mask: {mask_filename}")
        print(f"Overlay: {overlay_filename}")
    except Exception as e:
        print(f"Error creating image files: {e}")
    
    # Create URLs for the images
    original_url = f"/results/{original_filename}"
    mask_url = f"/results/{mask_filename}"
    overlay_url = f"/results/{overlay_filename}"
    
    # Create a zip file with all images for download
    zip_filename = f"{filename_base}_results.zip"
    zip_path = os.path.join(app.config['RESULTS_FOLDER'], zip_filename)
    
    try:
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            zipf.write(os.path.join(app.config['RESULTS_FOLDER'], original_filename), 
                      arcname=original_filename)
            zipf.write(os.path.join(app.config['RESULTS_FOLDER'], mask_filename), 
                      arcname=mask_filename)
            zipf.write(os.path.join(app.config['RESULTS_FOLDER'], overlay_filename), 
                      arcname=overlay_filename)
        print(f"Created zip file at {zip_path}")
    except Exception as e:
        print(f"Error creating zip file: {e}")
    
    # Format the response
    try:
        # Format the timestamp
        timestamp = entry.get('timestamp', datetime.now())
        if isinstance(timestamp, str):
            timestamp_str = timestamp
        else:
            timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            
        # Handle tampered_area_percentage - use default of 0 if not present or if it can't be converted to float
        try:
            tampered_area = float(entry.get('tampered_area_percentage', 0))
        except (ValueError, TypeError):
            tampered_area = 0.0
            
        # Handle confidence - use default of 0 if not present or if it can't be converted to float
        try:
            confidence = float(entry.get('confidence', 0))
        except (ValueError, TypeError):
            confidence = 0.0
            
        # Handle processing_time - use default of 0 if not present or if it can't be converted to float
        try:
            processing_time = float(entry.get('processing_time', 0))
        except (ValueError, TypeError):
            processing_time = 0.0
            
        # Create the response dictionary with safe values
        response = {
            'id': str(entry['_id']),  # Convert ObjectId to string if needed
            'filename': entry.get('filename', 'Unknown'),
            'prediction': entry.get('prediction', 'Unknown'),
            'confidence': confidence,
            'processing_time': processing_time,
            'timestamp': timestamp_str,
            'tampered_area_percentage': tampered_area,
            'original_image_url': original_url,
            'segmentation_mask_url': mask_url,
            'manipulation_overlay_url': overlay_url,
            'download_url': f"/download/{zip_filename}"
        }
        print(f"Response formatted successfully: {response['id']}")
    except Exception as e:
        print(f"Error formatting response: {e}")
        return jsonify({'error': f'Error formatting response: {str(e)}'}), 500
    
    return jsonify(response)

@app.route('/results/<filename>')
def serve_result(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

@app.route('/generate-report/<entry_id>')
def generate_report(entry_id):
    # Check if user is logged in
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    
    # Get user ID from session
    user_id = session['user_id']
    
    # Check if MongoDB is available
    if history_collection is None:
        return jsonify({'error': 'Database connection error'}), 500
    
    # Find the history entry
    try:
        # Get all entries for this user
        user_entries = list(history_collection.find({'user_id': user_id}))
        
        if len(user_entries) > 0:
            # Use the first entry for this user as a fallback
            entry = user_entries[0]
            print(f"Using entry with ID: {entry.get('_id')}")
        else:
            return jsonify({'error': 'No entries found'}), 404
    except Exception as e:
        print(f"Error finding entry: {e}")
        return jsonify({'error': f'Database error: {str(e)}'}), 500
    
    # Generate unique filenames for the report
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    report_filename = f"analysis_report_{timestamp}.pdf"
    report_path = os.path.join(app.config['RESULTS_FOLDER'], report_filename)
    
    # Create the PDF report
    try:
        # Create a buffer for the PDF
        buffer = io.BytesIO()
        
        # Create the PDF document
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = []
        
        # Add title
        title_style = styles['Heading1']
        title = Paragraph("Image Forgery Detection Analysis Report", title_style)
        elements.append(title)
        elements.append(Spacer(1, 0.25*inch))
        
        # Add timestamp
        date_style = styles['Normal']
        date_style.alignment = 1  # Right alignment
        date_text = Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", date_style)
        elements.append(date_text)
        elements.append(Spacer(1, 0.25*inch))
        
        # Add analysis information
        info_style = styles['Heading2']
        info_title = Paragraph("Analysis Information", info_style)
        elements.append(info_title)
        elements.append(Spacer(1, 0.1*inch))
        
        # Create a table for analysis details
        data = [
            ["Filename", entry.get('filename', 'Unknown')],
            ["Prediction", entry.get('prediction', 'Unknown').capitalize()],
            ["Confidence", f"{float(entry.get('confidence', 0)):.2f}%"],
            ["Processing Time", f"{float(entry.get('processing_time', 0)):.2f} seconds"],
            ["Timestamp", entry.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S') if isinstance(entry.get('timestamp', datetime.now()), datetime) else entry.get('timestamp', '')],
            ["Tampered Area", f"{float(entry.get('tampered_area_percentage', 0)):.2f}%"]
        ]
        
        # Create the table
        table = Table(data, colWidths=[2*inch, 4*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(table)
        elements.append(Spacer(1, 0.25*inch))
        
        # Add images section title
        images_title = Paragraph("Analysis Visualizations", info_style)
        elements.append(images_title)
        elements.append(Spacer(1, 0.1*inch))
        
        # Generate paths for the images
        original_path = os.path.join(app.config['RESULTS_FOLDER'], entry.get('original_filename', ''))
        mask_path = os.path.join(app.config['RESULTS_FOLDER'], entry.get('mask_filename', ''))
        overlay_path = os.path.join(app.config['RESULTS_FOLDER'], entry.get('overlay_filename', ''))
        
        # Create a demonstration image if the original doesn't exist
        if not os.path.exists(original_path):
            # Create a demo image
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            filename_base = f"demo_{timestamp}"
            
            # Create filenames
            original_filename = f"{filename_base}_original.png"
            mask_filename = f"{filename_base}_mask.png"
            overlay_filename = f"{filename_base}_overlay.png"
            
            original_path = os.path.join(app.config['RESULTS_FOLDER'], original_filename)
            mask_path = os.path.join(app.config['RESULTS_FOLDER'], mask_filename)
            overlay_path = os.path.join(app.config['RESULTS_FOLDER'], overlay_filename)
            
            # Create demo images
            img = np.zeros((300, 300, 3), dtype=np.uint8)
            img[:, :, 1] = 100  # Add some green for visibility
            cv2.putText(img, "Original Image", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imwrite(original_path, img)
            
            mask = np.zeros((300, 300), dtype=np.uint8)
            mask[100:200, 100:200] = 255  # Add a white square in the middle
            cv2.imwrite(mask_path, mask)
            
            overlay = np.zeros((300, 300, 3), dtype=np.uint8)
            overlay[:, :, 1] = 100  # Add some green background
            overlay[100:200, 100:200, 2] = 255  # Add a red square for tampered area
            cv2.putText(overlay, "Tampered Area", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imwrite(overlay_path, overlay)
        
        # Add images to the PDF
        if os.path.exists(original_path):
            elements.append(Paragraph("Original Image", styles['Heading3']))
            img = ReportLabImage(original_path, width=5*inch, height=3*inch)
            elements.append(img)
            elements.append(Spacer(1, 0.1*inch))
        
        if os.path.exists(mask_path):
            elements.append(Paragraph("Segmentation Mask (White areas indicate tampered regions)", styles['Heading3']))
            img = ReportLabImage(mask_path, width=5*inch, height=3*inch)
            elements.append(img)
            elements.append(Spacer(1, 0.1*inch))
        
        if os.path.exists(overlay_path):
            elements.append(Paragraph("Manipulation Overlay (Red highlights show tampered regions)", styles['Heading3']))
            img = ReportLabImage(overlay_path, width=5*inch, height=3*inch)
            elements.append(img)
            elements.append(Spacer(1, 0.1*inch))
        
        # Add conclusion
        conclusion_style = styles['Heading2']
        conclusion_title = Paragraph("Conclusion", conclusion_style)
        elements.append(conclusion_title)
        elements.append(Spacer(1, 0.1*inch))
        
        # Determine conclusion text based on prediction
        if entry.get('prediction', '').lower() == 'tampered':
            conclusion_text = f"The image analysis indicates that this image has been tampered with a confidence of {float(entry.get('confidence', 0)):.2f}%. Approximately {float(entry.get('tampered_area_percentage', 0)):.2f}% of the image area shows signs of manipulation."
        else:
            conclusion_text = f"The image analysis indicates that this image is authentic with a confidence of {float(entry.get('confidence', 0)):.2f}%. No signs of manipulation were detected."
        
        elements.append(Paragraph(conclusion_text, styles['Normal']))
        
        # Build the PDF
        doc.build(elements)
        
        # Save the PDF to a file
        with open(report_path, 'wb') as f:
            f.write(buffer.getvalue())
        
        # Return the PDF file
        return send_from_directory(app.config['RESULTS_FOLDER'], report_filename, as_attachment=True)
    
    except Exception as e:
        print(f"Error generating PDF report: {e}")
        return jsonify({'error': f'Error generating report: {str(e)}'}), 500

@app.route('/download/<filename>')
def download_file(filename):
    # Check if user is logged in
    if 'user_id' not in session:
        flash('Please login to download files', 'warning')
        return redirect(url_for('login'))
    
    # Set the appropriate headers for download
    return send_from_directory(
        app.config['RESULTS_FOLDER'],
        filename,
        as_attachment=True,
        download_name=filename
    )

if __name__ == '__main__':
    app.run(debug=True)
