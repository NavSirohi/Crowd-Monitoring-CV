# Crowd Monitoring System for Computer Vision

## Real-World Use Case Focus
This project provides a highly robust **Computer Vision** solution specifically designed to analyze images of heavily congested spaces and calculate precise foot counts by identifying and annotating every person detected in the frame. 

**Intended Applications:**
- **Temples and Places of Worship:** Calculating actual foot traffic in highly dense corridors to predict spatial needs and prevent crushing hazards.
- **Concerts/Stadiums:** Instantly validating occupancy levels at entrance gates or standing floor sections.
- **Public Infrastructure:** Managing transport hubs like railway stations or localized security events.

By using the state-of-the-art **YOLOv8 Object Detection Framework** (Ultralytics), it accurately separates overlap in dense crowds without succumbing to the noise of background interference that earlier CV methods (like Haar Cascades or HOG) generate.

---

## 1. Prerequisites 
Before running the project, ensure you have the following installed on your machine:
- **Python 3.8+** (We strongly suggest Python 3.10)
- **pip** package installer
- A working terminal or command line (Mac, Linux, or Windows WSL/Powershell works perfectly).

---

## 2. Environment Setup & Dependency Installation

In your terminal, navigate to the root directory of this repository (where this `README.md` file is located). Then perfectly execute these steps sequentially:

### Step 2.1: Create a Virtual Environment (Highly Recommended)
Creating a local virtual environment prevents dependency conflict with your other projects.

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate
```

### Step 2.2: Install Requirements
Once your virtual environment is activated, install all system dependencies:
```bash
pip install -r requirements.txt
```
*(Note: Be patient as `ultralytics` installs PyTorch under the hood, which may take a few minutes).*

---

## 3. Project File Structure
```text
.
├── crowd_monitor.py      # Main CLI executable program
├── requirements.txt      # Dependency specification file
├── data/
│   └── sample_crowd.jpg  # Pre-supplied image for testing
├── results/              # Directory where the algorithm outputs counted images
└── README.md             # This instruction manual
```

---

## 4. Execution (Strictly CLI Based)

This software operates purely from your command-line interface. It intentionally does not implement graphical Pop-up UI commands (e.g. `cv2.imshow`) so it is completely safe to run in headless auto-evaluation pipelines.

### Let's run a test query!
From the root directory with your virtual environment still running, execute the prediction python file. You must supply an `--image` and an `--output` path.

```bash
python crowd_monitor.py --image data/sample_crowd.jpg --output results/output_test.jpg
```

### What Happens When You Run This?
1. **Model Download**: On the very first run, YOLO will securely auto-download a tiny pre-trained weights file (`yolov8n.pt`, approx 6 MB) onto your computer automatically.
2. **Analysis**: The script instantly loads the image via OpenCV and processes it through the advanced YOLOv8 detector searching explicitly for the `person` class in a dense environment.
3. **Response**: The algorithm will print an analysis breakdown straight to your terminal indicating exact foot limits.
4. **Visual Result**: It generates and saves a *new* image file located exactly where you pointed it (`results/output_test.jpg`). If you open that file in your file explorer, you'll see every individual correctly boxed in bright green with the total counted overlay on the top left.

### Custom Configurations
To adjust the confidence threshold of the algorithm (e.g., if you are missing distant people or incorrectly identifying shapes in terrible lighting):
```bash
python crowd_monitor.py --image data/your_image.jpg --output results/out.jpg --conf 0.15
```
*(Default configuration is optimized natively at `0.25`)*

---

## Clean Up
Once evaluation is complete, you can safely deactivate the Python environment by simply typing:
```bash
deactivate
```
