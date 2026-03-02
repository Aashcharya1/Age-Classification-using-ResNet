# Age Classification Assignment - Running Instructions

This guide provides step-by-step instructions to run the training pipeline and generate all submission files.

---

## Prerequisites

### 1. System Requirements
- Python 3.7 or higher
- CUDA-capable GPU (recommended) or CPU
- At least 8GB RAM (16GB+ recommended)
- Sufficient disk space for dataset and model files (~2GB+)

---

## GPU Setup (Windows + NVIDIA) — **No code changes required**

This project already uses:
- `DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`
- `USE_AMP = torch.cuda.is_available()`

So **GPU will be used automatically** as soon as you install a CUDA-enabled PyTorch build and run Jupyter from that same environment.

### 0. Confirm your GPU driver is installed
Open **PowerShell** and run:

```powershell
nvidia-smi
```

If this prints your GPU name + driver/CUDA version, your NVIDIA driver is working.

### 1. Create a clean Python environment (recommended)
Open **cmd** in the project folder:

```bat
cd /d "F:\Age Classifier"
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
```

### 2. Install **GPU** PyTorch
Install PyTorch + torchvision from the official CUDA wheels index:

```bat
python -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Then install the remaining packages:

```bat
python -m pip install numpy pillow jupyter
```

### 3. Verify PyTorch sees your GPU

```bat
python -c "import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

You want `cuda available True` and a device name (e.g., RTX 3050).

### 4. Start Jupyter from the same environment

```bat
jupyter notebook
```

Open `starter_notebook.ipynb` and re-run the device cell — it should print `Device: cuda | AMP: True`.

### Common pitfall: Jupyter using a different Python
If the notebook still shows CPU, it’s almost always because the Jupyter kernel is not using your `.venv` environment. The easiest fix is: **activate `.venv` and start Jupyter from that same terminal** (step 4).

### If you accidentally installed CPU-only torch

```bat
python -m pip uninstall -y torch torchvision
python -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 2. Install Dependencies

Open a terminal/command prompt in the project directory and install required packages:

```bash
pip install torch torchvision pillow numpy
```

**Note for Windows users:** The code automatically sets `NUM_WORKERS=0` to avoid multiprocessing issues. On Linux/Mac, you can use multiple workers for faster data loading.

---

## Dataset Setup

### 1. Download the Dataset
Download the dataset from the assignment link and extract it to your project directory.

### 2. Verify Directory Structure

Ensure your directory structure looks like this:

```
Age Classifier/
├── dataset/
│   ├── train/
│   │   ├── 0/          # 9,166 Young images
│   │   └── 1/          # 9,166 Old images
│   ├── valid/          # 134 validation images
│   └── valid_labels.csv
├── starter_notebook.ipynb
├── model_class.py
├── training_utils.py
├── evaluate_submission_student.py
└── ... (other files)
```

### 3. Verify Dataset Files

- Training images: 18,332 total (9,166 per class)
- Validation images: 134 total
- `valid_labels.csv` should contain ground truth labels for validation images

---

## Configuration

### 1. Set Your Roll Number

**IMPORTANT:** Before running, you must set your roll number in the notebook.

Open `starter_notebook.ipynb` and locate the configuration cell (Cell 2). Find this line:

```python
ROLL_NO = 'roll_no'  # <-- CHANGE THIS TO YOUR ROLL NUMBER
```

Change it to your actual roll number (lowercase):

```python
ROLL_NO = 'your_roll_number'  # e.g., '2024cs12345'
```

**This is critical** - the submission files will be named using this roll number.

---

## Running the Training Pipeline

### Option 1: Using Jupyter Notebook (Recommended)

1. **Start Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

2. **Open `starter_notebook.ipynb`**

3. **Run All Cells:**
   - Go to `Cell` → `Run All`
   - Or run cells sequentially using `Shift + Enter`

4. **Monitor Training:**
   - The notebook will print progress every 5 epochs (or every epoch for first 5)
   - Phase 1: Trains on training set, validates on validation set
   - Phase 2: Retrains on combined train+validation data
   - Total training time: ~2-4 hours on GPU (depending on hardware)

### Option 2: Convert to Python Script (Alternative)

If you prefer running as a Python script:

1. **Convert notebook to script:**
   ```bash
   jupyter nbconvert --to script starter_notebook.ipynb
   ```

2. **Run the script:**
   ```bash
   python starter_notebook.py
   ```

---

## What Happens During Training

### Phase 1: Training on Training Set

1. **Data Loading:**
   - Loads training images from `dataset/train/0/` and `dataset/train/1/`
   - Loads validation images from `dataset/valid/`
   - Applies data augmentation to training images

2. **Model Initialization:**
   - Creates ResNet-18 model with heavy classification head
   - Initializes optimizer (Adam), scheduler (OneCycleLR), EMA, and AMP scaler

3. **Training Loop (100 epochs):**
   - Each epoch:
     - Applies MixUp/CutMix augmentation (50% probability)
     - Forward pass with mixed precision
     - Computes loss (label-smoothed CE or combined loss)
     - Backward pass with gradient clipping
     - Updates optimizer and scheduler
     - Updates EMA weights
   - Every epoch:
     - Validates using EMA weights + TTA
     - Saves best model if validation accuracy improves

4. **Output:**
   - Prints training loss, training accuracy, validation accuracy, learning rate
   - Saves `best_model.pth` when validation accuracy improves

### Phase 2: Retraining on Combined Data

1. **Combined Dataset:**
   - Merges training and validation samples
   - Total: 18,466 images (18,332 train + 134 valid)

2. **Fresh Model:**
   - Creates a new model (not continuing from Phase 1)
   - Reinitializes optimizer, scheduler, EMA, scaler

3. **Training Loop (100 epochs):**
   - Same training procedure as Phase 1
   - Trains on the combined dataset

4. **Final Processing:**
   - Applies EMA weights to model
   - Calibrates BatchNorm running statistics (80 forward passes)
   - Sets module path for proper model loading

5. **Saving:**
   - Saves final model as `{ROLL_NO}.pth`
   - Copies `model_class.py` to `{ROLL_NO}.py`

---

## Generated Files

After successful training, you should have:

1. **`{ROLL_NO}.pth`**: Trained model file (full model, not just weights)
   - Size: ~45-50 MB
   - Contains complete model state

2. **`{ROLL_NO}.py`**: Model class definition file
   - Copy of `model_class.py`
   - Required for `torch.load()` to reconstruct the model

3. **`best_model.pth`** (optional): Best model from Phase 1
   - Saved for reference
   - Not required for submission

---

## Verification Before Submission

### 1. Run Evaluation Script

Verify your submission files work correctly:

```bash
python evaluate_submission_student.py \
    --model_path {ROLL_NO}.pth \
    --model_file {ROLL_NO}.py \
    --data_dir dataset/
```

**Expected Output:**
```
Device: cuda (or cpu)
Importing model definitions from: {ROLL_NO}.py
Loading model from: {ROLL_NO}.pth
Running inference on valid images...
  134 images processed.

Accuracy: XX.XX%  (XX/134)
```

If this runs without errors and prints an accuracy, your submission is correctly formatted.

### 2. Check File Sizes

- `{ROLL_NO}.pth`: Should be ~45-50 MB
- `{ROLL_NO}.py`: Should be ~2-3 KB

### 3. Verify File Names

- All files must use **lowercase** roll number
- Files must be named exactly: `{roll_no}.pth`, `{roll_no}.py`
- No spaces or special characters (except underscore if part of roll number)

---

## Creating the PDF Report

### Requirements:
- **One page maximum** (reports exceeding one page will not be evaluated)
- PDF format
- Explain your core approach/idea

### Suggested Content:
1. **Model Architecture**: Brief description of ResNet-18 + heavy head
2. **Key Techniques**: List main techniques (augmentation, MixUp/CutMix, EMA, etc.)
3. **Training Strategy**: Two-phase approach
4. **Results**: Validation accuracy achieved
5. **Key Insights**: What worked well, any observations

### Save as:
- `{ROLL_NO}.pdf` (lowercase roll number)

---

## Final Submission Checklist

Before submitting, verify:

- [ ] `{ROLL_NO}.pth` exists and is ~45-50 MB
- [ ] `{ROLL_NO}.py` exists and contains model class definition
- [ ] `{ROLL_NO}.pdf` exists and is one page
- [ ] All files use **lowercase** roll number
- [ ] Evaluation script runs successfully on your files
- [ ] Model was trained on combined train+validation data (Phase 2)
- [ ] Model uses ResNet-18 backbone (as required)
- [ ] Model was trained from scratch (no pretrained weights)

---

## Troubleshooting

### Common Issues

#### 1. **CUDA Out of Memory**
**Solution:**
- Reduce `BATCH_SIZE` in Cell 2 (try 32 or 16)
- Close other applications using GPU
- Use CPU (slower but works)

#### 2. **File Not Found Errors**
**Solution:**
- Verify dataset directory structure
- Check that `valid_labels.csv` exists
- Ensure paths are correct (use absolute paths if needed)

#### 3. **Import Errors**
**Solution:**
- Ensure all dependencies are installed: `pip install torch torchvision pillow numpy`
- Check that `model_class.py` and `training_utils.py` are in the same directory

#### 4. **Model Loading Errors**
**Solution:**
- Ensure `{ROLL_NO}.py` contains the exact model class definition
- Verify model was saved with `torch.save(model, ...)` not `torch.save(model.state_dict(), ...)`
- Check that module path is set correctly (done automatically in Phase 2)

#### 5. **Windows Multiprocessing Issues**
**Solution:**
- The code automatically sets `NUM_WORKERS=0` on Windows
- If you still have issues, manually set `NUM_WORKERS=0` in Cell 2

#### 6. **Training is Too Slow**
**Solution:**
- Ensure CUDA is available: `torch.cuda.is_available()` should return `True`
- Check GPU is being used: Look for "Device: cuda" in output
- Reduce batch size if GPU memory is limited
- On CPU, training will be very slow (expect 10+ hours)

#### 7. **Validation Accuracy Not Improving**
**Solution:**
- This is normal - check the assignment hint about train/valid patterns
- Ensure you're using EMA weights for validation
- Try adjusting hyperparameters (learning rate, weight decay)

---

## Quick Start Summary

1. **Install dependencies:**
   ```bash
   pip install torch torchvision pillow numpy
   ```

2. **Set roll number in `starter_notebook.ipynb` (Cell 2):**
   ```python
   ROLL_NO = 'your_roll_number'
   ```

3. **Run notebook:**
   - Open `starter_notebook.ipynb` in Jupyter
   - Run all cells (`Cell` → `Run All`)

4. **Wait for training to complete:**
   - Phase 1: ~1-2 hours (GPU) or ~5-10 hours (CPU)
   - Phase 2: ~1-2 hours (GPU) or ~5-10 hours (CPU)

5. **Verify submission:**
   ```bash
   python evaluate_submission_student.py \
       --model_path {ROLL_NO}.pth \
       --model_file {ROLL_NO}.py \
       --data_dir dataset/
   ```

6. **Create PDF report:**
   - Write one-page report explaining your approach
   - Save as `{ROLL_NO}.pdf`

7. **Submit:**
   - `{ROLL_NO}.pth`
   - `{ROLL_NO}.py`
   - `{ROLL_NO}.pdf`

---

## Expected Training Time

| Hardware | Phase 1 | Phase 2 | Total |
|----------|---------|---------|-------|
| GPU (RTX 3060/3070) | ~1-2 hours | ~1-2 hours | ~2-4 hours |
| GPU (RTX 2080) | ~1.5-2.5 hours | ~1.5-2.5 hours | ~3-5 hours |
| CPU (8 cores) | ~8-12 hours | ~8-12 hours | ~16-24 hours |
| CPU (4 cores) | ~15-20 hours | ~15-20 hours | ~30-40 hours |

**Note:** Times are approximate and depend on hardware, batch size, and other factors.

---

## Additional Notes

- **Reproducibility**: The code uses a fixed random seed (42) for reproducibility
- **Progress Monitoring**: Training progress is printed every 5 epochs (or every epoch for first 5)
- **Best Model**: Phase 1 saves the best model based on validation accuracy
- **Final Model**: Phase 2 model is the one used for submission
- **No Pretrained Weights**: Model is trained from scratch as required
- **ResNet-18 Only**: Uses ResNet-18 backbone as specified in assignment

---

## Support

If you encounter issues not covered here:

1. Check the assignment README for requirements
2. Verify your dataset structure matches the expected format
3. Ensure all dependencies are correctly installed
4. Check that your roll number is set correctly
5. Review error messages carefully - they often indicate the issue

Good luck with your submission!
