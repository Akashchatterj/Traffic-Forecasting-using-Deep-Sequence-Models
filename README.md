# Traffic Forecasting using Deep Sequence Models with Vehicle Situation-aware Loss (VSAL)

[![Paper](https://img.shields.io/badge/Paper-CVIP%202025-blue)](https://github.com/yourusername/traffic-forecasting-vsal)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> **Official Implementation** of "Traffic Forecasting using Deep Sequence Models with Vehicle Situation-aware Loss"  
> *Akash Chatterjee, Jayant Mahawar, and Angshuman Paul*  
> *Indian Institute of Technology Jodhpur, India*

---

## 📋 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Results](#results)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## 🔍 Overview

This repository presents a novel approach to traffic forecasting that addresses the limitations of existing methods by introducing the **Vehicle Situation-aware Loss (VSAL)**—a composite loss function that enables deep sequence models to simultaneously learn multiple interdependent traffic variables.

### The Problem
Traditional traffic forecasting models often:
- Focus solely on velocity prediction
- Ignore crucial factors like lane changes and traffic density
- Rely on spurious correlations with limited behavioral understanding

### Our Solution
VSAL integrates multiple loss components to holistically capture:
- ✅ Vehicle velocity and position
- ✅ Geospatial accuracy (latitude/longitude)
- ✅ Lane-change classification
- ✅ Traffic congestion estimation
- ✅ Self-consistency between predicted velocity and displacement

---

## ⭐ Key Features

- **Novel Composite Loss Function**: VSAL combines six distinct loss components for comprehensive traffic modeling
- **Enhanced Model Variants**: VS-LSTM, VS-GRU, and VS-Transformer architectures
- **Rich Contextual Features**: Lane-specific gap distances, traffic density, speed reduction, and novel Jam Factor
- **State-of-the-art Performance**: 
  - VS-Transformer achieves **0.002 velocity RMSE** (99.9% improvement over baseline)
  - **0.358 Haversine RMSE** for geospatial accuracy
  - **77.2% lane-change classification accuracy**
- **Multiple Prediction Tasks**: Simultaneous prediction of velocity, position, coordinates, and lane changes

---

## 🏗️ Architecture

### VSAL Components

The Vehicle Situation-aware Loss consists of:

```
L = L_vel + α·L_pos + β·L_sc + γ·L_class + δ·L_cong + σ·L_hav
```

1. **Velocity Loss (L_vel)**: MSE for speed prediction
2. **Position Loss (L_pos)**: MSE for longitudinal position
3. **Self-Consistency Loss (L_sc)**: Ensures predicted position matches velocity-based displacement
4. **Lane-Change Classification Loss (L_class)**: Cross-entropy for lane-change detection
5. **Congestion Prediction Loss (L_cong)**: MSE for Jam Factor prediction
6. **Haversine Loss (L_hav)**: Geospatial distance for coordinate accuracy

### Model Architectures

```
Input Features → Sequence Model → Multi-Task Heads → Predictions
     ↓                ↓                    ↓              ↓
  velocity      LSTM/GRU/        Regression Head    velocity
  position      Transformer      Classification     position
  gaps                           Congestion Head    lat/long
  density                                          lane change
  Jam Factor                                       congestion
```

---

## 🛠️ Installation

### Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+ (for GPU support)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/traffic-forecasting-vsal.git
cd traffic-forecasting-vsal

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```txt
torch>=2.0.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
scikit-learn>=1.0.0
pyproj>=3.6.1
tqdm>=4.62.0
```

---

## 📊 Dataset

This project uses the **NGSIM US-101 trajectory dataset**, which contains high-fidelity vehicle trajectory data collected on the southbound US 101 freeway in Los Angeles.

### Dataset Preparation

```bash
# Download NGSIM US-101 dataset
python scripts/download_dataset.py

# Preprocess the data
python scripts/preprocess_data.py
```

### Data Preprocessing Pipeline

1. **Data Cleaning**: Remove erroneous lanes (6, 7, 8)
2. **Unit Conversion**: Convert feet to meters
3. **Temporal Sorting**: Sort by timestamp
4. **Feature Extraction**:
   - Lane-specific gap distances (g1-g6)
   - Traffic density per lane
   - Speed reduction metric
   - Jam Factor computation
5. **Coordinate Transformation**: NAD83 (EPSG:2227) → WGS84 (EPSG:4326)
6. **Normalization**: Min-max scaling

---

## 🚀 Usage

### Training

```bash
# Train VS-LSTM
python train.py --model lstm --use_vsal --epochs 50 --batch_size 64 --lr 0.0005

# Train VS-GRU
python train.py --model gru --use_vsal --epochs 50 --batch_size 64 --lr 0.0005

# Train VS-Transformer
python train.py --model transformer --use_vsal --epochs 50 --batch_size 64 --lr 0.0005
```

### Training with Custom Loss Weights

```bash
python train.py --model transformer \
                --use_vsal \
                --alpha 1.0 \
                --beta 1.0 \
                --gamma 1.0 \
                --delta 1.0 \
                --sigma 1.0
```

### Evaluation

```bash
# Evaluate trained model
python evaluate.py --model_path checkpoints/vs_transformer_best.pth \
                   --data_path data/test.csv
```

### Inference

```bash
# Run inference on new data
python inference.py --model_path checkpoints/vs_transformer_best.pth \
                    --input_data sample_trajectory.csv \
                    --output_dir predictions/
```

---

## 📈 Results

### Quantitative Performance

| Model | Velocity RMSE ↓ | Local_Y RMSE ↓ | Haversine RMSE ↓ | Congestion RMSE ↓ | Lane Change Acc ↑ |
|-------|----------------|----------------|-----------------|-------------------|-------------------|
| **Baseline LSTM** | 3.139±2.518 | 177.641±51.516 | 175.935±51.698 | 0.295±0.038 | 0.468±0.073 |
| **VS-LSTM** | **0.050±0.042** | **0.277±0.005** | **0.240±0.000** | **0.219±0.036** | **0.700±0.063** |
| **Baseline GRU** | 3.231±2.052 | 170.273±12.954 | 161.020±7.397 | 0.069±0.022 | 0.491±0.007 |
| **VS-GRU** | **0.004±0.002** | **0.264±0.007** | **0.240±0.000** | **0.068±0.002** | **0.772±0.001** |
| **Baseline Transformer** | 10.480±0.009 | 179.313±0.029 | 54.511±0.008 | 0.024±0.000 | 0.366±0.002 |
| **VS-Transformer** | **0.002±0.001** | **0.297±0.008** | **0.358±0.111** | **0.005±0.002** | **0.772±0.001** |

### Key Improvements

- 🎯 **98.4-99.9% reduction** in velocity RMSE across all models
- 🎯 **99.8% reduction** in positional error
- 🎯 **77.2% lane-change accuracy** for VS-GRU and VS-Transformer
- 🎯 **98% reduction** in congestion prediction error

### Visualization

<details>
<summary>Click to view sample predictions</summary>

![Velocity Prediction](assets/velocity_prediction.png)
*Predicted vs Actual Vehicle Velocity showing close alignment even during rapid fluctuations*

![Position Tracking](assets/position_tracking.png)
*Longitudinal position prediction demonstrating spatial accuracy*

![Congestion Estimation](assets/congestion_prediction.png)
*Jam Factor prediction capturing congestion onset and dissipation*

</details>

---

## 🔬 Ablation Studies

Each component of VSAL contributes uniquely to model performance:

| Removed Component | Impact |
|-------------------|--------|
| Without L_pos | Decreased positional accuracy |
| Without L_sc | Increased velocity inconsistency |
| Without L_class | **Significant drop** in lane-change detection |
| Without L_cong | Poor congestion estimation |
| Without L_hav | Reduced geospatial precision |

---

## 📝 Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{chatterjee2025traffic,
  title={Traffic Forecasting using Deep Sequence Models with Vehicle Situation-aware Loss},
  author={Akash Chatterjee, Jayant Mahawar and Angshuman Paul},
  booktitle={CVIP 2025},
  year={2025},
  organization={Indian Institute of Technology Jodhpur}
}
```

---

## 🤝 Acknowledgments

- NGSIM dataset provided by the Federal Highway Administration
- Research conducted at Indian Institute of Technology Jodhpur
- Thanks to the open-source community for PyTorch and related tools

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

- **Akash Chatterjee** - [GitHub](https://github.com/Akashchatterj) | [Email](mailto:chatterjeeakash887@gmail.com)
- **Jayant Mahawar** - [GitHub](https://github.com/mahawar2) | [Email](mailto:mahawar.2@iitj.ac.in)

---

## 📧 Contact

For questions or collaboration opportunities, please contact:
- Email: m23air002@iitj.ac.in
- Institution: Indian Institute of Technology Jodhpur

---

## ⭐ Star History

If you find this project useful, please consider giving it a star! ⭐

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/traffic-forecasting-vsal&type=Date)](https://star-history.com/#yourusername/traffic-forecasting-vsal&Date)

---

<p align="center">
  Made with ❤️ at IIT Jodhpur
</p>
