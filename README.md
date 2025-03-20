# Logo Clustering Pipeline

## Overview
This project is a **logo clustering pipeline** that fetches, processes, and clusters website favicons using deep learning and clustering techniques. It extracts features from favicons and groups similar ones together using **DBSCAN clustering**.

## Features
- **Fetch favicons** from websites via Google Favicon API & direct URL.
- **Download favicons in-memory** (no local storage needed).
- **Remove duplicate favicons** based on MD5 hashing.
- **Extract deep features** using a choice of:
  - `ResNet50`
  - `MobileNetV2`
- **Cluster similar favicons** using `DBSCAN` with cosine distance.
- **Visualize clusters** using `t-SNE` for dimensionality reduction.

## Installation
### Prerequisites
Ensure you have **Python 3.8+** installed.

### Clone the Repository
```sh
git clone https://github.com/RaresMarta/Logo-Clustering-Pipeline
cd Logo-Clustering-Pipeline
```

### Install Dependencies
```sh
pip install -r requirements.txt
```

## Usage
### Running the Pipeline
To run the logo clustering pipeline, execute:
```sh
python LogoClustering.py --model mobilenet_v2
```

#### Arguments:
| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Model to use for feature extraction (`resnet50` or `mobilenet_v2`) | `mobilenet_v2` |

## Pipeline Steps
1. **Load & Clean Data**: Remove duplicates and filter domains.
2. **Fetch Favicon URLs**: Retrieve favicons from websites.
3. **Download Images**: Download and process favicons in-memory.
4. **Remove Duplicates**: Ensure each favicon is unique before clustering.
5. **Extract Features**: Use deep learning models to extract logo features.
6. **Cluster Favicons**: Use DBSCAN to group similar logos.
7. **Visualize Clusters**: Reduce dimensions using `t-SNE` and plot clusters.

## Example Output
After clustering, the program will output sample icons from detected clusters, visualizing them in groups based on similarity.

## License
This project is licensed under the MIT License.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss your idea.

## Author
Developed by [Marta Rares]
