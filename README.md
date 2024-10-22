# Clustering Project

This project implements clustering algorithms on the MNIST dataset using various machine learning techniques and libraries. It includes both Python scripts and visualizations to analyze and demonstrate the performance of the clustering methods.

## Project Overview

- **Data**: The project uses the MNIST dataset, stored in both raw and processed formats, to perform clustering.
- **Clustering Algorithms**: The main Python script performs clustering using techniques such as k-means and PCA.
- **Visualizations**: The project includes several visualizations of the clustering results, including feature vector representations and noise analysis.

## Project Structure

- **Clustering.py**: The main Python script that runs the clustering algorithms on the MNIST dataset.
- **data/MNIST**: Contains the MNIST dataset in both raw and processed formats.
- **figures/**: Directory containing visualizations of the clustering results, including noise analysis and k-means results.

## Installation

### Prerequisites

- **Python 3.x**: Ensure Python 3.x is installed on your machine.
- **Required Libraries**: Install the required libraries listed in `requirements.txt`.

### Setup Instructions

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/YourUsername/Clustering.git
    cd Clustering
    ```

2. **Install Dependencies**:
    Install the necessary dependencies for running the project:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Project

1. **Run the Clustering Script**:
    Execute the main clustering script to perform clustering on the MNIST dataset:
    ```bash
    python Clustering.py
    ```

2. **Visualize the Results**:
    View the visualizations generated from the clustering process in the `figures/` directory.

## Project Workflow

1. **Data Loading**: Load the MNIST dataset from the `data/MNIST/` directory.
2. **Clustering**: Perform clustering on the dataset using techniques like k-means and PCA.
3. **Visualization**: Generate visualizations of the clustering results and analyze the performance of the algorithms.
