# "argohycom toolbox" - Argo and Colocated Model Fields Comparison Toolbox


Thanks to the generous and valuable contributions of:  
Annette Samuelsen, Research Leader, Senior Researcher, Ocean Modelling Group, NERSC, Norway  
Çağlar Yumruktepe, Researcher, Ocean Modelling Group, NERSC, Norway


NOTE: The toolbox requires argo and colocated model datasets in netCDF format


*This toolbox provides a collection of functions designed to facilitate the preprocessing of Argo float and colocated model fields. Its primary purpose is to enable a straightforward comparison between Argo float observations and model data by processing and formatting them for side-by-side analysis*


# Features

* Argo Float Preprocessing: Streamline the preprocessing of Argo float data, including quality control, data interpolation, and formatting.
* Model Field Preprocessing: Prepare colocated model fields for comparison with Argo data, including format conversion and data alignment.
* Side-by-Side Comparison: Easily visualize and analyze Argo float observations and colocated model fields in a comparative manner.
	* Visual Comparison: Generate plots and visualizations to compare the datasets.
	* Statistical Comparison: Calculate and compare 23 statistical metrics to quantify the differences and similarities between the datasets.
* Analysis: Perform further analysis and interpretation based on the comparative results, leveraging the toolbox's flexibility and extensibility.
    

The toolbox includes the following 23 statistical metrics for a thorough comparison:
    

```
Pearson correlation (r)
Probability value (p-value)
Spearman correlation (r)
Probability value (p-value)
R-squared (r2)
Root Mean Square Error (RMSE)
Normalised Root Mean Square Error (NRMSE)
Mean Error (ME)
Index of Agreement (d)
Modified Index of Agreement (md)
Relative Index of Agreement (rd)
Kling Gupta Efficiency (KGE)
Deviation of gain (DG)
Standard deviation of residual (SDR)
Mean Absolute Error (MAE)
Mean Bias Error (MBE)
Anomaly Correlation Coefficient (ACC)
Nash-Sutcliffe Efficiency (NSE)
Modified Nash-Sutcliffe Efficiency (MNSE)
Relative Nash-Sutcliffe Efficiency (RNSE)
Model Efficiency (ME)
Percentage model bias (Pbias)
Bias (bias)

```

# Installation requirements

### Clone this repository 

```
git clone https://github.com/akashspunnayil/cmlprojects-argohycom_toolbox.git
```
this directory contains following subdirectories:  
examples: Examples jupyter notebooks to work with the toolbox functions. It is your working directory. Your data should be in this directory.  
helper_functions: python files contain essential files of the toolbox  
outputs: this is where your outputs were saved..



The functions in this toolbox requires some Python packages listed in the requirements.txt file.

To install these dependencies, you can use pip.

*I strongly recommend using a Python virtual environment or a conda environment for the toolbox before installing the dependencies. This is not mandatory, but it is advisable to be on the safe side.*


### Using Python Virtual Environment

1. **Create a virtual environment:**
   ```bash
   python -m venv argo-env
   ```

2. **Activate the virtual environment:**

   - **On Windows:**
     ```bash
     .\argo-env\Scripts\activate
     ```

   - **On macOS and Linux:**
     ```bash
     source argo-env/bin/activate
     ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

### Using Conda Environment

1. **Create a conda environment:**
   ```bash
   conda create --name argo-env python=3.8
   ```

2. **Activate the conda environment:**
   ```bash
   conda activate argo-env
   ```

3. **Install the required packages:**
   ```bash
   conda install --file requirements.txt
   ```
### Without any environment

```
pip install -r requirements.txt

```


# Development Status

The automation of data download and colocating is not yet fully implemented and is currently under development. Contributions to these areas are highly appreciated. If you have experience in automating data workflows or any other relevant expertise, your input would be very valuable.


# Contributing

Contributions are welcome! If you encounter any issues, have suggestions for improvements, or would like to contribute new features, please feel free to open an issue or submit a pull request. 


# Citation

Users can freely access, modify, and use the code. However, if you find it useful, an acknowledgment is highly appreciated.

# cite:
```
The related paper is currently under submission. Please check back for updates on the citation details once the paper is published.
```

