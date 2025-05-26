# Project Structure

## Directory Structure

- **data/**  
  - Contains data processing code  
  *(Note: Requires modification of original data path)*

- **FMD_code/**  
  - Contains MATLAB code for FMD method

- **model/**  
  - Stores model files  
  *(Note: Requires modification of input tensor's first dimension and number of categories)*

- **plot/**  
  - Contains plotting code  
  *(Note: Requires modification of data path and weight saving path)*

## Key Files

- **train.py**  
  - Main training program  
  *(Important: Ensure data path consistency with other scripts)*

## Implementation Notes
1. Path configurations need to be consistent across:
   - Data folder processing scripts
   - Plotting scripts
   - Main training program (train.py)

2. Model adjustments required for:
   - Input tensor dimensions
   - Number of classification categories
