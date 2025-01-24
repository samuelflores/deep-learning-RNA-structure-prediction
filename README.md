# RNA tetraloop prediction using machine learning

This is the branch maintained by Xinhe Xing for the Flores Lab RNA structure prediction project.

## RNA tetraloop dataset generation
The `data_generation` folder contains all RNA tetraloop dataset-related code and figures generated for the project. The dataset we generate is modified from https://github.com/sbottaro/RNA_TETRALOOPS.
- `classes.py` and `utils.py` are general utility modules required for all the other scripts
- `generate_fragments.sh` runs the Python scripts `generate_annotated_chains.py` and `generate_fragments.py` with different command-line arguments to generate a repository of `.pickle` files containing tetraloops of varying lengths for downstream analysis (note that the default paths are not necessarily applicable).
- `dataset_statistics.ipynb` contains all the statistical analysis and plotting code performed on the tetraloop dataset. All generated figures are placed into the `figures` folder.
- `training_data_generation.ipynb` (inside of the `training_data` folder) generates training, validation, and testing sets from the aforementioned tetraloop dataset for machine learning purposes

## Neural network
The `tloop_prediction` folder contains all machine learning model-related code and figures generated for the project.
- `ANN.ipynb` is where the neural network itself (TensorFlow) is defined and run. As both the model and dataset are relatively small, the training times are fairly short so a Jupyter notebook format was selected for faster debugging.
- `statistics.ipynb` contains all the model performance evaluation (with the testing sets) and plotting code. All generated figures are placed into the `figures` folder.
