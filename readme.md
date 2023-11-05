### 1
pip install tensorflow
pip install keras
pip install --no-deps argument git+https://github.com/haven-jeon/PyKoSpacing.git

### 2
pip install -U sentence-transformers


### 3
pip install rank_bm25

### 4
pip install -r requirements.txt


install tensorflow for mac m1
1. Install Miniforge3 for mac
curl -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
chmod +x Miniforge3-MacOSX-arm64.sh
sh Miniforge3-MacOSX-arm64.sh

2. Activate Miniforge3 virtualenv, You should use Python version 3.10 or less.
source ~/miniforge3/bin/activate

3. Install the Tensorflow dependencies 
conda install -c apple tensorflow-deps 

4. Install base tensorflow 
python -m pip install tensorflow-macos 
5. Install metal plugin 
python -m pip install tensorflow-metal
