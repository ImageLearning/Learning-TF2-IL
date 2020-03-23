
call conda create -n TF-GPU pip tensorflow-gpu

call conda activate TF-GPU


call python -m pip install --upgrade pip

call pip install --ignore-installed --upgrade tensorflow-gpu

call conda install -c anaconda protobuf

call pip install pillow

call pip install lxml

call pip install Cython

call pip install contextlib2

call pip install jupyter

call pip install matplotlib

call pip install pandas

call pip install contextlib2

call pip install lxml

call pip install opencv-python

call pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI

call conda activate TF-GPU
