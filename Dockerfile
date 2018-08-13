FROM russelljarvis/neuronunit

#FROM jupyter/base-notebook
ENV QT_QPA_PLATFORM offscreen
USER jovyan
#RUN sudo apt-get install --fix-missing
RUN sudo apt-get -y autoremove
RUN sudo apt -y update
RUN sudo apt -y upgrade 
RUN pip uninstall -y neo
RUN /opt/conda/bin/pip3 install dask psutil natsort pyspike pyNN lazyarray neo matplotlib 

WORKDIR /opt/conda/lib/python3.5/site-packages/pyNN/neuron/nmodl
RUN nrnivmodl
# RUN sudo apt-get install -y g++ python3-dev python3-numpy python3-scipy python3-cffi python3-h5py python3-networkx python3-pyopencl aptitude


#RUN sudo apt-get install python-devel libxslt-devel libffi-devel openssl-devel

# RUN sudo aptitude install -y python3-pyopencl pyopencl
# RUN sudo /opt/conda/bin/pip3 install git+https://github.com/inducer/pyopencl.git
# RUN /opt/conda/bin/pip3 install --upgrade pip
# RUN python3 -c "import pyopencl as cl"

#CPU
# RUN git clone https://github.com/jlizier/jidt
# WORKDIR jidt
# RUN sudo ant build
# RUN sudo apt-get install python-jpype && sudo /opt/conda/bin/pip install JPype1 && sudo /opt/conda/bin/pip install git+https://github.com/pwollstadt/IDTxl.git
# RUN python3 -c "from idtxl.multivariate_te import MultivariateTE; network_analysis = MultivariateTE()"

WORKDIR $HOME
ADD . QIASCOLI
WORKDIR $HOME/QIASCOLI/pickles
WORKDIR $HOME/QIASCOLI
RUN pip install git+https://github.com/HumanBrainProject/hbp-neuromorphic-client.git

RUN sudo chown -R jovyan $HOME
USER jovyan
# RUN python3 forked.py
# RUN python3 sate.py