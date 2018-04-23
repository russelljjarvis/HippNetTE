FROM russelljarvis/neuronunit

#FROM jupyter/base-notebook
ENV QT_QPA_PLATFORM offscreen
USER jovyan
#RUN sudo apt-get install --fix-missing && sudo apt-get install -f && sudo apt-get autoremove && sudo apt update && sudo apt upgrade 
RUN pip uninstall -y neo
RUN /opt/conda/bin/pip3 install dask psutil natsort pyspike pyNN lazyarray neo matplotlib pyopencl

WORKDIR /opt/conda/lib/python3.5/site-packages/pyNN/neuron/nmodl
RUN nrnivmodl


#RUN sudo apt-get install python-devel libxslt-devel libffi-devel openssl-devel

RUN sudo apt-get install -y g++ python3-dev python3-numpy python3-scipy python3-cffi python3-h5py python3-networkx python3-pyopencl aptitude
RUN sudo aptitude install -y python3-pyopencl pyopencl
RUN sudo /opt/conda/bin/pip3 install git+https://github.com/inducer/pyopencl.git
RUN /opt/conda/bin/pip3 install --upgrade pip
RUN python3 -c "import pyopencl as cl"

#CPU
RUN git clone https://github.com/jlizier/jidt
WORKDIR jidt
RUN sudo ant build
RUN sudo apt-get install python-jpype && sudo /opt/conda/bin/pip install JPype1 && sudo /opt/conda/bin/pip install git+https://github.com/pwollstadt/IDTxl.git
RUN python3 -c "from idtxl.multivariate_te import MultivariateTE; network_analysis = MultivariateTE()"
WORKDIR $HOME/QIASCOLI
RUN sudo chown -R jovyan $HOME
USER jovyan
