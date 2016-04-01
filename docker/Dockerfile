FROM ubuntu:16.04

RUN apt-get update
RUN apt-get -y upgrade

RUN apt-get install -y python3-pip python3-numpy python3-scipy \
                       python3-matplotlib python3-pandas python3-sklearn \
                       python3-astropy python3-venv python3-zmq \
                       curl

RUN ln -s /usr/bin/python3 /usr/bin/python
RUN ln -s /usr/bin/pip3 /usr/bin/pip
RUN pip install setuptools
RUN pip install jupyter notedown>=1.4 scikit-image toolz networkx \
                xlrd seaborn sympy
RUN ipython3 kernel install

RUN mkdir /data
RUN curl --remote-name \
    http://hgdownload.cse.ucsc.edu/goldenPath/dm6/bigZips/dm6.fa.gz \
    -o /data/dm6.fa.gz

