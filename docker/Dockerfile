FROM ubuntu:16.10

RUN apt-get update
RUN apt-get -y upgrade

RUN apt-get install -y python3-pip python3-numpy python3-scipy \
                       python3-matplotlib python3-pandas python3-sklearn \
                       python3-astropy python3-venv python3-zmq \
                       curl zip

# gsutil is installed to deploy the book to Google Cloud Storage
RUN apt-get install -y python-pip python-pkg-resources python-setuptools \
                       libffi-dev libssl-dev

# Install dependencies for HTMLBook build
RUN apt-get install -y npm
RUN npm install -g htmlbook

RUN pip2 install gsutil

# Make sure Python 3 is the default
RUN ln -sf /usr/bin/python3 /usr/bin/python
RUN ln -sf /usr/bin/pip3 /usr/bin/pip

# Install dependencies not included in Debian
RUN pip install setuptools
RUN pip install jupyter notedown>=1.4 scikit-image toolz networkx \
                xlrd seaborn sympy
RUN ipython3 kernel install

# Download big data files
# (Potential refactor: use Wercker's cache to store data files)
RUN mkdir /data
RUN curl http://hgdownload.cse.ucsc.edu/goldenPath/dm6/bigZips/dm6.fa.gz \
         -o /data/dm6.fa.gz

