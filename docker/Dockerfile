FROM ubuntu:16.10

RUN apt-get update && \
    apt install -y software-properties-common && \
    add-apt-repository -y ppa:fkrull/deadsnakes && \
    apt-get install -y python3.6 python3.6-dev curl zip \
                       libffi-dev libssl-dev npm git \
                       python-pip python-pkg-resources python-setuptools \
                       libffi-dev libssl-dev libxml2-utils && \
    rm -rf /var/lib/apt/lists/*

# gsutil is installed to deploy the book to Google Cloud Storage
RUN pip2 install --upgrade pip && pip2 install gsutil

# Make sure Python 3.6 is the default
RUN ln -sf /usr/bin/python3.6 /usr/bin/python && \
    curl https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && \
    python /tmp/get-pip.py

# Install dependencies not included in Debian
RUN pip install --upgrade pip pyzmq setuptools cython numpy scipy \
                          jupyter notedown>=1.4 toolz networkx \
                          xlrd seaborn sympy ipykernel && \
    python -m ipykernel install --user
RUN pip install scikit-image  # Separate step, since numpy, Cython needed

# Install dependencies for HTMLBook build
RUN ln -s /usr/bin/nodejs /usr/bin/node && \
    npm install -g htmlbook

# Download big data files
# (Potential refactor: use Wercker's cache to store data files)
RUN mkdir /data && \
    curl http://hgdownload.cse.ucsc.edu/goldenPath/dm6/bigZips/dm6.fa.gz \
         -o /data/dm6.fa.gz

