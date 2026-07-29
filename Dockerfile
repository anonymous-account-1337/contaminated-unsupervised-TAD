FROM debian:trixie
USER root
WORKDIR /python
# compile specific python version, because the default docker image python:3.8.10 uses Debian Buster which is discontinued
ENV PYTHON_VERSION="3.14.4"
RUN apt update && \
    apt install -y vim curl build-essential gdb lcov pkg-config libbz2-dev libffi-dev libgdbm-dev libgdbm-compat-dev liblzma-dev libncurses5-dev libreadline6-dev libsqlite3-dev libssl-dev lzma liblzma-dev tk-dev uuid-dev zlib1g-dev libmpdec-dev libzstd-dev inetutils-inetd && \
    curl https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz --output py.tgz && \
    tar -xvzf py.tgz && \
    cd Python-${PYTHON_VERSION} && \
    ./configure --with-pydebug && \
    make -s -j $(nproc) && \
    ln -f -s /python/Python-${PYTHON_VERSION}/python /bin/python && \
    ln -f -s /python/Python-${PYTHON_VERSION}/python /bin/python3 && \
    python3 -m ensurepip --upgrade && \
    python3 -m pip install -U pip wheel setuptools && \
    apt-get clean && \
    echo 'set mouse-=a' > /root/.vimrc
WORKDIR /app
ENV PYTHONPATH="/app"
COPY requirements.txt .
RUN apt update && \
    python3 -m pip install -U pip wheel setuptools && \
    python3 -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu129 && \
    python3 -m pip install -r requirements.txt
COPY . .
CMD ["/bin/bash", "-l"]