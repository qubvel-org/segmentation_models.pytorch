FROM anibali/pytorch:cuda-9.0

WORKDIR /tmp/smp/

COPY ./requirements.txt /tmp/smp/requirements.txt
RUN pip install -r requirements.txt
RUN pip install pytest

COPY . /tmp/smp/
RUN pip install .
