FROM anibali/pytorch:1.5.0-nocuda

WORKDIR /tmp/smp/

COPY ./requirements.txt /tmp/smp/requirements.txt
RUN pip install -r requirements.txt
RUN pip install pytest mock

COPY . /tmp/smp/
RUN pip install .
