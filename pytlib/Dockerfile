FROM ubuntu:16.04
RUN apt-get update
RUN apt-get install -y curl
RUN apt-get install -y python-pip 
RUN apt-get install -y python-tk
RUN apt-get install -y graphviz
RUN pip install virtualenv
RUN virtualenv pytenv
RUN pwd > pytenv/lib/python2.7/site-packages/pyenv.pth
RUN rm /bin/sh && ln -s /bin/bash /bin/sh
RUN source pytenv/bin/activate
RUN pip install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp27-cp27mu-manylinux1_x86_64.whl
ADD requirements.txt /
RUN pip install -r requirements.txt
RUN echo 'export PYTHONPATH="${PYTHONPATH}:/rays_repo"' >> ~/.bashrc 

CMD exec /bin/bash -c "trap : TERM INT; sleep infinity & wait"

