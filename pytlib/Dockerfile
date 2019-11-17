# this is currently broken, untested
FROM ubuntu:16.04
RUN apt-get update
RUN apt-get install -y curl
RUN apt-get install -y python3-pip 
RUN apt-get install -y python3-tk
RUN apt-get install -y graphviz
RUN pip3 install pipenv
RUN rm /bin/sh && ln -s /bin/bash /bin/sh
ADD Pipfile /
RUN pip3 install Cython
RUN pipenv install
RUN echo 'export PYTHONPATH="${PYTHONPATH}:/pytlib"' >> ~/.bashrc 

CMD exec /bin/bash -c "trap : TERM INT; sleep infinity & wait"

