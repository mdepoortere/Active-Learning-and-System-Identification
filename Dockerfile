From sinzlab/pytorch:dj_v0.12.0
# install third-party libraries
RUN pip3 --no-cache-dir install\
                        gitpython
# installing other developing packages
RUN mkdir -p /notebooks/lib
COPY lib /notebooks/lib
# install ml-utils (in develop mode)
RUN cd lib/ml-utils/ && python3 setup.py develop
RUN cd lib/nnfabrik/ && python3 setup.py develop



