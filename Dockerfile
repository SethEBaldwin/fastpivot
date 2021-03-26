FROM quay.io/pypa/manylinux_2_24_x86_64
WORKDIR /usr/src/app
COPY . .
RUN auditwheel repair dist/fastpivot-0.1.7-cp37-cp37m-linux_x86_64.whl