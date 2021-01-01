#! /usr/bin/env sh
ENV="conda"
if [ ${ENV} == "conda" ]; then
    conda env create -f environment.yml;
    conda activate artist-gen;
else
    pip install -r requirements.txt;
fi
