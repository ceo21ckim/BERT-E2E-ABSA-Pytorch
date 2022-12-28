FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime


RUN pip install torch -f https://data.pyg.org/whl/torch.1.10.0+cu113.html \ 
    && pip install transformers==4.24.0 && pip install matplotlib \ 
    && pip install jupyter && pip install pandas && pip install numpy \
    && pip install dgl-cu113 dglgo -f https://data.dgl.ai/wheels/repo.html \ 
    && conda install -n base ipykernel --update-deps --force-reinstall

WORKDIR /workspace

CMD ["bash"]
