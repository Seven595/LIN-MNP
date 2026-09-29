import setuptools


setuptools.setup(
    name="loki",  # The name of your package on PyPI
    version="0.0.1",  # Choose your initial release version
    author="Weiqing Chen",
    author_email="wec4005@med.cornell.edu",
    description="The Loki platform offers 5 core functions: tissue alignment, tissue annotation, cell type decomposition, image-transcriptomics retrieval, and ST gene expression prediction",
    packages=setuptools.find_packages(),  # Finds the 'loki' folder automatically
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: BSD 3-Clause License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',  # or the minimum version you support
    install_requires=[
        "anndata==0.10.9",
        "matplotlib==3.9.2",
        "numpy==1.25.0",
        "pandas==2.2.3",
        "opencv-python==4.10.0.84",
        "pycpd==2.0.0",
        "torch==2.3.1",
        "tangram-sc==1.0.4",
        "tqdm==4.66.5",
        "torchvision==0.18.1",
        "open_clip_torch==2.26.1",
        "pillow==10.4.0",
        "ipykernel==6.29.5",
        "ipywidgets==8.1.6",
    ],
)
