# Run Fenics Jobs on Docker

Tested by chtung 04/24/2022 on Docker Desktop 4.6.1

* ## For the first time
  
  ### Create Docker container
  
        docker run --name notebook -w /home/fenics -v ${pwd}:/home/fenics/shared -d -p 127.0.0.1:8888:8888 quay.io/fenicsproject/stable 'jupyter-notebook --ip=0.0.0.0'
        start "http://localhost:8888"
  
  ### Get the login token for jupyter-notebook
  
        docker logs notebook

* ## Start the created Docker container
  
        docker start notebook
        start "http://localhost:8888"

* ## Start a new terminal in jupyter-notebook and install the required packages
  
        sudo apt install gmsh
        pip install pygmsh==6.1.1
        pip install git+https://github.com/dolfin-adjoint/pyadjoint.git@2019.1.0
        pip install h5py
        pip install tqdm
  
  ### Alternatively, SSH into your Docker container
  
      docker exec -it notebook /bin/bash

* ## Run your scripts!

* ## Stop the created Docker container
  
        docker stop notebook