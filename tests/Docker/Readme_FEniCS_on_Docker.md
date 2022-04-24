# Run Fenics Jobs on Docker
## For the first time
### Create Docker container
docker run --name notebook -w /home/fenics -v ${pwd}:/home/fenics/shared -d -p 127.0.0.1:8888:8888 quay.io/fenicsproject/stable 'jupyter-notebook --ip=0.0.0.0'
start "http://localhost:8888"
### Get the login token for jupyter-notebook
docker logs notebook # to get the login token for jupyter-notebook

## Start the created Docker container
docker start notebook
start "http://localhost:8888"

## Start a new terminal in jupyter-notebook and install the required packages
sudo apt install gmsh
pip install pygmsh==6.1.1
pip install git+https://github.com/dolfin-adjoint/pyadjoint.git@2019.1.0
pip install h5py
pip install tqdm

## Run your scripts!

## Stop the created Docker container
docker stop notebook