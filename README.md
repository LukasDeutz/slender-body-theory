#Python package to solve slender-body-theory 

This is a python package which implements slender-body-theory for the Kirchhoff rod [1] and the Cosserat rod [2]. Slender-body-theory approximates the fluid dynamics of a slender bodies immersed in a low Reynlodsnumber flow. It is often used to model the fluid dynamics of slender microorganisms. Slender bodies like the Kirchhoff and Cosserat rod are described by a centreline and a local reference frame which determines the orientation of the cross sections as a function of the centreline position. The main result of Slender-body-theory is an integral expression which establishes a relationship between the velocity of each centreline segment and the drag force line density distribution. The integral expressions can be written as a convolution integral. Here, the drag force line density refers to the friction froce between the rod's surface area and surrounding fluid environemnt. It is a linear function of the centreline velocity and counteracts the centreline's movement.           

For an arbirary (normally uniform) discretization of the centreline, the convolution integral can be approximated as a matrix vector multiplication with the vector being the flattened force line density and the convolution matrix being a function of the current shape of the rod. The convolution matrix is invertible so that the drag force line densiy can be computed as a funcion of the centreline velocity distribution. The matrix representation of the convolution kernel can be used to integrate slender-body-theory into a larger PDE system which describes the dynamics of the Kirchhoff or Cosserat rod model for given set of constituency laws and potential additional external force and torque terms. The PDE systems can be reduced to a matrix problem using one of the standard linearization and discretization methods like e.g. finite elements. Slender-body-theory can then be integrated by incorporating the convolution matrix into the system matrix derived from the PDE system.        
   
#Installation

First, you need a python 3.x enviroment. The only third-party packages required are numpy and scipy. To run the example files matplotlib needs to be installed as well. 

The `slender_body_theory` package can be installed into the active python environment using the setup.py. From the parameter-scan package directory run

```bash
# The e option adds a symoblic link to the python-package library of the active python environment 
pip install -e . 
```

#Testing

Test the package installation by executing 

```bash
cd slender_body_theory
python ./tests/test_slender_body_theory.py 
```

#Usage

The main interface is provided `SBT_Matrix` and `SBT_Integral` classes in the `slender_body_theory.py` module. An example use case can be found in `./examples/semicircle.py`. 


#References

1.  Johnson, R. E. (1980). An improved slender-body theory for Stokes flow.
2. Garg, M., & Kumar, A. (2022). A slender body theory for the motion of special Cosserat filaments in Stokes flow. 



