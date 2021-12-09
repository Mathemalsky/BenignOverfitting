# BenignOverfitting
Seminar of Mathematics and Numerics of Deep Neural Networks for Physical Simulations

## Link to paper
[Bartlett et al.](https://www.pnas.org/content/early/2020/04/22/1907378117)

## Libraries
The implementation uses eigen3 and gnuplot, which depends on boost. So eigen3, gnuplot and boost muss be on the machine to compile the code.

### installing eigen3
- download eigen (e.g. eigen-3.4.0)
- type the following commands to your console:
    - cd eigen-3.4.0/
    - mkdir build
    - cd build
    - cmake ..
    - make blas
    - sudo make install

### installing boost
- sudo apt-get install libboost-all-dev

### installing gnuplot
- sudo apt-get install gnuplot
- sudo apt-get install libgnuplot-iostream-dev

## compiling the project
The project can be compiled with the following commands:
- cd BenignOverfitting/
- mkdir build
- cd build/
- cmake ..
- make
