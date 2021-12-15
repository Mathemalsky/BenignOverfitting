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
This last step may take a while because eigen is a header only library which needs to be compiled again every time.

## running the program
After the program is compiled. There will be a `bin/` directory with the binary file in it.
The mnist example needs the mnist dataset, which can be downloaded [here](https://deepai.org/dataset/mnist).
The extractet foler `mnist/` has be in the `bin/` directory and the files whithin it must be unpacked too.

The simple polynomial regression can be invoked with:

`./BeningOverfitting <n> <k> <mu>` with
- `<n>` number of data points
- `<k>` number of degrees of freedom
- `<mu>` weight for the regularization penalty.

The mnist example can be invoked with:
`./BeningOverfitting mnist <n> <t> <mu> [-d]` with:
- `<n>` number of images used for training
- `<t>` number of images used for estimating the prediction accuracy.
- `<mu>` weight for regularization penalty
- `-d` optional argument to suppress output and add the accuracy to a file `accuracy.dat`

A plot for a estimated density of accuracy can be obtained by calling
`./BenignOverfitting density <filename> <h>`
- `<filename>` name of the file to be read
- `<h>` band width that is affected by a data point
