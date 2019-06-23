## Installation and Setup
1. Creating the environment
  * Create a new environment 'test' cloned from the environment 'nk'
  * Checkout the neurodriver branch to feature/code-generation
  * On top of this, run the installation necessary for FlyBrainLab.
2. Inside Neuroballad/neuroballad replace neuroballad.py with the new version.
3. Inside neurodriver/examples/code-generation/Neuropiler, replace neuropile.py, translate.py, blocks.py with their new versions, and add any notebook you want to execute here.
4. cd ~/run_scripts. Use tmux to run processor, nlp, neuroarch and neurokernel
5. cd ffbo/neurodriver/examples/code-generation/Neuropiler and run:
  * jupyter lab --ip 127.0.0.1
  * In a new window, ssh -L 9999:127.0.0.1:8888 -p 25654 e6095@amacrine.ee.columbia.edu
6. Open localhost:9999/lab, start a new launcher, change the kernel of FBLCodeGeneration and run the example notebook.