# traffic_light_rl
This is a single traffic signal control system implemented using reinforcement learning. To further improve the training performance of DQN, the gradient-based meta-learning algorithm is integrated (GBML-DQN). It belongs to AutoRL and can adjust the parameter gamma during training. Experiments show that GBML-DQN promotes convergence of the Q-value function and avoids overestimation to some extent, especially in the case of inappropriate training settings, whereas DQN fails in training.

Experiments are designed for comparing the training performance using different algorithms (DQN, GBML-DQN) with different optimizers (SGD, RMSprop). The figures of experiments are as follows. For more figures, please check in folder *record*.
![image](https://github.com/Desny/traffic_light_rl/blob/main/record/exp1.png)

## Quick Start
#### 1. Install SUMO (a traffic simulator)

**Linux:**
[the github repository of SUMO](https://github.com/eclipse/sumo)

Download the code:
```
git clone https://github.com/eclipse/sumo.git
```
Build and installation for Ubuntu:
```
sudo apt-get install cmake python g++ libxerces-c-dev libfox-1.6-dev libgdal-dev libproj-dev libgl2ps-dev swig
cd <SUMO_DIR> # please insert the correct directory name here
export SUMO_HOME="$PWD"
mkdir build/cmake-build && cd build/cmake-build
cmake ../..
make -j$(nproc)
```

**Windows:**
[Download zip or msi](https://sumo.dlr.de/docs/Downloads.php)

Set the environment variable:
```
SUMO_HOME=<SUMO_DIR>
```

#### 2. Check and install the dependencies

```
cd <traffic_light_rl>
pip install -r requirements.txt
```

#### 3. Run code

There are two modes, *train* and *eval*. The mode *train* is for training and *eval* is for testing the performance of the trained agent. You can choose values of your parameters by directly modifying the code (*FLAGS*) in ```main.py```, or running the code with parameters like:
```
# directly append parameters to the console
python main.py -gamma=0.9 -num_episodes=600

# or modifying the config file and append it to the console
python main.py -flagfile=xx.cfg
```

**Train:**
```
cd <traffic_light_rl>
python main.py
```

**Eval:**
```
cd <traffic_light_rl>
python main.py -flagfile=eval.cfg
```
