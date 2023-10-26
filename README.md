# STARC reward distances
These experiments empirically examine different ways of measuring distances
between pairs of reward functions and compare their performance. The first version
of the paper can be found [here](https://arxiv.org/abs/2309.15257).


## Repo structure
All the source code is located in the `/src` directory. The files that execute
the individual experiments are located in the `/src/experiments` subdirectory,
with `/src/__main__.py` being a convenient way to execute them from the command line.

There is one exception to this structure — all the experiments involving continuous
environments are located in `/src/continuous`. These are almost entirely disconnected
from the rest of the repo because the data structures are completely different.
