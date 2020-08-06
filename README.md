# Multi-agent Tic-Tac-Toe using RLLib

In this repository I create a multi-agent Tic-Tac-Toe environment that supports the integration with Ray's Reinforcement Learning agents. Furthermore, a few of the Reinforcement Learning algorithms that are supported by Ray are trained by playing against a heurisitc policy. The details of this heuristic and the results of the training can be found below. 


## The heuristic

The heuristic a greedy heuristic that simply goes over the fields of the Tic-Tac-Toe board and places its next move on the first empty field it can find. It iterates the board row by row, from left to right. Below 2 examples of this can be found.

1. 

| X | O | X |
|---|---|---|
| _ | _ | _ |
| _ | _ | _ |

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&darr;
   
| X | O | X |
|---|---|---|
| O | _ | _ |
| _ | _ | _ |

2. 

| X | O | X |
|---|---|---|
| O | X | _ |
| _ | _ | _ |

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&darr;
   
| X | O | X |
|---|---|---|
| O | X | O |
| _ | _ | _ |



## The results  :chart_with_upwards_trend:

Below the results of various Reinforcement Learning Algorithms playing against the heuristic can be found. Here the PPO, PG and DQN seem to converge to the optimum score per episode of 10, although the DQN seems to be much more unstable than the PPO and PG. The A3C's performance against the heuristic is increasing during the training but was not able to get better than the heuristic within the first 10K episode. 

<img width="800" src="/results/PG.png">
<img width="800" src="/results/A3C.png">
<img width="800" src="/results/DQN.png">
<img width="800" src="/results/PPO.png">
<img width="800" src="/results/ALL.png">






