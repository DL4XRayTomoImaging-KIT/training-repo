# training-repo

Ideally, I plan to use this as a universal training repository.
Each branch is planned to be for a separate task, and after some new features are proven to be good, they are to be merged into the master.
Each time I need a separate training task, I plan to create a branch from master, and move on with experiments.

We will see how this thing unroll, but for now it's the best I can imagine. 
Probably, at some point core trainer could be moved from Catalyst to, say, Fast.ai. 
But for now let roll on with what we have.

To check the work of the initial setup one can run
```shell
nice -n 5 python train.py +dataset=eyes +model=eyes
```
which should train segmentation model for medaka eyes for 10 epochs.
