# confidentExplanations

## What metrics do we need for evaluation?

Using always AUC with soft probabilities and accuracy

- Performance when not to predict. Expected behaviour, that in 2d should be very easy to draw straight lines. So AUC should be 1, if the dataset is more complicated then it should go below 1.

- Actual coverage @90. The best method should hold a straight line @90
  
- Performance of model F when abstentions are done of model G. We should see if we increase coverage that the performance of F increases.

The function should be something like

```python
def calculate_metrics(detector,detector_base,plugin,data):
    return 3Evaluations with 2Metrics = 6 values
```