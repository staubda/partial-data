Experiment Log
--------------
**Run1**
- Model
    - SSD + Mobilenet-v1
- Dataset
    - COCO val with 7 categories
    - Complete labels
- Training
    - 200k steps
    - Default config
    - Run on laptop CPU
        - Ran for ~1 day, only got through a few thousand steps
- Eval
    - mAP COCO metrics
- Results
    - None, training didn't get far enough
 

**Run2**
- Model
    - SSD + Mobilenet-v1
- Dataset
    - COCO val with 7 categories
    - Complete labels
- Training
    - 200k steps
    - Default config
    - Run on p2.xlarge AWS spot instance
        - Completed in ~ 2 days
- Eval
    - mAP COCO metrics
- Results
    - pretty crappy
 
**Run3**
- Model
    - SSD + Mobilenet-v1
- Dataset
    - COCO val with 7 categories
    - Partial labels
- Training
    - 200k steps
    - Default config
    - Run on p2.xlarge AWS spot instance
        - Completed in ~ 2 days
- Eval
    - mAP COCO metrics
- Results
    - pretty crappy, though slightly better than run2 with complete labels