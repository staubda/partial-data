Experiment Log
--------------
**Run 1**
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
 

**Run 2**
- Model
    - SSD + Mobilenet-v1
- Dataset
    - COCO val with 7 categories
    - Complete labels
- Training
    - 200k steps
    - Default config
    - Run on p2.xlarge AWS spot instance
        - Completed in ~ 3 days, ~ 1 step/sec
    - Classification loss starts increasing after ~20k steps
    - Total loss starts increasing after ~80k steps
- Eval
    - mAP COCO metrics
- Results
    - pretty crappy
 
**Run 3**
- Model
    - SSD + Mobilenet-v1
- Dataset
    - COCO val with 7 categories
    - Partial labels
- Training
    - 200k steps
    - Default config
    - Run on p2.xlarge AWS spot instance
        - Completed in ~ 3 days, ~ 1 step/sec
    - Classification loss starts increasing after ~20k steps
    - Total loss starts increasing after ~80k steps
- Eval
    - mAP COCO metrics
- Results
    - pretty crappy, though slightly better than run2 with complete labels

**Run 4**
- Model
    - SSD + Mobilenet-v1
- Dataset
    - COCO val with 7 categories
    - Increased dataset size compared to run2,3
    - Decreased label overlap between artificial sub-datasets compared to run2,3
    - Partial labels
- Training
    - 200k steps
    - Default config
    - Run on p2.xlarge AWS spot instance
        - Completed in ~ 3 days, ~ 1 step/sec
    - Loss decreased the whole way
- Eval
    - mAP COCO metrics
- Results
    - mAP@0.5 = 0.196

**Run 5**
- Model
    - SSD + Mobilenet-v1
- Dataset
    - COCO val with 7 categories
    - Complete labels
    - Increased dataset size compared to run2,3
    - Decreased label overlap between artificial sub-datasets compared to run2,3
- Training
    - 200k steps
    - Default config
    - Run on p2.xlarge AWS spot instance
        - Completed in ~ 3 days, ~ 1 step/sec
    - Loss decreased the whole way
- Eval
    - mAP COCO metrics
- Results
    - mAP@0.5 = 0.251

**Run 6**
- Model
    - SSD + Mobilenet-v1
- Dataset
    - COCO val with 7 categories
    - Partial labels
    - Increased dataset size compared to run2,3
    - Decreased label overlap between artificial sub-datasets compared to run2,3
- Training
    - Updated loss to support per-example partial label masks. 
    - 200k steps
    - Default config
    - Run on p2.xlarge AWS spot instance
        - Completed in ~ 3 days, ~ 1 step/sec
    - Loss decreased until step 120k
- Eval
    - mAP COCO metrics
- Results
    - mAP@0.5 = 0.201
