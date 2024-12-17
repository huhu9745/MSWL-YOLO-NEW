# MSWL-YOLO-NEW
This project extends and optimizes the YOLO series for object detection by integrating **MASPPF**, **BIFPN**, **WISEIOU**, and **PURN pruning** modules to improve performance.



## Module Descriptions

The VisDrone2019 dataset is located in dataset/VOCdevkit.

1. **MASPPF Module**  
   - **Functionality**: Implements a multi-scale adaptive feature fusion pyramid to improve object detection.  
   - **Location**: `MASPPF.py`

2. **BIFPN Module**  
   - **Functionality**: Optimizes the feature pyramid network using weighted bidirectional feature fusion.  
   - **Location**: `bifpn.py`

3. **WISEIOU Loss**  
   - **Functionality**: Enhances the IoU loss function for better localization accuracy in object detection.  
   - **Location**: `wiseiou.py`

4. **PURN Pruning Module**  
   - **Functionality**: Prunes the YOLO model to reduce computational costs while maintaining performance.  
   - **Location**: `ultralytics-prune/`
  
   - 
lamp
The LAMP pruning is located in ultralytics-prune/compress.py.

    'model': 'runs/train/yolov10n-visdrone/weights/best.pt',    # Here, you need to specify the weights of the model that was trained earlier.
    'data':'/home/hjj/Desktop/dataset/dataset_visdrone/data.yaml',
    'imgsz': 640,
    'epochs': 200,
    'batch': 32,
    'workers': 4,
    'cache': True,
    'optimizer': 'SGD',
    'device': '0',
    'close_mosaic': 0,
    'project':'runs/prune',
    'name':'yolov10n-visdrone-lamp-exp2',
    
    # prune
    'prune_method':'lamp',
    'global_pruning': True,
    'speed_up': 2.0,                  #  Here, you need to select the pruning **speed**.
    'reg': 0.0005,
    'sl_epochs': 500,
    'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
    'sl_model': None,
}
run compress.py
