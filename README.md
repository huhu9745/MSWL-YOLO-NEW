# MSWL-YOLO-NEW
This project extends and optimizes the YOLO series for object detection by integrating **MASPPF**, **BIFPN**, **WISEIOU**, and **PURN pruning** modules to improve performance.


## Module Descriptions

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
