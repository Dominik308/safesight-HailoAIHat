class ModelMode:
    """Set styles for Widgets"""


    def set_dropdown_style() -> str:
        return """
            QComboBox {
                background-color: #3a3a3a;
                color: #f8f9fa;
                border: 2px solid #495057;
                border-radius: 8px;
                padding: 10px;
                font-size: 14px;
            }
            QComboBox::drop-down {
                border: none;
                width: 30px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #f8f9fa;
                margin-right: 10px;
            }
            QComboBox QAbstractItemView {
                background-color: #3a3a3a;
                color: #f8f9fa;
                selection-background-color: #495057;
                padding: 8px;
                border: 2px solid #495057;
            }
        """
    
    def determineModeFromModel(modelName) -> str:
        """Determine model mode based on selected model"""
        name_lower = modelName.lower()
        
        if "hand" in name_lower:
            return "hand"
        elif "pose" in name_lower:
            return "pose"
        elif "segment" in name_lower or "seg" in name_lower:
            return "segmentation"
        elif "css" in name_lower:
            return "css"
        else:
            return "detection"
        

    def fillModelModeDropdown(modeltype) -> list:
        """Get mode options based on model type"""
        if modeltype == "detection":
            return ["Mittelpunkt", "Beliebige Überlappung", "Vollständig innerhalb"]
        elif modeltype == "segmentation":
            return ["Mittelpunkt", "Beliebige Überlappung", "Maskenüberlappung 50%"]
        elif modeltype == "pose":
            return ["Beliebiger Keypoint", "Mittelpunkt", "Kopf-Keypoints", "Rumpf-Keypoints"]
        elif modeltype == "hand":
            return ["Beliebiger Keypoint", "Mittelpunkt", "Fingerspitzen", "Handflächen-Keypoints"]
        elif modeltype == "css":
            return ["Mittelpunkt", "Beliebige Überlappung"]
        else:
            return ["Mittelpunkt", "Beliebige Überlappung"]