# Proposed Fix for matterport_processor.py 
# This would be the proper way to fix it at the source

# Instead of:
bbox_9dof['category'] = f"class_{box_data['label']}"

# We should do:
from utils.class_mapping import get_class_name
bbox_9dof['category'] = get_class_name(int(box_data['label']))
bbox_9dof['original_label_id'] = int(box_data['label'])

# This would produce processed data with semantic names from the start:
# "category": "hair_dryer" instead of "category": "class_84"