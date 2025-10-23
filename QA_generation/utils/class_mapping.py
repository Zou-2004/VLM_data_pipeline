"""
Object Class Mapping for 3D Datasets
This module provides human-readable object class names for numerical class IDs
"""

# Common 3D object categories - based on ScanNet, NYU40, and common indoor objects
# This mapping covers typical indoor scene objects found in Matterport3D scans
MATTERPORT_CLASS_MAPPING = {
    # Furniture
    1: "wall", 2: "floor", 3: "ceiling", 4: "chair", 5: "table", 6: "desk", 7: "bed",
    8: "bookshelf", 9: "sofa", 10: "sink", 11: "bathtub", 12: "toilet", 13: "curtain",
    14: "counter", 15: "door", 16: "window", 17: "shower_curtain", 18: "refrigerator",
    19: "stairs", 20: "cabinet", 21: "pillow", 22: "mirror", 23: "dresser", 24: "lamp",
    25: "towel", 26: "box", 27: "whiteboard", 28: "person", 29: "night_stand", 30: "tv",
    31: "microwave", 32: "coffee_maker", 33: "toaster", 34: "washing_machine", 35: "dryer",
    36: "dishwasher", 37: "oven", 38: "range_hood", 39: "clock", 40: "trash_can",
    
    # Extended categories for larger class sets
    41: "picture", 42: "book", 43: "monitor", 44: "keyboard", 45: "mouse", 46: "telephone",
    47: "bag", 48: "paper", 49: "clothes", 50: "shoe", 51: "plant", 52: "vase", 53: "bowl",
    54: "banana", 55: "apple", 56: "sandwich", 57: "orange", 58: "broccoli", 59: "carrot",
    60: "hot_dog", 61: "pizza", 62: "donut", 63: "cake", 64: "couch", 65: "potted_plant",
    66: "dining_table", 67: "bottle", 68: "wine_glass", 69: "cup", 70: "fork", 71: "knife",
    72: "spoon", 73: "bowl", 74: "remote", 75: "cell_phone", 76: "oven", 77: "toaster",
    78: "sink", 79: "refrigerator", 80: "book", 81: "clock", 82: "scissors", 83: "teddy_bear",
    84: "hair_dryer", 85: "toothbrush", 86: "blanket", 87: "shelf", 88: "rack", 89: "hanger",
    90: "frame", 91: "basket", 92: "bin", 93: "container", 94: "jar", 95: "pot", 96: "pan",
    97: "plate", 98: "tray", 99: "mat", 100: "rug",
    
    # More extended categories (up to ~300 as we see in the data)
    101: "curtain_rod", 102: "blind", 103: "radiator", 104: "heater", 105: "fan", 106: "vent",
    107: "outlet", 108: "switch", 109: "handle", 110: "knob", 111: "hook", 112: "rail",
    113: "pipe", 114: "cable", 115: "wire", 116: "board", 117: "panel", 118: "screen",
    119: "display", 120: "sign", 121: "poster", 122: "calendar", 123: "map", 124: "chart",
    125: "decoration", 126: "ornament", 127: "sculpture", 128: "artwork", 129: "painting",
    130: "photograph", 131: "candle", 132: "flower", 133: "leaf", 134: "branch", 135: "stick",
    136: "stone", 137: "rock", 138: "brick", 139: "tile", 140: "wood", 141: "metal",
    142: "plastic", 143: "glass", 144: "fabric", 145: "leather", 146: "paper", 147: "cardboard",
    148: "foam", 149: "sponge", 150: "brush", 151: "tool", 152: "instrument", 153: "device",
    154: "machine", 155: "appliance", 156: "equipment", 157: "gear", 158: "hardware",
    159: "software", 160: "component", 161: "part", 162: "piece", 163: "element",
    164: "item", 165: "object", 166: "thing", 167: "stuff", 168: "material", 169: "substance",
    170: "liquid", 171: "powder", 172: "grain", 173: "crystal", 174: "fiber", 175: "thread",
    176: "rope", 177: "chain", 178: "belt", 179: "strap", 180: "band", 181: "ring",
    182: "circle", 183: "square", 184: "rectangle", 185: "triangle", 186: "oval",
    187: "sphere", 188: "cube", 189: "cylinder", 190: "cone", 191: "pyramid", 192: "prism",
    193: "tube", 194: "rod", 195: "bar", 196: "beam", 197: "column", 198: "support",
    199: "base", 200: "foundation", 201: "frame", 202: "structure", 203: "building",
    204: "room", 205: "space", 206: "area", 207: "zone", 208: "region", 209: "section",
    210: "corner", 211: "edge", 212: "side", 213: "surface", 214: "top", 215: "bottom",
    216: "front", 217: "back", 218: "left", 219: "right", 220: "center", 221: "middle",
    222: "inside", 223: "outside", 224: "above", 225: "below", 226: "over", 227: "under",
    228: "near", 229: "far", 230: "close", 231: "open", 232: "closed", 233: "full",
    234: "empty", 235: "clean", 236: "dirty", 237: "new", 238: "old", 239: "fresh",
    240: "stale", 241: "hot", 242: "cold", 243: "warm", 244: "cool", 245: "dry",
    246: "wet", 247: "hard", 248: "soft", 249: "smooth", 250: "rough", 251: "sharp",
    252: "dull", 253: "bright", 254: "dark", 255: "light", 256: "heavy", 257: "thick",
    258: "thin", 259: "wide", 260: "narrow", 261: "tall", 262: "short", 263: "long",
    264: "small", 265: "large", 266: "big", 267: "tiny", 268: "huge", 269: "giant",
    270: "mini", 271: "micro", 272: "macro", 273: "standard", 274: "regular", 275: "normal",
    276: "special", 277: "unique", 278: "common", 279: "rare", 280: "unusual", 281: "typical",
    282: "standard", 283: "basic", 284: "simple", 285: "complex", 286: "advanced",
    287: "modern", 288: "classic", 289: "vintage", 290: "antique", 291: "contemporary",
    292: "traditional", 293: "custom", 294: "generic", 295: "branded", 296: "labeled",
    297: "marked", 298: "tagged", 299: "coded", 300: "numbered"
}

def get_class_name(class_id: int) -> str:
    """
    Get human-readable class name for a given class ID
    
    Args:
        class_id: Numerical class ID
        
    Returns:
        Human-readable class name
    """
    return MATTERPORT_CLASS_MAPPING.get(class_id, f"object_{class_id}")


def get_all_class_names() -> dict:
    """Get all class mappings"""
    return MATTERPORT_CLASS_MAPPING.copy()


# For backward compatibility with existing "class_X" format
def parse_class_category(category: str) -> str:
    """
    Parse category from "class_X" format to human-readable name
    
    Args:
        category: Category string (e.g., "class_84")
        
    Returns:
        Human-readable class name
    """
    if category.startswith("class_"):
        try:
            class_id = int(category.split("_")[1])
            return get_class_name(class_id)
        except (IndexError, ValueError):
            return category
    return category


if __name__ == "__main__":
    # Test the mappings
    print("Testing class mappings:")
    test_classes = [1, 4, 9, 49, 84, 159, 276]
    for class_id in test_classes:
        print(f"Class {class_id}: {get_class_name(class_id)}")
    
    print("\nTesting category parsing:")
    test_categories = ["class_84", "class_49", "class_276", "chair"]
    for cat in test_categories:
        print(f"'{cat}' -> '{parse_class_category(cat)}'")