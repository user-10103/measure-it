"""
COCO category + edge-type constants shared across the RGB pipeline.

Single source of truth for the instance-segmentation category ids produced by
the annotation side (Label Studio / SAM3 / LiDAR edge derivation) and consumed
by segmentation, edge classification, evaluation, and the report generator.
"""

# COCO category ids (match the annotation export, adresses cell39)
CAT_ROOF = 1        # roof outline polygon
CAT_FACET = 2       # roof-plane facet polygon
CAT_FO_START = 3    # 3-7 foreign objects (solar/AC/skylight/chimney/vent) - out of scope
CAT_EAVE = 8
CAT_RAKE = 9
CAT_RIDGE = 10
CAT_VALLEY = 11
CAT_HIP = 12

# edge category id -> canonical edge-type name
EDGE_CAT_TO_NAME = {
    CAT_EAVE: "eave",
    CAT_RAKE: "rake",
    CAT_RIDGE: "ridge",
    CAT_VALLEY: "valley",
    CAT_HIP: "hip",
}
EDGE_NAME_TO_CAT = {v: k for k, v in EDGE_CAT_TO_NAME.items()}

# geometric edge types the pipeline derives from facet geometry
GEOMETRIC_EDGE_TYPES = ("eave", "rake", "ridge", "valley", "hip")

# full set of report edge types (the rest are wall/adjacency edges added later/manually)
REPORT_EDGE_TYPES = (
    "eave", "ridge", "hip", "valley", "rake",
    "step_flashing", "wall_flashing", "transition", "parapet", "unspecified",
)

# diagram colours (RGB) per edge type, roughly matching the Roofr report
EDGE_COLORS = {
    "eave": (60, 160, 90),
    "ridge": (150, 210, 150),
    "hip": (150, 90, 200),
    "valley": (220, 90, 60),
    "rake": (235, 180, 40),
    "step_flashing": (120, 90, 60),
    "wall_flashing": (70, 120, 220),
    "transition": (230, 90, 200),
    "parapet": (235, 150, 40),
    "unspecified": (90, 200, 220),
}

# pitch-class label (Label Studio) -> representative slope in degrees (bin midpoint)
PITCH_LABEL_TO_SLOPE = {
    "flat": 2.5,
    "low": 10.0,
    "medium": 22.5,
    "steep": 45.0,
}
