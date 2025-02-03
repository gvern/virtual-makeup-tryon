# src/makeup_config.py

from collections import namedtuple

# Define a namedtuple for makeup type configurations
MakeupTypeConfig = namedtuple('MakeupTypeConfig', [
    'name',
    'facemesh_regions',
    'default_color',
    'default_intensity'
])

# Define configurations for each makeup type
MAKEUP_TYPES_CONFIG = [
    # --- Lipstick Upper ---
    MakeupTypeConfig(
        name='Lipstick Upper',
        facemesh_regions={
            'upper_lip': frozenset([
                (61, 185),
                (185, 40),
                (40, 39),
                (39, 37),
                (37, 0),
                (0, 267),
                (267, 269),
                (269, 270),
                (270, 409),
                (409, 291),
                (291, 308),
                (308, 415),
                (415, 310),
                (310, 312),
                (312, 13),
                (13, 82),
                (82, 81),
                (81, 80),
                (80, 191),
                (191, 78)
            ])
        },
        default_color=(0, 0, 255),  # Red (BGR)
        default_intensity=0.2
    ),
    # --- Lipstick Lower ---
    MakeupTypeConfig(
        name='Lipstick Lower',
        facemesh_regions={
            'lower_lip': frozenset([
                (61, 146),
                (146, 91),
                (91, 181),
                (181, 84),
                (84, 17),
                (17, 314),
                (314, 405),
                (405, 321),
                (321, 375),
                (375, 291),
                (291, 308),
                (308, 324),
                (324, 402),
                (402, 317),
                (317, 14),
                (14, 87),
                (87, 178),
                (178, 88),
                (88, 95),
                (95, 78),
                (78, 61)
            ])
        },
        default_color=(0, 0, 255),  # Red (BGR)
        default_intensity=0.2
    ),
    # --- Blush (unchanged) ---
    MakeupTypeConfig(
        name='Blush',
        facemesh_regions={
            'left_blush': frozenset([
                (50, 205),
                (205, 187),
                (187, 123),
                (123, 117),
                (117, 118),
                (118, 101),
                (101, 36),
                (36, 205)
            ]),
            'right_blush': frozenset([
                (280, 330),
                (330, 347),
                (347, 346),
                (346, 352),
                (352, 411),
                (411, 425),
                (425, 266)
            ])
        },
        default_color=(255, 0, 0),  # Blue (BGR)
        default_intensity=0.2
    ),
    # --- Eyebrow (unchanged) ---
    MakeupTypeConfig(
        name='Eyebrow',
        facemesh_regions={
            'left_eyebrow': frozenset([
                (276, 283),
                (283, 282),
                (282, 295),
                (295, 285),
                (300, 293),
                (293, 334),
                (334, 296),
                (296, 336)
            ]),
            'right_eyebrow': frozenset([
                (46, 53),
                (53, 52),
                (52, 65),
                (65, 55),
                (70, 63),
                (63, 105),
                (105, 66),
                (66, 107)
            ])
        },
        default_color=(0, 255, 0),  # Green (BGR)
        default_intensity=0.25
    ),
    # --- Foundation (unchanged) ---
    MakeupTypeConfig(
        name='Foundation',
        facemesh_regions={
            'face': frozenset([
                (10, 338),
                (338, 297),
                (297, 332),
                (332, 284),
                (284, 251),
                (251, 389),
                (389, 454),
                (454, 323),
                (323, 361),
                (361, 288),
                (288, 397),
                (397, 365),
                (365, 379),
                (379, 378),
                (378, 400),
                (400, 377),
                (377, 152),
                (152, 148),
                (148, 176),
                (176, 149),
                (149, 150),
                (150, 136),
                (136, 172),
                (172, 58),
                (58, 132),
                (132, 93),
                (93, 234),
                (234, 127),
                (127, 162),
                (162, 21),
                (21, 54),
                (54, 103),
                (103, 67),
                (67, 109),
                (109, 10)
            ])
        },
        default_color=(128, 128, 128),  # Gray (BGR)
        default_intensity=0.2
    ),
    # --- Eyeliner Left ---
    MakeupTypeConfig(
        name='Eyeliner Left',
        facemesh_regions={
            'eyeliner': frozenset([
                (243, 112),
                (112, 26),
                (26, 22),
                (22, 23),
                (23, 24),
                (24, 110),
                (110, 25),
                (25, 226),
                (226, 130),
                (130, 33),
                (33, 7),
                (7, 163),
                (163, 144),
                (144, 145),
                (145, 153),
                (153, 154),
                (154, 155),
                (155, 133),
                (133, 243)
            ])
        },
        default_color=(0, 0, 0),  # Black (BGR)
        default_intensity=0.5
    ),
    # --- Eyeliner Right ---
    MakeupTypeConfig(
        name='Eyeliner Right',
        facemesh_regions={
            'eyeliner': frozenset([
                (463, 362),
                (362, 382),
                (382, 381),
                (381, 380),
                (380, 374),
                (374, 373),
                (373, 390),
                (390, 249),
                (249, 263),
                (263, 359),
                (359, 446),
                (446, 255),
                (255, 339),
                (339, 254),
                (254, 253),
                (253, 252),
                (252, 256),
                (256, 341),
                (341, 463)
            ])
        },
        default_color=(0, 0, 0),  # Black (BGR)
        default_intensity=0.5
    )
]
