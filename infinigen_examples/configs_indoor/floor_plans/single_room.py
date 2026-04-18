"""Predefined single-room floor plans for use with Infinigen.

Pass any of the ``*_<room>`` functions to
``Solver.floor_plan='infinigen_examples.configs_indoor.floor_plans.single_room.<fn>'``
(the dotted form — ``PredefinedFloorPlanSolver`` supports both JSON files and
python module paths). The room is centred on the world origin so that
``generate_indoor_center_table``'s centre-table logic puts the table at
``(0, 0, floor_z)``.

Room names must follow Infinigen's ``{room_type}_{level}/{id}`` schema, with
``room_type`` one of the strings defined in ``infinigen/core/tags.py``
(``living-room``, ``dining-room``, ``bedroom``, ``kitchen``, ``bathroom`` …).
"""

import shapely


def _centred_box(width: float, depth: float) -> shapely.Polygon:
    hw, hd = width / 2, depth / 2
    return shapely.box(-hw, -hd, hw, hd)


def _plan(room_tag: str, width: float, depth: float) -> dict:
    """Single rectangular room with a door and a couple of windows."""
    hw, hd = width / 2, depth / 2
    door_half = min(0.6, hw * 0.3)
    side_win_half = min(0.8, hd * 0.35)
    rear_win_half = min(1.0, hw * 0.4)
    return {
        "rooms": {
            f"{room_tag}_0/0": {"shape": _centred_box(width, depth)},
        },
        "doors": {
            "door": {
                "shape": shapely.LineString(
                    [(-door_half, -hd), (door_half, -hd)]
                ),
            },
        },
        "windows": {
            "window_side": {
                "shape": shapely.LineString(
                    [(hw, -side_win_half), (hw, side_win_half)]
                ),
            },
            "window_rear": {
                "shape": shapely.LineString(
                    [(-rear_win_half, hd), (rear_win_half, hd)]
                ),
            },
        },
    }


def square_living_room(factory_seed):
    """6 m x 6 m square living room."""
    return _plan("living-room", 6.0, 6.0)


def square_dining_room(factory_seed):
    """6 m x 6 m square dining room."""
    return _plan("dining-room", 6.0, 6.0)


def rectangle_living_room(factory_seed):
    """8 m x 5 m rectangular living room."""
    return _plan("living-room", 8.0, 5.0)


def rectangle_dining_room(factory_seed):
    """7 m x 5 m rectangular dining room."""
    return _plan("dining-room", 7.0, 5.0)


def bedroom(factory_seed):
    """5 m x 4 m bedroom."""
    return _plan("bedroom", 5.0, 4.0)


def kitchen(factory_seed):
    """5 m x 4 m kitchen."""
    return _plan("kitchen", 5.0, 4.0)
