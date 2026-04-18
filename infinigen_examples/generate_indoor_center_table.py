"""Generate an Infinigen indoor scene with a dining table fixed at the room centre.

This is a thin wrapper around ``infinigen_examples.generate_indoors``:

1. Run the normal Infinigen indoor pipeline (rooms + furniture + lighting).
2. Spawn a ``TableDiningFactory`` at the house's xy-centre (post-solve).
3. Delete any solved furniture whose footprint overlaps the table to avoid
   obvious intersections.
4. Re-pose the primary camera rig so that it looks at the centre table,
   guaranteeing the table is visible in the final render.

By default the pipeline produces **a single room with full Infinigen
furniture**: ``singleroom.gin`` caps the solver at 1 room, ``--room_type``
fixes the room type so its furniture constraints always fire, and we do NOT
pass ``no_objects.gin`` so the solver keeps placing assets.

Typical usage (run inside Blender's bundled python, same as ``generate_indoors``):

Scene generation (cheap host, no GPU required for ``coarse``):

    python -m infinigen_examples.generate_indoor_center_table \
        --output_folder outputs/center_00 --seed 0 --task coarse \
        --room_type LivingRoom

Use a predefined floor plan (a simple 6x6 m square living room) instead of
Infinigen's random layout:

    python -m infinigen_examples.generate_indoor_center_table \
        --output_folder outputs/center_00 --seed 0 --task coarse \
        --room_type LivingRoom \
        --floor_plan infinigen_examples.configs_indoor.floor_plans.single_room.square_living_room

Rendering (run on a GPU box with the same repo checkout + scene.blend):

    python -m infinigen_examples.generate_indoor_center_table \
        --input_folder outputs/center_00 --output_folder outputs/center_00/frames \
        --seed 0 --task render

Any of the hook parameters can be overridden via gin ``-p`` flags,
e.g. ``-p place_center_table.clear_radius_xy=1.2``.
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(
    format="[%(asctime)s.%(msecs)03d] [%(module)s] [%(levelname)s] | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

# ruff: noqa: E402
import bpy
import gin
import numpy as np
from mathutils import Vector

from infinigen.assets.objects.tables import TableDiningFactory
from infinigen.core import execute_tasks, init
from infinigen.core.util import blender as butil
from infinigen.core.util.math import FixedSeed

from . import generate_indoors  # noqa: F401 — register gin configurables
from .generate_indoors import compose_indoors

logger = logging.getLogger(__name__)


# --- helpers ---------------------------------------------------------------


_CENTER_TABLE_NAME = "center_table"


def _world_bbox(obj: bpy.types.Object) -> tuple[np.ndarray, np.ndarray]:
    corners = np.array(
        [list(obj.matrix_world @ Vector(list(c))) for c in obj.bound_box]
    )
    return corners.min(axis=0), corners.max(axis=0)


def _objs_world_bbox(
    objs: list[bpy.types.Object],
) -> tuple[np.ndarray, np.ndarray]:
    mins, maxs = [], []
    for o in objs:
        if o.type != "MESH" or len(o.data.vertices) == 0:
            continue
        mn, mx = _world_bbox(o)
        mins.append(mn)
        maxs.append(mx)
    if not mins:
        return np.zeros(3), np.zeros(3)
    return np.stack(mins).min(axis=0), np.stack(maxs).max(axis=0)


def _descendants(obj: bpy.types.Object) -> list[bpy.types.Object]:
    """``obj`` and every descendant in the Blender parent hierarchy."""
    out = [obj]
    stack = list(obj.children)
    while stack:
        o = stack.pop()
        out.append(o)
        stack.extend(o.children)
    return out


def _xy_overlap(a_min, a_max, b_min, b_max, margin: float) -> bool:
    return (
        a_min[0] < b_max[0] + margin
        and a_max[0] > b_min[0] - margin
        and a_min[1] < b_max[1] + margin
        and a_max[1] > b_min[1] - margin
    )


def _collect_solved_asset_meshes() -> list[bpy.types.Object]:
    """Meshes produced by the constraint solver (furniture, decor, …).

    Everything grouped under the ``unique_assets`` collection tree, minus
    room-structure meshes (wall/floor/ceiling/exterior) which we do not
    want to delete.
    """
    out: list[bpy.types.Object] = []
    col = bpy.data.collections.get("unique_assets")
    if col is None:
        logger.warning(
            "`unique_assets` collection not found — no furniture to deconflict"
        )
        return out
    room_substrings = (".floor", ".wall", ".ceiling", ".exterior", "room_meshes")
    for obj in col.all_objects:
        if obj.type != "MESH":
            continue
        name_lower = obj.name.lower()
        if any(s in name_lower for s in room_substrings):
            continue
        out.append(obj)
    return out


def _clear_conflicting_furniture(
    table_bbox_min: np.ndarray,
    table_bbox_max: np.ndarray,
    margin: float,
    table_top_margin: float = 0.15,
    exclude: set[bpy.types.Object] | None = None,
) -> int:
    """Delete any solved furniture whose xy footprint intersects the table.

    Objects sitting *on top* of the table (e.g. future decorations we add
    ourselves) are kept — we only prune things whose base is below the
    table top.

    ``exclude`` is the set of objects belonging to the centre table itself —
    without it, the table and every mesh it is made of (top, legs, …) would
    be deleted because they of course overlap their own bbox. We also guard
    against accidental name collisions by skipping anything whose name
    starts with ``center_table``.
    """
    exclude = exclude or set()
    max_keep_z = table_bbox_max[2] + table_top_margin
    removed = 0
    for obj in _collect_solved_asset_meshes():
        if obj in exclude or obj.name.startswith(_CENTER_TABLE_NAME):
            continue
        o_min, o_max = _world_bbox(obj)
        if not _xy_overlap(
            o_min, o_max, table_bbox_min, table_bbox_max, margin=margin
        ):
            continue
        if o_min[2] > max_keep_z:
            continue
        try:
            butil.delete(obj)
            removed += 1
        except Exception as e:  # pragma: no cover — defensive
            logger.warning(f"Failed to delete {obj.name}: {e}")
    return removed


def _point_camera_at_table(
    table_centre: np.ndarray,
    table_height: float,
    distance: float,
    cam_height: float,
    azimuth_deg: float,
) -> None:
    """Override the primary camera rig so it looks at the centre table."""
    rig = None
    rigs_col = bpy.data.collections.get("camera_rigs")
    if rigs_col is not None and len(rigs_col.objects) > 0:
        rig = rigs_col.objects[0]
    elif bpy.context.scene.camera is not None:
        cam = bpy.context.scene.camera
        rig = cam.parent if cam.parent is not None else cam
    if rig is None:
        logger.warning("No camera rig found — skipping camera override")
        return

    az = np.deg2rad(azimuth_deg)
    offset = np.array(
        [distance * np.cos(az), distance * np.sin(az), cam_height - table_centre[2]]
    )
    cam_world_loc = table_centre + offset
    look_at = np.array([table_centre[0], table_centre[1], table_centre[2] + table_height * 0.6])

    rig.location = Vector(cam_world_loc.tolist())
    direction = Vector((look_at - cam_world_loc).tolist())
    if direction.length == 0:
        return
    quat = direction.to_track_quat("-Z", "Y")
    rig.rotation_mode = "XYZ"
    rig.rotation_euler = quat.to_euler()

    # Any animation data left over from pose_cameras would fight our override.
    if rig.animation_data is not None:
        rig.animation_data_clear()
    for child in rig.children:
        if child.animation_data is not None:
            child.animation_data_clear()

    bpy.context.view_layer.update()
    active_cam = bpy.context.scene.camera
    if active_cam is None and rig.children:
        bpy.context.scene.camera = rig.children[0]


# --- main hook -------------------------------------------------------------


@gin.configurable
def place_center_table(
    scene_seed: int,
    whole_bbox: tuple[np.ndarray, np.ndarray],
    table_location: tuple[float, float, float] | None = None,
    clear_radius_xy: float = 0.2,
    repose_camera: bool = True,
    camera_distance: float = 3.2,
    camera_height: float = 1.55,
    camera_azimuth_deg: float = 35.0,
) -> bpy.types.Object:
    """Place a dining table at the scene centre and clean up collisions.

    Parameters
    ----------
    scene_seed
        Scene seed, also used to seed the table factory for reproducibility.
    whole_bbox
        ``(min, max)`` bounding box of the whole house — used to find the
        xy-centre and the floor height. Supplied by ``compose_indoors``.
    table_location
        Override for the table's world location. Defaults to the xy-centre
        of ``whole_bbox`` at floor height.
    clear_radius_xy
        Extra horizontal margin (metres) around the table's footprint when
        deleting conflicting furniture.
    repose_camera
        If True, move/rotate the primary camera rig to face the table.
    camera_distance, camera_height, camera_azimuth_deg
        Controls for the re-posed camera.
    """
    bb_min = np.asarray(whole_bbox[0], dtype=float)
    bb_max = np.asarray(whole_bbox[1], dtype=float)

    if table_location is None:
        loc = np.array(
            [(bb_min[0] + bb_max[0]) / 2, (bb_min[1] + bb_max[1]) / 2, bb_min[2]]
        )
    else:
        loc = np.asarray(table_location, dtype=float)

    logger.info(f"Placing centre table at {loc.tolist()}")

    factory_seed = int(scene_seed) + 0xC0FFEE
    with FixedSeed(factory_seed):
        fac = TableDiningFactory(factory_seed=factory_seed)
        table = fac.spawn_asset(i=0, loc=tuple(loc.tolist()), rot=(0, 0, 0))
    table.name = _CENTER_TABLE_NAME

    assets_col = bpy.data.collections.get("unique_assets")
    if assets_col is not None:
        for c in list(table.users_collection):
            c.objects.unlink(table)
        assets_col.objects.link(table)

    bpy.context.view_layer.update()

    # Freeze the set of objects that belong to the centre table BEFORE we
    # run the cleanup sweep, otherwise the table (and every mesh it is made
    # of) would match its own footprint and be deleted.
    table_objs = _descendants(table)
    exclude_set = set(table_objs)
    tbl_min, tbl_max = _objs_world_bbox(table_objs)
    table_height = float(tbl_max[2] - tbl_min[2])
    logger.info(
        f"Centre table bbox xy=({tbl_min[0]:.2f}..{tbl_max[0]:.2f},"
        f" {tbl_min[1]:.2f}..{tbl_max[1]:.2f}) h={table_height:.2f}m"
    )

    removed = _clear_conflicting_furniture(
        tbl_min, tbl_max, margin=clear_radius_xy, exclude=exclude_set
    )
    logger.info(f"Centre-table cleanup removed {removed} conflicting object(s)")

    if repose_camera:
        centre = np.array([(tbl_min[0] + tbl_max[0]) / 2,
                           (tbl_min[1] + tbl_max[1]) / 2,
                           float(tbl_min[2])])
        _point_camera_at_table(
            table_centre=centre,
            table_height=table_height,
            distance=camera_distance,
            cam_height=camera_height,
            azimuth_deg=camera_azimuth_deg,
        )

    return table


def compose_indoors_with_center_table(
    output_folder: Path, scene_seed: int, **overrides
):
    info = compose_indoors(output_folder, scene_seed, **overrides)
    whole_bbox = info.get("whole_bbox") if isinstance(info, dict) else None
    if whole_bbox is None:
        logger.warning(
            "compose_indoors did not return a whole_bbox — defaulting to origin"
        )
        whole_bbox = (np.zeros(3), np.zeros(3))
    place_center_table(scene_seed=scene_seed, whole_bbox=whole_bbox)
    return info


_SUPPORTED_ROOM_TYPES = {
    "LivingRoom",
    "DiningRoom",
    "Bedroom",
    "Kitchen",
    "Bathroom",
}


def _build_gin_overrides(args) -> list[str]:
    """Translate ergonomic CLI flags into gin-format overrides."""
    ovr = list(args.overrides)

    def has(prefix: str) -> bool:
        return any(o.startswith(prefix) for o in ovr)

    # Sensible defaults for "one room with furniture" scenes.
    if not has("compose_indoors.terrain_enabled"):
        ovr.append("compose_indoors.terrain_enabled=False")

    if args.room_type is not None:
        if args.room_type not in _SUPPORTED_ROOM_TYPES:
            raise ValueError(
                f"--room_type={args.room_type!r} not in {_SUPPORTED_ROOM_TYPES}"
            )
        ovr.append(f'restrict_solving.restrict_parent_rooms=["{args.room_type}"]')
        ovr.append("restrict_solving.solve_max_rooms=1")
        # restrict_single_supported_roomtype would overwrite the above with a
        # random choice — disable it.
        if not has("compose_indoors.restrict_single_supported_roomtype"):
            ovr.append("compose_indoors.restrict_single_supported_roomtype=False")
    else:
        if not has("compose_indoors.restrict_single_supported_roomtype"):
            ovr.append("compose_indoors.restrict_single_supported_roomtype=True")

    if args.floor_plan is not None and not has("Solver.floor_plan"):
        ovr.append(f'Solver.floor_plan="{args.floor_plan}"')

    return ovr


def main(args):
    scene_seed = init.apply_scene_seed(args.seed)
    init.apply_gin_configs(
        configs=["base_indoors.gin"] + args.configs,
        overrides=_build_gin_overrides(args),
        config_folders=[
            "infinigen_examples/configs_indoor",
            "infinigen_examples/configs_nature",
        ],
    )

    execute_tasks.main(
        compose_scene_func=compose_indoors_with_center_table,
        populate_scene_func=None,
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        task=args.task,
        task_uniqname=args.task_uniqname,
        scene_seed=scene_seed,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", type=Path, required=True)
    parser.add_argument("--input_folder", type=Path, default=None)
    parser.add_argument(
        "-s", "--seed", default=None, help="seed used to generate the scene"
    )
    parser.add_argument(
        "-t",
        "--task",
        nargs="+",
        default=["coarse"],
        choices=[
            "coarse",
            "populate",
            "fine_terrain",
            "ground_truth",
            "render",
            "mesh_save",
            "export",
        ],
    )
    parser.add_argument(
        "-g",
        "--configs",
        nargs="+",
        default=["singleroom.gin", "fast_solve.gin"],
        help="gin configs for the indoor pipeline (without .gin suffix works too)",
    )
    parser.add_argument(
        "-p",
        "--overrides",
        nargs="+",
        default=[],
        help="gin-format overrides, e.g. place_center_table.clear_radius_xy=1.2",
    )
    parser.add_argument(
        "--room_type",
        type=str,
        default=None,
        choices=sorted(_SUPPORTED_ROOM_TYPES),
        help="Fix the single generated room to this type (LivingRoom, etc.). "
        "Picks a random supported type if omitted.",
    )
    parser.add_argument(
        "--floor_plan",
        type=str,
        default=None,
        help="Optional predefined floor plan for the solver. Either a JSON path "
        "or a python dotted path like "
        "'infinigen_examples.configs_indoor.floor_plans.single_room.square_living_room'.",
    )
    parser.add_argument("--task_uniqname", type=str, default=None)
    parser.add_argument("-d", "--debug", type=str, nargs="*", default=None)

    args = init.parse_args_blender(parser)

    logging.getLogger("infinigen").setLevel(logging.INFO)
    logging.getLogger("infinigen.core.nodes.node_wrangler").setLevel(logging.CRITICAL)

    if args.debug is not None:
        for name in logging.root.manager.loggerDict:
            if not name.startswith("infinigen"):
                continue
            if len(args.debug) == 0 or any(name.endswith(x) for x in args.debug):
                logging.getLogger(name).setLevel(logging.DEBUG)

    main(args)
