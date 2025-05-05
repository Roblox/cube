"""
Blender script to render images of 3D models.

This script is adopted from the Trellis rendering script:
https://github.com/microsoft/TRELLIS/blob/main/dataset_toolkits/render.py

"""

import argparse
import math
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Literal, Optional, Tuple

import bpy
import numpy as np
from mathutils import Vector

pathdir = Path(__file__).parent
sys.path.append(pathdir.as_posix())

print(dir(bpy), bpy.__path__)

IMPORT_FUNCTIONS: Dict[str, Callable] = {
    ".obj": bpy.ops.wm.obj_import,
    ".glb": bpy.ops.import_scene.gltf,
    ".gltf": bpy.ops.import_scene.gltf,
}


def center_and_scale_mesh(scale_value: float = 1.0) -> None:
    """Centers and scales the scene to fit in a unit cube.
    For example,
        scale_value = 1.0 ==> [-0.5, 0.5]
        scale_value = 2.0 ==> [-1.0, 1.0]
    """
    # Get all mesh objects
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not mesh_objects:
        return

    # Calculate bounds
    min_coords = Vector((float("inf"),) * 3)
    max_coords = Vector((float("-inf"),) * 3)

    for obj in mesh_objects:
        # Get all vertices in world space
        for vertex in obj.data.vertices:
            world_coord = obj.matrix_world @ vertex.co
            min_coords.x = min(min_coords.x, world_coord.x)
            min_coords.y = min(min_coords.y, world_coord.y)
            min_coords.z = min(min_coords.z, world_coord.z)
            max_coords.x = max(max_coords.x, world_coord.x)
            max_coords.y = max(max_coords.y, world_coord.y)
            max_coords.z = max(max_coords.z, world_coord.z)

    # Calculate center and dimensions
    center = (min_coords + max_coords) / 2
    dimensions = max_coords - min_coords
    scale = scale_value / max(
        dimensions.x, dimensions.y, dimensions.z
    )  # Scale to fit in [-scale_value/2, scale_value/2] cube

    # Create an empty to serve as the parent
    empty = bpy.data.objects.new("Parent_Empty", None)
    bpy.context.scene.collection.objects.link(empty)

    # Parent all mesh objects to the empty
    for obj in mesh_objects:
        obj.parent = empty

    # Move empty to center everything
    empty.location = -center

    # Apply scale to empty
    empty.scale = (scale, scale, scale)

    bpy.context.view_layer.update()
    bpy.ops.object.select_all(action="DESELECT")
    empty.select_set(True)
    bpy.context.view_layer.objects.active = empty
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    print(f"Empty location: {empty.location}")
    print(f"Empty scale: {empty.scale}")

    return scale


def normalize_scene() -> None:
    """Normalizes the scene by scaling and translating it to fit in a unit cube centered
    at the origin.

    Mostly taken from the Point-E / Shap-E rendering script
    (https://github.com/openai/point-e/blob/main/point_e/evals/scripts/blender_script.py#L97-L112),
    but fix for multiple root objects: (see bug report here:
    https://github.com/openai/shap-e/pull/60).

    Returns:
        The new parent object that all objects descend from.
    """
    if len(list(get_scene_root_objects())) > 1:
        # create an empty object to be used as a parent for all root objects
        parent_empty = bpy.data.objects.new("ParentEmpty", None)
        bpy.context.scene.collection.objects.link(parent_empty)

        # parent all root objects to the empty object
        for obj in get_scene_root_objects():
            if obj != parent_empty:
                obj.parent = parent_empty

    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    for obj in get_scene_root_objects():
        obj.scale = obj.scale * scale

    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in get_scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")
    bbox_min, bbox_max = scene_bbox()
    print(f"After normalize_scene: bbox_min: {bbox_min}, bbox_max: {bbox_max}")

    # unparent the camera
    bpy.data.objects["Camera"].parent = None

    return parent_empty


def reset_cameras() -> None:
    """Resets the cameras in the scene to a single default camera."""
    # Delete all existing cameras
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="CAMERA")
    bpy.ops.object.delete()

    # Create a new camera with default properties
    bpy.ops.object.camera_add()

    # Rename the new camera to 'NewDefaultCamera'
    new_camera = bpy.context.active_object
    new_camera.name = "Camera"

    # Set the new camera as the active camera for the scene
    scene.camera = new_camera


def get_camera_with_position(x, y, z, fov_degrees=40):
    camera = bpy.data.objects["Camera"]
    camera.data.angle = math.radians(fov_degrees)
    camera.location = np.array([x, y, z])
    direction = -camera.location
    rot_quat = direction.to_track_quat("-Z", "Y")
    camera.rotation_euler = rot_quat.to_euler()
    return camera


def reset_scene() -> None:
    """Resets the scene to a clean state.

    Returns:
        None
    """
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)

    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)

    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)

    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


def load_object(object_path: str) -> None:
    """Loads a model with a supported file extension into the scene.

    Args:
        object_path (str): Path to the model file.

    Raises:
        ValueError: If the file extension is not supported.

    Returns:
        None
    """
    file_extension = Path(object_path).suffix
    if file_extension is None or file_extension == "":
        raise ValueError(f"Unsupported file type: {object_path}")

    # load from existing import functions
    import_function = IMPORT_FUNCTIONS[file_extension]

    if file_extension in {".glb", ".gltf"}:
        import_function(filepath=object_path, merge_vertices=True)
    else:
        import_function(filepath=object_path)


def clear_lights():
    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, bpy.types.Light):
            obj.select_set(True)
    bpy.ops.object.delete()


def create_light(
    location,
    energy=1.0,
    angle=0.5 * math.pi / 180,
    light_type: Literal["POINT", "SUN", "SPOT", "AREA"] = "SUN",
):
    # https://blender.stackexchange.com/questions/215624/how-to-create-a-light-with-the-python-api-in-blender-2-92
    light_data = bpy.data.lights.new(name="Light", type=light_type)
    light_data.energy = energy
    if light_type != "AREA" and light_type != "POINT":
        light_data.angle = angle
    light_object = bpy.data.objects.new(name="Light", object_data=light_data)

    direction = -location
    rot_quat = direction.to_track_quat("-Z", "Y")
    light_object.rotation_euler = rot_quat.to_euler()
    bpy.context.view_layer.update()

    bpy.context.collection.objects.link(light_object)
    light_object.location = location


def create_uniform_lights(
    distance=2.0,
    energy=3.0,
    light_type: Literal["POINT", "SUN", "SPOT", "AREA"] = "SUN",
):
    clear_lights()
    create_light(Vector([1, 0, 0]) * distance, energy=energy, light_type=light_type)
    create_light(-Vector([1, 0, 0]) * distance, energy=energy, light_type=light_type)
    create_light(Vector([0, 1, 0]) * distance, energy=energy, light_type=light_type)
    create_light(-Vector([0, 1, 0]) * distance, energy=energy, light_type=light_type)
    create_light(Vector([0, 0, 1]) * distance, energy=energy, light_type=light_type)
    create_light(-Vector([0, 0, 1]) * distance, energy=energy, light_type=light_type)


def create_light_at_camera_position(
    camera_position: Vector,
    energy=1.5,
    use_shadow=False,
    light_type: Literal["POINT", "SUN", "SPOT", "AREA"] = "SUN",
):
    clear_lights()
    create_light(camera_position, energy=energy, light_type=light_type)
    # disable shadows
    if not use_shadow:
        for light in bpy.data.lights:
            light.use_shadow = False


def set_world_background_color(
    color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
) -> None:
    bpy.context.scene.world.use_nodes = True
    bpy.context.scene.world.node_tree.nodes["Background"].inputs[
        0
    ].default_value = color
    bpy.context.scene.view_settings.view_transform = "Standard"


def scene_bbox(
    single_obj: Optional[bpy.types.Object] = None, ignore_matrix: bool = False
) -> Tuple[Vector, Vector]:
    """Returns the bounding box of the scene.

    Taken from Shap-E rendering script
    (https://github.com/openai/shap-e/blob/main/shap_e/rendering/blender/blender_script.py#L68-L82)

    Args:
        single_obj (Optional[bpy.types.Object], optional): If not None, only computes
            the bounding box for the given object. Defaults to None.
        ignore_matrix (bool, optional): Whether to ignore the object's matrix. Defaults
            to False.

    Raises:
        RuntimeError: If there are no objects in the scene.

    Returns:
        Tuple[Vector, Vector]: The minimum and maximum coordinates of the bounding box.
    """
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in get_scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))

    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")

    return Vector(bbox_min), Vector(bbox_max)


def get_scene_root_objects() -> Generator[bpy.types.Object, None, None]:
    """Returns all root objects in the scene.

    Yields:
        Generator[bpy.types.Object, None, None]: Generator of all root objects in the
            scene.
    """
    for obj in bpy.context.scene.objects.values():
        if not obj.parent and not isinstance(obj.data, bpy.types.Light):
            yield obj


def get_scene_meshes() -> Generator[bpy.types.Object, None, None]:
    """Returns all meshes in the scene.

    Yields:
        Generator[bpy.types.Object, None, None]: Generator of all meshes in the scene.
    """
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


def delete_missing_textures() -> Dict[str, Any]:
    """Deletes all missing textures in the scene.

    Returns:
        Dict[str, Any]: Dictionary with keys "count", "files", and "file_path_to_color".
            "count" is the number of missing textures, "files" is a list of the missing
            texture file paths, and "file_path_to_color" is a dictionary mapping the
            missing texture file paths to a random color.
    """
    missing_file_count = 0
    out_files = []
    file_path_to_color = {}

    # Check all materials in the scene
    for material in bpy.data.materials:
        if material.use_nodes:
            for node in material.node_tree.nodes:
                if node.type == "TEX_IMAGE":
                    image = node.image
                    if image is not None:
                        file_path = bpy.path.abspath(image.filepath)
                        if file_path == "":
                            # means it's embedded
                            continue

                        if not os.path.exists(file_path):
                            # Find the connected Principled BSDF node
                            connected_node = node.outputs[0].links[0].to_node

                            if connected_node.type == "BSDF_PRINCIPLED":
                                if file_path not in file_path_to_color:
                                    # Set a random color for the unique missing file path
                                    random_color = [random.random() for _ in range(3)]
                                    file_path_to_color[file_path] = random_color + [1]

                                connected_node.inputs[
                                    "Base Color"
                                ].default_value = file_path_to_color[file_path]

                            # Delete the TEX_IMAGE node
                            material.node_tree.nodes.remove(node)
                            missing_file_count += 1
                            out_files.append(image.filepath)
    return {
        "count": missing_file_count,
        "files": out_files,
        "file_path_to_color": file_path_to_color,
    }


def setup_environment_lighting(envmap_path):
    world = bpy.context.scene.world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links

    # Clear existing nodes
    for node in nodes:
        nodes.remove(node)

    # Create Background node
    bg_node = nodes.new(type="ShaderNodeBackground")
    bg_node.location = (0, 0)

    # Create Environment Texture node
    env_tex_node = nodes.new(type="ShaderNodeTexEnvironment")
    env_tex_node.location = (-300, 0)

    # Set the environment texture path (replace this with your file path)
    env_tex_node.image = bpy.data.images.load(envmap_path)

    # Create World Output node
    world_output_node = nodes.new(type="ShaderNodeOutputWorld")
    world_output_node.location = (300, 0)

    # Link nodes
    links.new(env_tex_node.outputs["Color"], bg_node.inputs["Color"])
    links.new(bg_node.outputs["Background"], world_output_node.inputs["Surface"])


def create_solid_color_material(name, color):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    node_tree = mat.node_tree
    color_node = node_tree.nodes.new("ShaderNodeBsdfDiffuse")
    color_node.inputs["Color"].default_value = color
    mat_output = node_tree.nodes["Material Output"]
    node_tree.links.new(color_node.outputs["BSDF"], mat_output.inputs["Surface"])
    return mat


def create_phong_material(name, color):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    node_tree = mat.node_tree
    spec_node = node_tree.nodes.new("ShaderNodeBsdfPrincipled")
    print(spec_node.inputs.keys())
    spec_node.inputs["Base Color"].default_value = color
    spec_node.inputs["Roughness"].default_value = 0.5
    spec_node.inputs["Metallic"].default_value = 1.0
    mat_output = node_tree.nodes["Material Output"]
    node_tree.links.new(spec_node.outputs["BSDF"], mat_output.inputs["Surface"])
    return mat


def render_object(
    object_file: str,
    num_renders: int,
    output_dir: str,
    transparent_background: bool = False,
    environment_map: str = None,
) -> None:
    """Saves rendered images for given asset to specified output directory.

    Args:
        object_file (str): Path to the object file.
        num_renders (int): Number of renders to save of the object.
        output_dir (str): Path to the directory where the rendered images and metadata
            will be saved. The rendered images will be saved in the subdirectory
            `output_dir/stemname`.
        transparent_background (bool): Whether to use transparent background,
            otherwise the background is white.
    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)

    # load the object
    reset_scene()
    load_object(object_file)

    if transparent_background:
        scene.render.film_transparent = True
    else:
        scene.render.film_transparent = False

    set_world_background_color([0.2, 0.2, 0.2, 1.0])

    # normalize the scene
    _ = normalize_scene()

    # Set up cameras
    cam = scene.objects["Camera"]
    fov_degrees = 40.0
    cam.data.angle = np.radians(fov_degrees)

    # Set up camera constraints
    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"
    empty = bpy.data.objects.new("Empty", None)
    empty.location = (0, 0, 0)
    scene.collection.objects.link(empty)
    cam_constraint.target = empty
    cam.parent = empty

    # delete all objects that are not meshes
    delete_missing_textures()

    if environment_map:
        setup_environment_lighting(environment_map)
    else:
        create_uniform_lights(energy=1.0, light_type="SUN")

    camera_position = [0, -2, 0]

    # determine how much to orbit camera by.
    stepsize = 360.0 / num_renders

    def render_views(name):
        for i in range(num_renders):
            # set camera
            _ = get_camera_with_position(
                camera_position[0],
                camera_position[1],
                camera_position[2],
                fov_degrees=fov_degrees,
            )

            # Set output paths with absolute paths
            render_path = os.path.abspath(
                os.path.join(output_dir, f"{i:03d}_{name}.png")
            )

            # Set file output paths
            scene.render.filepath = render_path

            # Make sure the output directory exists
            os.makedirs(output_dir, exist_ok=True)

            # Render
            bpy.ops.render.render(write_still=True)

            context.view_layer.objects.active = empty
            empty.rotation_euler[2] += math.radians(stepsize)

    # ensure that all objects have materials, if not then add a default
    # one.
    textured_mat = create_solid_color_material("default texture", [0.6, 0.6, 0.6, 1])

    for obj in get_scene_meshes():
        if obj.active_material is None:
            obj.active_material = textured_mat

    render_views("textured")


def enable_gpus(device_type, use_cpus=False):
    preferences = bpy.context.preferences
    cycles_preferences = preferences.addons["cycles"].preferences
    cycles_preferences.refresh_devices()
    try:
        devices = cycles_preferences.devices
    except:
        print("No devices detected")
        if device_type == "CPU":
            return []
        else:
            raise RuntimeError("No devices detected, set use_cpus to True")

    assert device_type in [
        "CUDA",
        "METAL",
        "OPENCL",
        "CPU",
        "NONE",
    ], f"Unsupported device type: {device_type}"

    try:
        # print(devices)
        iter(devices)
    except TypeError:
        # print("Single GPU Detected")
        devices = [devices]

    activated_gpus = []
    for device in devices:
        if device.type == "CPU":
            device.use = use_cpus
        else:
            device.use = True
            activated_gpus.append(device.name)

    if device_type == "CUDA":
        cycles_preferences.compute_device_type = "CUDA"
        bpy.context.scene.cycles.device = "GPU"
    elif device_type == "METAL":
        cycles_preferences.compute_device_type = "METAL"
        bpy.context.scene.cycles.device = "GPU"
    elif device_type == "OPENCL":
        cycles_preferences.compute_device_type = "OPENCL"
        bpy.context.scene.cycles.device = "GPU"
    else:
        raise RuntimeError(f"Unsupported device type: {device_type}")

    return activated_gpus


def set_render_settings(engine, resolution):
    # Set render settings
    render.engine = engine  #
    render.image_settings.file_format = "PNG"
    render.image_settings.color_mode = "RGBA"
    render.resolution_x = resolution
    render.resolution_y = resolution
    render.resolution_percentage = 100

    # Set cycles settings
    scene.cycles.device = "GPU"
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.1
    scene.cycles.samples = 64
    scene.cycles.adaptive_min_samples = 1
    scene.cycles.filter_width = 2
    scene.cycles.use_fast_gi = True
    scene.cycles.fast_gi_method = "REPLACE"
    world.light_settings.ao_factor = 1.0
    world.light_settings.distance = 10
    scene.cycles.use_denoising = True  # ML denoising
    scene.cycles.denoising_use_gpu = True

    # bake existing frames for faster future renders
    scene.render.use_persistent_data = True

    # Set eevee settings
    scene.eevee.use_shadows = True
    scene.eevee.use_raytracing = True
    scene.eevee.ray_tracing_options.use_denoise = True
    scene.eevee.use_fast_gi = True
    scene.eevee.fast_gi_method = "GLOBAL_ILLUMINATION"
    scene.eevee.ray_tracing_options.trace_max_roughness = 0.5
    scene.eevee.fast_gi_resolution = "2"
    scene.eevee.fast_gi_ray_count = 2
    scene.eevee.fast_gi_step_count = 8


def print_devices():
    print("Devices:")
    preferences = bpy.context.preferences
    cycles_preferences = preferences.addons["cycles"].preferences
    cycles_preferences.refresh_devices()

    devices = cycles_preferences.devices
    for device in devices:
        print(f'   [{device.id}]<{device.type}> "{device.name}" Using: {device.use}')

    print(f"Compute device type: {cycles_preferences.compute_device_type}")
    print(f"Cycles device: {bpy.context.scene.cycles.device}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--object_path",
        type=str,
        required=False,
        help="Path to the object file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the directory where the rendered images and metadata will be saved.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="BLENDER_EEVEE_NEXT",  # BLENDER_BLENDER_EEVEE_NEXT rasterization, better than nvdifrast, CYCLES
        choices=["CYCLES", "BLENDER_EEVEE_NEXT"],
    )
    parser.add_argument(
        "--num_renders",
        type=int,
        default=12,
        help="Number of renders to save of the object.",
    )
    parser.add_argument(
        "--render_resolution",
        type=int,
        default=512,
        help="Resolution of the rendered images.",
    )
    parser.add_argument(
        "--transparent_background",
        action="store_true",
        help="Whether to use transparent background",
    )
    parser.add_argument(
        "--environment_map",
        default=None,
        type=str,
        help="Use the given environment map for lighting",
    )

    argv = sys.argv[sys.argv.index("--") + 1 :]
    args = parser.parse_args(argv)

    context = bpy.context
    scene = context.scene
    render = scene.render
    world = bpy.data.worlds["World"]

    set_render_settings(args.engine, args.render_resolution)

    # detect platform and activate GPUs
    platform = platform.system()
    if platform == "Darwin":
        activated_gpus = enable_gpus("METAL", use_cpus=True)
    elif platform == "Linux":
        activated_gpus = enable_gpus("CUDA", use_cpus=False)
    else:
        raise RuntimeError("Unsupported platform")
    print(f"Activated GPUs: {activated_gpus}")

    print_devices()

    render_object(
        object_file=args.object_path,
        num_renders=args.num_renders,
        output_dir=args.output_dir,
        transparent_background=args.transparent_background,
        environment_map=args.environment_map,
    )
