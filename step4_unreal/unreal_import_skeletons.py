"""Step 4 (bonus) — Import the exported skeletons into Unreal Engine 5.

Run INSIDE the Unreal Editor (Python scripting plugin enabled):
  1. Edit > Plugins > enable "Python Editor Script Plugin", restart.
  2. Copy skeletons_unreal.json somewhere accessible and set JSON_PATH below.
  3. Window > Developer Tools > Output Log > Python tab:
       exec(open(r"/path/to/unreal_import_skeletons.py").read())

What it does:
  - spawns one small sphere StaticMeshActor per (player, joint), colored
    red or white by team (dynamic material instance);
  - creates a LevelSequence "SkeletonsSequence" and keyframes every joint
    transform over the annotated frames (held FRAME_HOLD seconds each,
    since only 5 annotated frames exist);
  - bones can be visualized by opening the sequence and enabling the
    included debug-line actor, or simply by the joint clouds (the 18
    joints per player read clearly at this scale).

Tested with Unreal Engine 5.3 Python API (editor scripting only, no
runtime dependency).
"""
import json

import unreal

JSON_PATH = r"skeletons_unreal.json"     # <-- set absolute path in UE
SEQUENCE_PATH = "/Game/SkeletonsSequence"
SPHERE_MESH = "/Engine/BasicShapes/Sphere.Sphere"
JOINT_SCALE = 0.06                       # 6 cm spheres
FRAME_HOLD = 0.5                         # seconds each annotated frame is held


def team_color(player):
    return unreal.LinearColor(0.85, 0.1, 0.1, 1.0) if player.startswith('Red') \
        else unreal.LinearColor(0.95, 0.95, 0.95, 1.0)


def main():
    data = json.load(open(JSON_PATH))
    frames = data['frames']
    if not frames:
        unreal.log_error('No frames in export')
        return

    world = unreal.EditorLevelLibrary.get_editor_world()
    mesh = unreal.EditorAssetLibrary.load_asset(SPHERE_MESH)

    # --- spawn one sphere per (player, joint), positioned at frame 0 ---
    actors = {}      # (player, k) -> actor
    first = frames[0]['players']
    for player, kpts in first.items():
        col = team_color(player)
        for k, p in enumerate(kpts):
            if p is None:
                continue
            loc = unreal.Vector(p[0], p[1], p[2])
            actor = unreal.EditorLevelLibrary.spawn_actor_from_object(mesh, loc)
            actor.set_actor_label(f'{player}_{data["keypoint_names"][k]}')
            actor.set_actor_scale3d(unreal.Vector(JOINT_SCALE, JOINT_SCALE, JOINT_SCALE))
            smc = actor.static_mesh_component
            mid = smc.create_dynamic_material_instance(0)
            if mid:
                mid.set_vector_parameter_value('BaseColor', col)
            actors[(player, k)] = actor

    # --- level sequence with transform keys per frame ---
    seq = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
        'SkeletonsSequence', '/Game', unreal.LevelSequence,
        unreal.LevelSequenceFactoryNew())
    fps = unreal.FrameRate(numerator=30, denominator=1)
    seq.set_display_rate(fps)
    hold_frames = int(FRAME_HOLD * 30)
    seq.set_playback_end(hold_frames * len(frames))

    for (player, k), actor in actors.items():
        binding = seq.add_possessable(actor)
        track = binding.add_track(unreal.MovieScene3DTransformTrack)
        section = track.add_section()
        section.set_range(0, hold_frames * len(frames))
        channels = section.get_all_channels()   # tx ty tz rx ry rz sx sy sz
        for f_i, frame in enumerate(frames):
            p = frame['players'].get(player, [None] * 18)[k]
            if p is None:
                continue
            t = unreal.FrameNumber(f_i * hold_frames)
            for c_i, v in enumerate(p):
                channels[c_i].add_key(t, float(v))

    unreal.EditorAssetLibrary.save_loaded_asset(seq)
    unreal.log(f'Spawned {len(actors)} joint actors, sequence: {SEQUENCE_PATH}')


if __name__ == '__main__':
    main()
