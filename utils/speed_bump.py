# utils/speed_bump.py

import os
import random
import numpy as np

from pathlib import Path
from stl import mesh as stl_mesh


def create_speed_bump(a, b, h, segments=80):
    """타원형 방지턱 mesh(삼각형면) 생성"""
    theta = np.linspace(0, 2 * np.pi, segments)
    y = a * np.cos(theta)
    z = b * np.sin(theta)
    x_top = np.full_like(y, h / 2)
    x_bottom = np.full_like(y, -h / 2)
    top = np.stack([x_top, y, z], axis=-1)
    bottom = np.stack([x_bottom, y, z], axis=-1)

    faces = []
    for i in range(segments - 1):
        p1, p2 = bottom[i], bottom[i + 1]
        p3, p4 = top[i], top[i + 1]
        faces.extend([[p1, p2, p3], [p2, p4, p3]])

    center_top = np.array([h / 2, 0, 0])
    center_bottom = np.array([-h / 2, 0, 0])
    for i in range(segments - 1):
        faces.append([center_top, top[i], top[i + 1]])
        faces.append([center_bottom, bottom[i + 1], bottom[i]])
    return np.array(faces)

def save_mesh_to_stl(data, save_dir, file_name):
    """mesh 데이터를 STL 파일로 저장"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    n_faces = data.shape[0]
    stl_data = np.zeros(n_faces, dtype=stl_mesh.Mesh.dtype)
    for i in range(n_faces):
        stl_data['vectors'][i] = data[i]
    mesh_obj = stl_mesh.Mesh(stl_data)
    mesh_obj.save(str(save_dir / file_name))

def generate_random_bumps(config: dict) -> list:
    """config dict 기준으로 bump mesh들과 파라미터 리스트 반환"""
    random.seed(config.get("seed", 42))
    bumps = []
    for _ in range(config.get("num_bumps", 3)):
        a = random.uniform(*config["a_range"])
        b = random.uniform(*config["b_range"])
        h = random.uniform(*config["h_range"])
        faces = create_speed_bump(a, b, h, config.get("segments", 80))
        bumps.append({"faces": faces, "a": a, "b": b, "h": h})
    return bumps

def save_bumps(bumps: list, bump_dir: str):
    """bump mesh dict 리스트를 STL로 저장"""
    for i, bump in enumerate(bumps):
        save_mesh_to_stl(bump["faces"], bump_dir, f"bump_{i+1}_a{bump['a']:.2f}_b{bump['b']:.2f}_h{bump['h']:.2f}.stl")

def get_bump_files(bump_dir: str) -> list:
    """STL bump 파일 리스트(Path) 반환"""
    bump_dir = Path(bump_dir)
    return sorted([f.resolve() for f in bump_dir.iterdir() if f.suffix == ".stl"])

def calc_bumps_pos(config: dict) -> list:
    """bump 위치 (x, y, z) 리스트 계산"""
    n = config.get("num_bumps", 1)
    x = config.get("x_start", 0.0)
    pos_y = config.get("base_pos_y", 0.0)
    pos_z = config.get("base_pos_z", 0.0)
    min_gap = config.get("min_gap", 5.0)
    max_gap = config.get("max_gap", 10.0)
    positions = []
    for i in range(n):
        positions.append((x, pos_y, pos_z))
        if i < n - 1:
            x += random.uniform(min_gap, max_gap)
    return positions

def tag_mesh(stl_files, scene_xml_path):
    scene_dir = Path(scene_xml_path).parent.resolve()
    tags = []
    for f in stl_files:
        rel_path = os.path.relpath(f.resolve(), scene_dir).replace("\\", "/")
        tags.append(f'    <mesh name="{f.stem}" file="{rel_path}"/>\n')
    return ''.join(tags)


def tag_body(stl_files, poses, bump_config):
    """bump body+geom 태그 문자열 생성"""
    rgba = getattr(bump_config, "rgba", "0.9 0.5 0.2 1")
    euler = getattr(bump_config, "euler", "0 0 -90")
    return ''.join(
        f'    <body name="bump{i+1}" pos="{x} {y} {z}">\n'
        f'        <geom type="mesh" mesh="{f.stem}" rgba="{rgba}" euler="{euler}"/>\n'
        f'    </body>\n'
        for i, (f, (x, y, z)) in enumerate(zip(stl_files, poses))
    )

def add_bumps(scene_config, bump_config) -> None:
    """scene xml에 mesh, body 태그 추가 및 저장"""
    stl_files = get_bump_files(scene_config.bump_dir)
    poses = calc_bumps_pos(bump_config.to_dict())
    with open(scene_config.base_xml, "r", encoding="utf-8") as f:
        xml = f.read()
    mesh_tags = tag_mesh(stl_files, scene_config.out_xml)
    body_tags = tag_body(stl_files, poses, bump_config)
    xml = xml.replace("<asset>", "<asset>\n" + mesh_tags)
    xml = xml.replace("</worldbody>", body_tags + "    </worldbody>")
    out_xml = Path(scene_config.out_xml)
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    with open(out_xml, "w", encoding="utf-8") as f:
        f.write(xml)
    print(f"✅ scene 저장 완료: {out_xml}")
