# utils/speed_bump.py

import os
import random
import numpy as np

from pathlib import Path
from stl import mesh as stl_mesh
from dataclasses import dataclass, asdict

@dataclass
class BumpConfig:
    
    seed: int = 42

    num_bumps: int = 5

    a_range: tuple = (2.5, 3.5)
    b_range: tuple = (0.1, 0.2)
    h_range: tuple = (4.5, 5.0)
    segments: int = 80

    pos_x: float = 0.0
    pos_y: float = 0.0
    pos_z: float = 0.0

    def to_dict(self):
        return asdict(self)

DEFAULT_BUMP_CONFIG = BumpConfig()

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

def tag_mesh(stl_file, scene_xml_path):
    scene_dir = Path(scene_xml_path).parent.resolve()
    rel_path = os.path.relpath(stl_file.resolve(), scene_dir).replace("\\", "/")
    return f'    <mesh name="{stl_file.stem}" file="{rel_path}"/>\n'

def tag_body(stl_file, pos, bump_config):
    rgba = getattr(bump_config, "rgba", "0.9 0.5 0.2 1")
    euler = getattr(bump_config, "euler", "0 0 -90")
    x, y, z = pos
    return (
        f'    <body name="bump" pos="{x} {y} {z}">\n'
        f'        <geom type="mesh" mesh="{stl_file.stem}" rgba="{rgba}" euler="{euler}"/>\n'
        f'    </body>\n'
    )

def create_new_scene(base_path, out_path, bump_config):
    # 1. 랜덤 파라미터 샘플링
    random.seed(bump_config.seed)
    a = random.uniform(*bump_config.a_range)
    b = random.uniform(*bump_config.b_range)
    h = random.uniform(*bump_config.h_range)
    segments = bump_config.segments

    # 2. STL 생성 및 저장
    faces = create_speed_bump(a, b, h, segments)
    bump_dir = "models/speed_bumps/"
    stl_file_name = f"bump_a{a:.2f}_b{b:.2f}_h{h:.2f}.stl"
    Path(bump_dir).mkdir(parents=True, exist_ok=True)
    save_mesh_to_stl(faces, bump_dir, stl_file_name)

    stl_file = Path(bump_dir) / stl_file_name
    pos = (bump_config.pos_x, bump_config.pos_y, bump_config.pos_z)

    # 3. base xml 불러오기
    with open(base_path, "r", encoding="utf-8") as f:
        xml = f.read()

    # 4. mesh 태그 생성 (asset)
    mesh_tag = tag_mesh(stl_file, out_path)
    xml = xml.replace("<asset>", "<asset>\n" + mesh_tag)

    # 5. body 태그 생성 (worldbody)
    body_tag = tag_body(stl_file, pos, bump_config)
    xml = xml.replace("</worldbody>", body_tag + "    </worldbody>")

    # 6. 저장
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(xml)
    print(f"✅ 새로운 scene 저장 완료: {out_path}")
