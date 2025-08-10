# utils/speed_bump.py

import os
import random
import numpy as np

from pathlib import Path
from dataclasses import dataclass, asdict

@dataclass
class BumpConfig:
    
    seed: int = 42

    num_bumps: int = 5

    #==========[speed bump 형상]==============
    # 국토교통부 기준 규격
    #   - 과속방지턱은 차량 속도를 30km/h 이하로 제한하기 위해 사용함
    #   - 주행방향 길이는 3m, 높이 10cm를 권장함
    #   - 단 폭이 6m 이하의 소도로에서는 주행방향 길이는 2m, 높이 7.5cm 권장함
    # 출처: https://www.molit.go.kr/USR/I0204/m_45/dtl.jsp?gubun=&search=&search_dept_id=&search_dept_nm=&old_search_dept_nm=&psize=10&search_regdate_s=&search_regdate_e=&srch_usr_nm=N&srch_usr_num=&srch_usr_year=&srch_usr_titl=N&srch_usr_ctnt=N&lcmspage=841&idx=8453

    # 따라서 시뮬레이션에 사용될 bump 형상은 주행방향 길이 3m +-0.2m, 높이 10cm +- 2cm에서 uniform distribution sampling
    a_range: tuple = (2.8, 3.2)  #[m]
    b_range: tuple = (0.08, 0.12)  #[m]
    h_range: tuple = (4.5, 5.5)  #[m]
    segments: int = 80  # backward compat (unused if len_segments/wid_segments 지정)
    len_segments: int = 48  # x(길이) 방향 분할 수
    wid_segments: int = 6   # y(폭) 방향 분할 수

    # 축 정렬 및 재생성 제어
    align_world_axes: bool = True  # 생성 시 OBJ 버텍스를 월드 축(x=길이, y=폭, z=높이)에 정렬
    force_regen: bool = False      # 기존 파일이 있어도 재생성
    add_side_walls: bool = False   # 성능 기본값: 측면 생성 안 함
    add_bottom: bool = False       # 성능 기본값: 바닥 생성 안 함
    double_sided_sides: bool = False  # 측면을 양면으로 복제 여부

    pos_x: float = 0.0
    pos_y: float = 0.0
    pos_z: float = 0.0

    def to_dict(self):
        return asdict(self)

DEFAULT_BUMP_CONFIG = BumpConfig()

def create_speed_bump(
    a: float,
    b: float,
    h: float,
    len_segments: int = 48,
    wid_segments: int = 6,
    align_world_axes: bool = True,
    add_side_walls: bool = False,
    add_bottom: bool = False,
    double_sided_sides: bool = False,
):
    """속도 방지턱(mesh) 생성.

    목표 형상:
    - 월드 축 정렬: x=주행방향 길이 a, y=차폭 h, z=높이 b
    - x 방향 단면은 반타원: z(x) = b * sqrt(1 - (2x/a)^2), |x|<=a/2, 그 외 0
    - y 방향은 -h/2..+h/2로 균일 확장(상부는 평행 이동만)
    - 상면 그리드 + 4면(좌우/전후) 벽 + 바닥면(닫힌 메쉬)
    """
    if not align_world_axes:
        # 구형 방식을 유지해야 한다면 분할수만 반영
        segments = max(16, len_segments)
        theta = np.linspace(0, 2 * np.pi, segments)
        y = a * np.cos(theta)
        z = b * np.sin(theta)
        x_top = np.full_like(y, h / 2)
        x_bottom = np.full_like(y, -h / 2)
        top = np.stack([x_top, y, z], axis=-1)
        bottom = np.stack([x_bottom, y, z], axis=-1)
        center_top = np.array([h / 2, 0.0, 0.0])
        center_bottom = np.array([-h / 2, 0.0, 0.0])
        faces: list[list[np.ndarray]] = []
        for i in range(segments - 1):
            p1, p2 = bottom[i], bottom[i + 1]
            p3, p4 = top[i], top[i + 1]
            faces.extend([[p1, p2, p3], [p2, p4, p3]])
        for i in range(segments - 1):
            faces.append([center_top, top[i], top[i + 1]])
            faces.append([center_bottom, bottom[i + 1], bottom[i]])
        return np.array(faces)

    # 분할수 설정
    n_len = max(16, int(len_segments))           # 길이 방향 분할
    n_wid = max(3, int(wid_segments))            # 폭 방향 분할

    # 격자 생성
    xs = np.linspace(-a / 2.0, a / 2.0, n_len)
    ys = np.linspace(-h / 2.0, h / 2.0, n_wid)

    # 반타원 높이 프로파일 (바닥은 z=0)
    def z_of_x(xv: float) -> float:
        t = 2.0 * xv / a
        val = 1.0 - t * t
        return b * np.sqrt(max(0.0, val))

    zs = np.array([z_of_x(x) for x in xs])  # shape: (n_len,)

    faces: list[list[np.ndarray]] = []

    # 상면(tri mesh) - 위에서 봤을 때 CCW가 되도록 삼각형 정점 순서 설정(법선 +z)
    for i in range(n_len - 1):
        z0, z1 = zs[i], zs[i + 1]
        for j in range(n_wid - 1):
            y0, y1 = ys[j], ys[j + 1]
            v00 = np.array([xs[i],     y0, z0])
            v01 = np.array([xs[i],     y1, z0])
            v10 = np.array([xs[i + 1], y0, z1])
            v11 = np.array([xs[i + 1], y1, z1])
            # 두 삼각형: [v00, v11, v01], [v00, v10, v11] 순서로 상면 법선이 +z가 되도록 구성
            faces.append([v00, v11, v01])
            faces.append([v00, v10, v11])

    if add_side_walls:
        # 좌/우 벽(y = ±h/2, 상면에서 바닥으로)
        yL, yR = ys[0], ys[-1]
        for i in range(n_len - 1):
            x0, x1 = xs[i], xs[i + 1]
            z0, z1 = zs[i], zs[i + 1]
            # 왼쪽(y=yL)
            v_top0 = np.array([x0, yL, z0])
            v_top1 = np.array([x1, yL, z1])
            v_bot0 = np.array([x0, yL, 0.0])
            v_bot1 = np.array([x1, yL, 0.0])
            faces.append([v_bot0, v_top0, v_top1])
            faces.append([v_bot0, v_top1, v_bot1])
            if double_sided_sides:
                faces.append([v_top1, v_top0, v_bot0])
                faces.append([v_bot1, v_top1, v_bot0])
            # 오른쪽(y=yR)
            v_top0 = np.array([x0, yR, z0])
            v_top1 = np.array([x1, yR, z1])
            v_bot0 = np.array([x0, yR, 0.0])
            v_bot1 = np.array([x1, yR, 0.0])
            faces.append([v_bot1, v_top1, v_top0])
            faces.append([v_bot1, v_top0, v_bot0])
            if double_sided_sides:
                faces.append([v_top0, v_top1, v_bot1])
                faces.append([v_bot0, v_top0, v_bot1])

    if add_side_walls:
        # 전/후 벽(x = ±a/2, 상면에서 바닥으로)
        xF, xB = xs[0], xs[-1]
        zF, zB = zs[0], zs[-1]
        for j in range(n_wid - 1):
            y0, y1 = ys[j], ys[j + 1]
            # 전면(x=xF)
            v_top0 = np.array([xF, y0, zF])
            v_top1 = np.array([xF, y1, zF])
            v_bot0 = np.array([xF, y0, 0.0])
            v_bot1 = np.array([xF, y1, 0.0])
            faces.append([v_bot0, v_top0, v_top1])
            faces.append([v_bot0, v_top1, v_bot1])
            if double_sided_sides:
                faces.append([v_top1, v_top0, v_bot0])
                faces.append([v_bot1, v_top1, v_bot0])
            # 후면(x=xB)
            v_top0 = np.array([xB, y0, zB])
            v_top1 = np.array([xB, y1, zB])
            v_bot0 = np.array([xB, y0, 0.0])
            v_bot1 = np.array([xB, y1, 0.0])
            faces.append([v_bot1, v_top1, v_top0])
            faces.append([v_bot1, v_top0, v_bot0])
            if double_sided_sides:
                faces.append([v_top0, v_top1, v_bot1])
                faces.append([v_bot0, v_top0, v_bot1])

    if add_bottom:
        # 바닥면(z=0) 닫기 - 아래에서 봤을 때 CCW가 되도록(법선 -z)
        for i in range(n_len - 1):
            for j in range(n_wid - 1):
                y0, y1 = ys[j], ys[j + 1]
                x0, x1 = xs[i], xs[i + 1]
                v00 = np.array([x0, y0, 0.0])
                v01 = np.array([x0, y1, 0.0])
                v10 = np.array([x1, y0, 0.0])
                v11 = np.array([x1, y1, 0.0])
                faces.append([v00, v11, v01])
                faces.append([v00, v10, v11])

    return np.array(faces)

def save_mesh_to_obj(data: np.ndarray, save_dir: str | Path, file_name: str) -> None:
    """mesh 데이터를 OBJ 파일로 저장

    data shape: (n_faces, 3, 3) with triangles.
    간단화를 위해 각 face의 3개 vertex를 모두 개별로 기록(중복 허용).
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    obj_path = save_dir / file_name
    lines: list[str] = []

    lines.append("o bump")

    # v lines
    n_faces = data.shape[0]
    for i in range(n_faces):
        for j in range(3):
            x, y, z = data[i, j]
            lines.append(f"v {x:.8f} {y:.8f} {z:.8f}")

    # f lines (1-based index)
    # 각 face당 3개의 vertex를 순차적으로 썼으므로, 인덱스는 (3*i+1, 3*i+2, 3*i+3)
    for i in range(n_faces):
        v1 = 3 * i + 1
        v2 = 3 * i + 2
        v3 = 3 * i + 3
        lines.append(f"f {v1} {v2} {v3}")

    obj_path.write_text("\n".join(lines), encoding="utf-8")

def tag_mesh(mesh_file, scene_xml_path):
    scene_dir = Path(scene_xml_path).parent.resolve()
    rel_path = os.path.relpath(Path(mesh_file).resolve(), scene_dir).replace("\\", "/")
    return f'    <mesh name="{Path(mesh_file).stem}" file="{rel_path}"/>\n'

def tag_body(mesh_file, pos, bump_config):
    rgba = getattr(bump_config, "rgba", "0.9 0.5 0.2 1")
    quat = getattr(bump_config, "quat", None)
    x, y, z = pos
    body_name = f"bump_{Path(mesh_file).stem}"
    rot_attr = f' quat="{quat}"' if quat else ""
    return (
        f'    <body name="{body_name}" pos="{x} {y} {z}">\n'
        f'        <geom type="mesh" mesh="{Path(mesh_file).stem}" rgba="{rgba}"{rot_attr}/>\n'
        f'    </body>\n'
    )

def create_new_scene(base_path, out_path, bump_config):
    # 1. 랜덤 파라미터 샘플링
    random.seed(bump_config.seed)
    a = random.uniform(*bump_config.a_range)
    b = random.uniform(*bump_config.b_range)
    h = random.uniform(*bump_config.h_range)
    segments = bump_config.segments

    # 2. OBJ 생성 및 저장
    faces = create_speed_bump(a, b, h, segments, align_world_axes=bump_config.align_world_axes)
    bump_dir = "models/speed_bumps/"
    obj_file_name = f"bump.obj"
    Path(bump_dir).mkdir(parents=True, exist_ok=True)
    save_mesh_to_obj(faces, bump_dir, obj_file_name)

    mesh_file = Path(bump_dir) / obj_file_name
    pos = (bump_config.pos_x, bump_config.pos_y, bump_config.pos_z)

    # 3. base xml 불러오기
    with open(base_path, "r", encoding="utf-8") as f:
        xml = f.read()

    # 4. mesh 태그 생성 (asset)
    mesh_tag = tag_mesh(mesh_file, out_path)
    xml = xml.replace("<asset>", "<asset>\n" + mesh_tag)

    # 5. body 태그 생성 (worldbody)
    body_tag = tag_body(mesh_file, pos, bump_config)
    xml = xml.replace("</worldbody>", body_tag + "    </worldbody>")

    # 6. 저장
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(xml)
    print(f"✅ 새로운 scene 저장 완료: {out_path}")

def create_all_bumps(base_scene_path, output_scene_path, bump_config):
    bump_dir = Path("models/speed_bumps")
    bump_dir.mkdir(parents=True, exist_ok=True)

    # 기존 OBJ 파일들 확인
    existing_names = set(p.name for p in bump_dir.glob("*.obj"))

    mesh_tags = []
    # body_tags는 사용하지 않음 - base_scene.xml에 이미 bump_geom 포함됨

    # 1부터 200까지 체크
    for i in range(1, 201):
        filename = f"{i:03}.obj"
        filepath = bump_dir / filename
        
        # OBJ 파일이 없으면 생성
        if bump_config.force_regen or filename not in existing_names:
            a = np.round(np.random.uniform(*bump_config.a_range), 2)
            b = np.round(np.random.uniform(*bump_config.b_range), 3)
            h = np.round(np.random.uniform(*bump_config.h_range), 2)

            faces = create_speed_bump(a, b, h, bump_config.segments, align_world_axes=bump_config.align_world_axes)
            save_mesh_to_obj(faces, bump_dir, filename)
            print(f"[Generated] {filename}")
        else:
            print(f"[Skipped] {filename} already exists")
        
        # 파일이 있든 없든 mesh 태그는 항상 추가 (1-200번)
        mesh_tags.append(tag_mesh(filepath, Path(output_scene_path)))

    # base XML에 mesh 태그만 삽입 (body 태그는 이미 base_scene에 있음)
    with open(base_scene_path, "r", encoding="utf-8") as f:
        xml = f.read()

    xml = xml.replace("<asset>", "<asset>\n" + "".join(mesh_tags))
    # worldbody는 수정하지 않음 - base_scene.xml에 이미 완성된 구조 포함

    with open(output_scene_path, "w", encoding="utf-8") as f:
        f.write(xml)

    print(f"✅ XML 저장 완료: {output_scene_path}")
    print(f"✅ 1-200번 mesh 태그만 추가됨")
    print(f"✅ worldbody는 base_scene.xml에서 그대로 복사됨")
