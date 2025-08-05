import numpy as np
import mujoco

def get_lidar_scan(model, data, site_name, num_rays=30, max_dist=5.0):
    site_id = model.site(site_name).id
    origin = data.site_xpos[site_id].copy()
    rot_mat = data.site_xmat[site_id].reshape(3, 3)  # Local → World 회전행렬

    # 차량 기준 벡터
    forward = np.array([1, 0, 0])  # theta=0일 때는 차량 정면방향
    up = np.array([0, 0, -1])  # theta=90일 때 차량 아래방향
    # 측정 각도: 
    angles = np.radians(np.linspace(30, 150, num_rays))

    scan = np.full(num_rays, max_dist, dtype=np.float32)  # initialization
    geomgroup = None  # 모든 geom을 대상으로 함
    flg_static = 1
    bodyexclude = model.body("chassis").id
    geomid = np.array([-1], dtype=np.int32)

    for i, theta in enumerate(angles):
        # local 방향 (X–Z 평면 기준)
        dir_local = np.cos(theta) * forward + np.sin(theta) * up
        dir_world = rot_mat @ dir_local  # Local → World

        dist = mujoco.mj_ray(
            model, data, origin, dir_world,
            geomgroup, flg_static, bodyexclude, geomid
        )
        scan[i] = dist if geomid[0] != -1 else max_dist
    #print(scan)
    return scan


def get_dual_lidar_scan(model, data, site_names=("lidar_left", "lidar_right"), **kwargs):
    scans = []
    for name in site_names:
        scan = get_lidar_scan(model, data, name, **kwargs)
        scans.append(scan)
    return np.stack(scans, axis=0)  # shape (2, num_rays)