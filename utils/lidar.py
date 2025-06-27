import numpy as np
import mujoco

def get_lidar_scan(model, data, site_name, num_rays=1, max_dist=10.0):
    """
    지정 site에서 z축(-1, 즉 지면 방향)으로만 ray를 쏨.
    여러 개면 중복/샘플링 용이지만 보통 1개면 충분!
    """
    site_id = model.site(site_name).id
    origin = data.site_xpos[site_id].copy()
    scan = np.full(num_rays, max_dist, dtype=np.float32)

    geomgroup = None
    flg_static = 0
    bodyexclude = -1
    geomid = np.array([-1], dtype=np.int32)

    for i in range(num_rays):
        dir_vec = np.array([0.0, 0.0, -1.0])  # z축 아래 방향
        dist = mujoco.mj_ray(
            model, data, origin, dir_vec,
            geomgroup, flg_static, bodyexclude, geomid
        )
        scan[i] = dist if geomid[0] != -1 else max_dist
    return scan

def get_dual_lidar_scan(model, data, site_names=("lidar_left", "lidar_right"), **kwargs):
    scans = []
    for name in site_names:
        scan = get_lidar_scan(model, data, name, **kwargs)
        scans.append(scan)
    return np.stack(scans, axis=0)  # shape (2, num_rays)
