import overpy
import numpy as np
import pickle
from SE3 import SE3

def sample_edge(a, b, sample_distance):
    ab = b - a
    dist = np.linalg.norm(ab)
    n = int(dist // sample_distance + 1)

    abs = np.linspace(a, b, n, endpoint=True)
    return abs


def sample_way(way, sample_distance):
    sampled_way = np.zeros((0, 3))

    for a, b in zip(way[:-1], way[1:]):
        sampled_edge = sample_edge(a, b, sample_distance)
        sampled_way = np.vstack((sampled_way, sampled_edge))
    return sampled_way

def download_OSM_ways(bbox, category="building"):
    query = f"""
    (
        node
        ({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
        way[{category}]
        ({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
    );
    out;
    """
    api = overpy.Overpass()
    res = api.query(query)
    ways = []
    for way in res.get_ways():
        try:
            nodes = way.get_nodes(resolve_missing=True)
            lats = np.array([node.lat for node in nodes], float)
            lons = np.array([node.lon for node in nodes], float)
            alts = np.zeros(len(lats))
            ways.append(np.column_stack((lats, lons, alts)))
        except overpy.exception.DataIncomplete as e:
            pass

    return ways


from util import llas_to_cart

def create_map(T_init,radius=-1,ul=np.zeros(2),br=np.zeros(2),sample_distance=0.1,out_file=None):
    if radius!=-1:
        ul_lla = (T_init.t - radius)[:2]
        br_lla = (T_init.t + radius)[:2]
    else:
        ul_lla = ul
        br_lla = br

    ways = download_OSM_ways([*ul_lla,*br_lla])
    R_init_inv = T_init.R.inv()

    sampled_ways = []
    for i in range(len(ways)):
        cart_ways = llas_to_cart(ways[i], T_init.t.copy())
        sampled_way = sample_way(cart_ways, sample_distance)
        sampled_way = R_init_inv.apply(sampled_way)
        sampled_way[:,2] = 0
        sampled_ways.append(sampled_way)

    #Returns all building in one list
    sampled_map = np.vstack(sampled_ways)

    #Save to file to path is not None
    if out_file is not None:
        with open(out_file, 'wb') as f:
            pickle.dump(sampled_map,f)

    return sampled_map

def get_bbox_around_poses(poses, margin=0.001):
    pos = np.array([T.t for T in poses])

    return [
        np.min(pos[:, 0]) - margin,
        np.min(pos[:, 1]) - margin,
        np.max(pos[:, 0]) + margin,
        np.max(pos[:, 1]) + margin,
    ]

kitti_odometry_map_creation_data = {
    '00':([0.5064426772205604, -0.8619859170094716, 0.022273157993690715, 48.982545240011, 0.8619081241860264, 0.5053070675316834, -0.04218000669341289, 8.3903743100045, 0.025103787598625422, 0.040559171341034136, 0.9988617289036216, 116.38214111328],[48.980361091781, 8.3878764851142],[48.987607795707994, 8.3970662432033]),
    '01':([-0.035663632424988156, 0.9979790131281588, 0.05259272457290057, 49.006719195871, -0.9993045103122654, -0.035038612314425066, -0.01275897004630585, 8.4893558806503, -0.010890408248474856, -0.05301117809316314, 0.9985345332062162, 122.40083312988],[49.005522947117, 8.4638478161687],[49.018469513433, 8.490355880650299]),
    '02':([0.5886199173874154, -0.8083990786493879, -0.004185987775079567, 48.987607723096, 0.8083433393668688, 0.5884970971924688, 0.01588119319375371, 8.4697469732634, -0.01037490029116594, -0.012731701962377598, 0.9998651235087108, 121.16960906982],[48.984868114784, 8.4687469732634],[48.994646946429, 8.4833559125875]),
    '04':([-0.11548005513143705, -0.993242904817217, 0.011527744666518305, 49.033603440345, 0.9923447444143368, -0.11587153997561633, -0.042728145957003166, 8.3950031909457, 0.043775165334743044, 0.006505248183991029, 0.9990202283467435, 112.39192962646],[49.032603440345, 8.3933636068354],[49.038232180312995, 8.396003190945699]),
    '05':([-0.17139593257698604, -0.9849712069255837, 0.02133438125711487, 49.0495375800, 0.9845840307442081, -0.17201523198000768, -0.03170246631340207, 8.3965961639946, 0.034895855048321836, 0.015571817312787813, 0.9992696321844388, 113.02558135986],[49.047728386793004, 8.391909727267501],[49.053800530615, 8.400198978124399]),
    '06':([-0.9961343621984408, 0.08730224506215492, 0.00972884652017179, 49.05350203001, -0.0876618873201919, -0.9950704036703631, -0.046371168313538276, 8.397219990011, 0.0056325801341474, -0.047044763219809736, 0.9988769014721607, 113.11204528807],[49.052093084493, 8.3921060053258],[49.054616695141995, 8.400365889497]),
    '07':([0.8486612436240476, 0.52881827765415, 0.011194766160258692, 48.98523696217, -0.5285010562056661, 0.8486311087554426, -0.0226246512018021, 8.3936414564418, -0.02146455589990113, 0.013284218885809504, 0.999681350415528, 116.37242889404],[48.984203066691, 8.3917734255884],[48.987737472447996, 8.3968454052086]),
    '08':([0.9924150320601569, -0.12056428477670272, 0.02401369145550308, 48.984262765672, 0.1216673949144067, 0.9912327134551785, -0.051524293212288484, 8.3976660698392, -0.017591166981749073, 0.0540551663818196, 0.9983829875511973, 106.01978302002],[48.979873358406, 8.3961614664544],[48.988951652444996, 8.404633603623399]),
    '09':([0.887059375755127, 0.4606268979634913, -0.03079812912265427, 48.972104544468, -0.46150361684718544, 0.8865026112652454, -0.03357874114477092, 8.4761469953335, 0.011835350518282328, 0.04399978514075507, 0.9989614323814893, 201.86813354492],[48.966486903369, 8.4745695351597],[48.973559865062, 8.4830509934124]),
    '10':([0.9614570037367414, 0.2695055256937767, -0.05447202572044673, 48.97253396005, -0.2711117642262215, 0.9622361205275943, -0.024496115001177692, 8.4785980847297, 0.04581311235567439, 0.03831996832626937, 0.9982147758692816, 220.36932373047],[48.965459913885, 8.4754446849301],[48.97353396005, 8.4796640553283])
}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        prog="create_maps",
        description="Creates map of region specified by latitude/longitude pairs",
        epilog="Text at the bottom of help",
    )

    parser.add_argument("--init_pose",nargs='+',default=[1,0,0,0,0,1,0,0,0,0,1,0], type=float)
    parser.add_argument("--radius",default=-1,type=float)
    parser.add_argument("--ul",default=[0, 0],type=list)
    parser.add_argument("--br", default=[0, 0], type=list)
    parser.add_argument("--output_file", default="", type=str)
    parser.add_argument("--dataloader", default="", type=str)
    parser.add_argument("--seq", default="", type=str)
    parser.add_argument("--show", action="store_true")

    args = parser.parse_args()

    if args.output_file == "":
        print("You have to specify the output file")
        exit()


    init_pose = SE3.from_matrix(np.array(args.init_pose).reshape((-1,4)))
    radius = args.radius
    ul = args.ul
    br = args.br

    if args.dataloader =="kitti":
        init_pose, ul, br = kitti_odometry_map_creation_data[args.seq]
        init_pose = SE3.from_matrix(np.array(init_pose).reshape((-1,4)))
        ul = np.array(ul)
        br = np.array(br)
        radius = -1

    init_pose = init_pose.as_euler()
    init_pose[2:5] = 0
    init_pose = SE3.from_euler(init_pose[:3],init_pose[3:])

    map = create_map(init_pose,radius=radius,ul=ul,br=br,out_file=args.output_file)

    if args.show:
        import matplotlib.pyplot as plt
        print(map.shape)
        plt.scatter(map[:,0],map[:,1])
        plt.show()












