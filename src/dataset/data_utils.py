import json
import numpy as np
from scipy.spatial.transform import Rotation as R
import math
import os
import trimesh as tm
import csv
from plyfile import PlyData, PlyElement
from tqdm import tqdm
# import pyviz3d.visualizer as viz
# import open3d as o3d
import numpy as np
from collections import defaultdict


OBJ_CLASS_IDS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]) # exclude wall (1), floor (2), ceiling (22)
EXCLUDE_LABELS = ["object", "wall", "floor", "ceiling"]


def represents_int(s):
    ''' if string s represents an int. '''
    try: 
        int(s)
        return True
    except ValueError:
        return False

def read_label_mapping(filename, label_from='raw_category', label_to='nyu40id'):
    assert os.path.isfile(filename)
    mapping = dict()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            mapping[row[label_from]] = int(row[label_to])
    if represents_int(list(mapping.keys())[0]):
        mapping = {int(k):v for k,v in mapping.items()}
    return mapping

def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt( arr[:,0]**2 + arr[:,1]**2 + arr[:,2]**2 )
    arr[:,0] /= (lens + 1e-8)
    arr[:,1] /= (lens + 1e-8)
    arr[:,2] /= (lens + 1e-8)                
    return arr

def compute_normal(vertices, faces):
    #Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    normals = np.zeros( vertices.shape, dtype=vertices.dtype )
    #Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces]
    #Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle             
    n = np.cross( tris[::,1 ] - tris[::,0]  , tris[::,2 ] - tris[::,0] )
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices, 
    # we need to normalize these, so that our next step weights each normal equally.
    normalize_v3(n)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle, 
    # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    normals[ faces[:,0] ] += n
    normals[ faces[:,1] ] += n
    normals[ faces[:,2] ] += n
    normalize_v3(normals)
    
    return normals

def read_mesh_vertices_rgb_normal(filename):
    """ read XYZ RGB normals point cloud from filename PLY file """
    assert(os.path.isfile(filename))
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 9], dtype=np.float32)
        vertices[:,0] = plydata['vertex'].data['x']
        vertices[:,1] = plydata['vertex'].data['y']
        vertices[:,2] = plydata['vertex'].data['z']
        vertices[:,3] = plydata['vertex'].data['red']
        vertices[:,4] = plydata['vertex'].data['green']
        vertices[:,5] = plydata['vertex'].data['blue']

        # compute normals
        xyz = np.array([[x, y, z] for x, y, z, _, _, _, _ in plydata["vertex"].data])
        face = np.array([f[0] for f in plydata["face"].data])
        nxnynz = compute_normal(xyz, face)
        vertices[:,6:] = nxnynz
    return vertices

def read_mesh_vertices_rgb(filename):
    """ read XYZ RGB normals point cloud from filename PLY file """
    # assert(os.path.isfile(filename))
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
        vertices[:,0] = plydata['vertex'].data['x']
        vertices[:,1] = plydata['vertex'].data['y']
        vertices[:,2] = plydata['vertex'].data['z']
        vertices[:,3] = plydata['vertex'].data['red']
        vertices[:,4] = plydata['vertex'].data['green']
        vertices[:,5] = plydata['vertex'].data['blue']
    return vertices

def read_aggregation(filename):
    object_id_to_segs = {}
    label_to_segs = {}
    with open(filename) as f:
        data = json.load(f)
        num_objects = len(data['segGroups'])
        object_id_to_label = []
        for i in range(num_objects):
            object_id = data['segGroups'][i]['objectId'] + 1 # instance ids should be 1-indexed
            label = data['segGroups'][i]['label']
            segs = data['segGroups'][i]['segments']
            object_id_to_segs[object_id] = segs
            object_id_to_label.append(label)    # 0-indexed, same as *.aggregation.json
            if label in label_to_segs:
                label_to_segs[label].extend(segs)
            else:
                label_to_segs[label] = segs
    return object_id_to_segs, label_to_segs, object_id_to_label

def read_segmentation(filename):
    seg_to_verts = {}
    with open(filename) as f:
        data = json.load(f)
        num_verts = len(data['segIndices'])
        for i in range(num_verts):
            seg_id = data['segIndices'][i]
            if seg_id in seg_to_verts:
                seg_to_verts[seg_id].append(i)
            else:
                seg_to_verts[seg_id] = [i]
    return seg_to_verts, num_verts

def get_rotation_matrix(oz):
    """     z                                                      z
            |   y (12 o'clock)                                     |   y (12 o'clock, line of sight after rotation, front)
            |  /                                                   |  /
            | /                                                    | /
    _ _ _ _ |/_ _ _ _ x (3 o'clock, right hand)     ->      _ _ _ _|/_ _ _ _ x (3 o'clock, right hand)
        oz <\
             \
              \
               \
        the original line of sight
    Rotate counterclockwise around the z-axis by 3/2*pi-oz radians, 
    and the sight line is parallel to the Oxy plane, pointing towards the positive y-axis.
    arguments:
        oz: The angle (in radians) between (the line connecting the agent to the origin) and (the negative x-axis direction)
    returns:
        rotation_matrix: The rotation matrix for rotating clockwise around the z-axis (viewed from the top of the Oxy plane) by ( pi/2 + (pi - oz) ) radians
    """
    oz = np.pi + np.pi/2 - oz
    rotation_matrix = np.array([[np.cos(oz), -np.sin(oz), 0, 0],
                                [np.sin(oz), np.cos(oz), 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])
    return rotation_matrix

def get_center_transformation_matrix(point_cloud):
    """
    arguments:
        point_cloud: Nx3, [[x,y,z]]
    """
    bs_center = (np.max(point_cloud[:, 0 : 3], axis=0) + np.min(point_cloud[:, 0 : 3], axis=0)) / 2
    scene_transformation = np.array([[1, 0, 0, -bs_center[0]],
                                     [0, 1, 0, -bs_center[1]],
                                     [0, 0, 1, -bs_center[2]],
                                     [0, 0, 0, 1]])
    return scene_transformation.transpose()

def get_agent_view_transformation_matrix(position):
    agent_quat = np.array(position[3:])  # qw, qx, qy, qz
    agent_pos = np.array(position[:3])  # x, y, z

    transformation_matrix = np.array([[1, 0, 0, -agent_pos[0]],
                                      [0, 1, 0, -agent_pos[1]],
                                      [0, 0, 1, 0],
                                      [0, 0, 0, 1]])
    r = R.from_quat(agent_quat)
    angles = r.as_euler('xyz', degrees=False)
    
    rotation_matrix = get_rotation_matrix(angles[-1])
    return np.dot(transformation_matrix.transpose(), rotation_matrix.transpose())


def export(mesh_file, agg_file, seg_file, label_map_file, position):
    """ points are XYZ RGB (RGB in 0-255),
    arguments:
        mesh_file: *_vh_clean_2.ply
        agg_file: *.aggregation.json
        seg_file: *_vh_clean_2.0.010000.segs.json
        label_map_file: scannetv2-labels.combined.tsv, mapping from raw_category to nyu40id
        position: [x,y,z,qw,qx,qy,qz]
    returns:
        selected_object_pcd: nxNx6, [[[x,y,z,r,g,b], ...], ...]]
        selected_object_label: ["table", ...], only include objects whose nyu40id is in OBJ_CLASS_IDS
        object_ids: [0, 2, 3, ...], object_id of selected_object_label in raw *.aggregation.json(0-indexed)
        instance_bboxes: [[x,y,z,l,w,h], ...]
        instance_rgb: [[r,g,b], ...]
    """
    label_map = read_label_mapping(label_map_file, label_from='raw_category', label_to='nyu40id')    
    mesh_vertices = read_mesh_vertices_rgb_normal(mesh_file)    # [[x,y,z,r,g,b,nx,ny,nz]]

    pts = np.ones((mesh_vertices.shape[0], 4))
    pts[:,0:3] = mesh_vertices[:,0:3]
    center_transformation_matrix = get_center_transformation_matrix(mesh_vertices)
    agent_view_transformation_matrix = get_agent_view_transformation_matrix(position)
    transformation_matrix = np.dot(center_transformation_matrix, agent_view_transformation_matrix)
    pts = np.dot(pts, transformation_matrix)

    aligned_vertices = np.copy(mesh_vertices)
    aligned_vertices[:,0:3] = pts[:,0:3]

    # Load semantic and instance labels
    if os.path.isfile(agg_file):
        object_id_to_segs, label_to_segs, object_id_to_label = read_aggregation(agg_file)
        seg_to_verts, num_verts = read_segmentation(seg_file)

        label_ids = np.zeros(shape=(num_verts), dtype=np.uint32) # 0: unannotated
        object_id_to_label_id = {}
        for label, segs in label_to_segs.items():
            label_id = label_map[label]
            for seg in segs:
                verts = seg_to_verts[seg]
                label_ids[verts] = label_id
        instance_ids = np.zeros(shape=(num_verts), dtype=np.uint32) # 0: unannotated
        num_instances = len(np.unique(list(object_id_to_segs.keys())))
        for object_id, segs in object_id_to_segs.items():
            for seg in segs:
                verts = seg_to_verts[seg]
                instance_ids[verts] = object_id
                if object_id not in object_id_to_label_id:
                    object_id_to_label_id[object_id] = label_ids[verts][0]
        
        instance_bboxes = np.zeros((num_instances,8)) # also include object id
        aligned_instance_bboxes = np.zeros((num_instances,8)) # also include object id
        aligned_instance_rgb = np.zeros((num_instances,3)) # also include object id
        object_pcd = []

        for obj_id in object_id_to_segs:
            label_id = object_id_to_label_id[obj_id]    # nyu40id
            # bboxes in the aligned meshes
            obj_pc = aligned_vertices[instance_ids==obj_id, 0:3]
            obj_rgb = aligned_vertices[instance_ids==obj_id, 3:6]
            
            if len(obj_pc) == 0: continue
            # Compute axis aligned box
            # An axis aligned bounding box is parameterized by
            # (cx,cy,cz) and (dx,dy,dz) and label id
            # where (cx,cy,cz) is the center point of the box,
            # dx is the x-axis length of the box.
            xmin = np.min(obj_pc[:,0])
            ymin = np.min(obj_pc[:,1])
            zmin = np.min(obj_pc[:,2])
            xmax = np.max(obj_pc[:,0])
            ymax = np.max(obj_pc[:,1])
            zmax = np.max(obj_pc[:,2])
            bbox = np.array([(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2, xmax-xmin, ymax-ymin, zmax-zmin, label_id, obj_id-1]) # also include object id
            # NOTE: this assumes obj_id is in 1,2,3,.,,,.NUM_INSTANCES
            aligned_instance_bboxes[obj_id-1,:] = bbox 
            rgb = np.array([np.mean(obj_rgb[:,0]), np.mean(obj_rgb[:,1]), np.mean(obj_rgb[:,2])])
            aligned_instance_rgb[obj_id-1,:] = rgb
            object_pcd.append(np.concatenate((obj_pc, obj_rgb), axis=1))
    else:
        # use zero as placeholders for the test scene
        print("use placeholders")
        num_verts = mesh_vertices.shape[0]
        label_ids = np.zeros(shape=(num_verts), dtype=np.uint32) # 0: unannotated
        instance_ids = np.zeros(shape=(num_verts), dtype=np.uint32) # 0: unannotated
        instance_bboxes = np.zeros((1, 8)) # also include object id
        aligned_instance_bboxes = np.zeros((1, 8)) # also include object id

    bbox_mask = np.in1d(aligned_instance_bboxes[:,-2], OBJ_CLASS_IDS)
    object_id_to_label_filter = []
    selected_object_pcd = []

    for i in range(bbox_mask.shape[0]):
        if bbox_mask[i]:
            if object_id_to_label[i] in EXCLUDE_LABELS:
                bbox_mask[i] = False
            else:
                object_id_to_label_filter.append(object_id_to_label[i])
                selected_object_pcd.append(object_pcd[i])
    selected_object_label = object_id_to_label_filter # ["table", ...], only include objects whose nyu40id is in OBJ_CLASS_IDS
    object_ids = np.array((range(len(object_id_to_label))))[bbox_mask] # [0, 2, 3, ...], object_id of selected_object_label in raw *.aggregation.json(0-indexed)
    instance_bboxes = aligned_instance_bboxes[bbox_mask,:]   # [[x,y,z,l,w,h]]
    instance_rgb = aligned_instance_rgb[bbox_mask,:] # [[r,g,b]]
    return selected_object_pcd, selected_object_label, object_ids, instance_bboxes, instance_rgb
