import numpy as np
from enum import Enum, unique
from shapely.geometry import MultiPoint, Point
from collections import defaultdict
import matplotlib.pyplot as plt

@unique
class OrientedSections(Enum):
    front = 0
    right = 1
    back = 2
    left = 3
    grey_area = 4

class HorizontalProximityHelper():
    def __init__(self, horizontal_gap=0.2, touch_dist=1, around_dist=1, enable_gap=True):
        '''closest, farthest, within reach, around
        @param horizontal_gap (float): the gap between the closest and second-closest objects
        @param enable_gap (bool): whether to enable the epsilon gap. E.g., if enabled, the distance between the closest and second-closest objects needs to be larger than the epsilon gap, to be considered to have a closest relation.
        '''
        self.horizontal_gap = horizontal_gap
        self.touch_dist = touch_dist
        self.around_dist = around_dist
        self.enable_gap = enable_gap

    def infer_close_far(self, tgt_objects, acr_object):
        ''' inputs target_object and anchor_object, outputs their horizontal relation
        @param tgt_objects (list[ThreeDObject]): target objects
        @param acr_object (ThreeDObject): anchor object
        @return (dict[int]): {'closest': closest_object_id, 'farthest': farthest_object_id}
        '''
        epsilon_gap = self.horizontal_gap
        enable_gap = self.enable_gap

        all_dists = []
        for target in tgt_objects:
            t_distance = acr_object.distance_from_other_object(target)
            all_dists.append(t_distance)
        
        s_idx = np.argsort(all_dists)
        all_dists = np.array(all_dists)[s_idx]

        if enable_gap:
            farthest_object_id = -1
            closest_object_id = -1
            if all_dists[0] + epsilon_gap < all_dists[1]:  # "closest" ref-type
                closest_object_id = tgt_objects[s_idx[0]].object_id
            if all_dists[-1] - epsilon_gap > all_dists[-2]:
                farthest_object_id = tgt_objects[s_idx[-1]].object_id
        else:
            closest_object_id = tgt_objects[s_idx[0]].object_id
            farthest_object_id = tgt_objects[s_idx[-1]].object_id
        return {'closest': closest_object_id, 'farthest': farthest_object_id}
    
    def infer_touch_around(self, tgt_object, acr_object):
        '''inputs target_object and anchor_object, outputs their horizontal relation
        @return (list[str]): a list of 'within reach' and 'around'
        '''
        touch_dist = self.touch_dist
        around_dist = self.around_dist
        output_rel = []
        dist = acr_object.distance_from_other_object(tgt_object)
        if dist < touch_dist:
            output_rel.append('within reach')
        if dist < around_dist:
            output_rel.append('around')
        return output_rel

class VerticalProximityHelper():
    def __init__(self, max_touch_distance = 0.15,
        min_above_below_distance = 0.02, # 0.06,
        max_supporting_area_ratio = 1.5,
        min_supported_area_ratio = 0.5, # 0.3
        min_to_be_above_below_area_ratio = 0.2):
        '''on, above, below
        @param max_touch_distance (float): the maximum distance between the target and anchor objects to be considered as touching
        @param min_above_below_distance (float): the minimum distance between the target and anchor objects to be considered as above/below
        @param max_supporting_area_ratio (float): the maximum ratio of the target object's area to the anchor object's area to be considered as supporting
        @param min_supported_area_ratio (float): the minimum ratio of the target object's area to the anchor object's area to be considered as supported
        @param min_to_be_above_below_area_ratio (float): the minimum ratio of the target object's area to the anchor object's area to be considered as above/below
        '''
        self.max_touch_distance = max_touch_distance
        self.min_above_below_distance = min_above_below_distance
        self.max_supporting_area_ratio = max_supporting_area_ratio
        self.min_supported_area_ratio = min_supported_area_ratio
        self.min_to_be_above_below_area_ratio = min_to_be_above_below_area_ratio


    def infer(self, tgt_object, acr_object):
        ''' inputs target_object and anchor_object, outputs their vertical relation
        @param tgt_object (ThreeDObject): target object
        @param acr_object (ThreeDObject): anchor object
        @return (list[str]): a list of vertical relation including 'on', 'above', 'below', 'under
        '''

        vertical_relations = []

        max_to_be_touching_distance = self.max_touch_distance
        min_above_below_distance = self.min_above_below_distance
        max_to_be_supporting_area_ratio = self.max_supporting_area_ratio
        min_to_be_supported_area_ratio = self.min_supported_area_ratio
        min_to_be_above_below_area_ratio = self.min_to_be_above_below_area_ratio

        a_zmin, a_zmax = acr_object.extrema[2], acr_object.extrema[5]
        t_zmin, t_zmax = tgt_object.extrema[2], tgt_object.extrema[5]

        iou_2d, i_ratios, a_ratios = tgt_object.iou_2d(acr_object)
        i_target_ratio, i_anchor_ratio = i_ratios
        target_anchor_area_ratio, anchor_target_area_ratio = a_ratios
        
        if iou_2d < 0.001:  # No intersection at all (not in the vicinty of each other)
            return "" # vertical_relations
        
        target_bottom_anchor_top_dist = t_zmin - a_zmax
        target_top_anchor_bottom_dist = a_zmin - t_zmax

        if i_target_ratio > min_to_be_supported_area_ratio and \
                abs(target_bottom_anchor_top_dist) <= max_to_be_touching_distance and \
                target_anchor_area_ratio < max_to_be_supporting_area_ratio:  # target is not quite larger in area than the anchor
            vertical_relations.append('on')
        
        if target_bottom_anchor_top_dist > min_above_below_distance and \
            max(i_anchor_ratio, i_target_ratio) > min_to_be_above_below_area_ratio:  # above
            vertical_relations.append('above')
        
        if target_top_anchor_bottom_dist > min_above_below_distance and \
            max(i_anchor_ratio, i_target_ratio) > min_to_be_above_below_area_ratio:  # below
            vertical_relations.extend(['below', 'under'])

        if len(vertical_relations) == 0:
            if tgt_object.bbox[2] > acr_object.bbox[2]:
                vertical_relations.append('on')
            else:
                vertical_relations.extend(['below','under'])

        return vertical_relations

class BetweenHelper():
    def __init__(self,
        min_target_to_anchor_z_intersection = 0.2,
        min_forbidden_occ_ratio = 0.1,
        target_anchor_intersect_ratio_thresh = 0.1,
        occ_thresh = 0.51):
        '''between
        @param min_target_to_anchor_z_intersection (float): the minimum z intersection between the target and anchor objects to be considered as between
        @param min_forbidden_occ_ratio (float): the minimum ratio of the target object's area to the anchor object's area to be considered as between
        @param target_anchor_intersect_ratio_thresh (float): the minimum ratio of the target object's area to the anchor object's area to be considered as between
        @param occ_thresh (float): the minimum ratio of the target object's area to the anchor object's area to be considered as between
        '''
        self.min_target_to_anchor_z_intersection = min_target_to_anchor_z_intersection
        self.min_forbidden_occ_ratio = min_forbidden_occ_ratio
        self.target_anchor_intersect_ratio_thresh = target_anchor_intersect_ratio_thresh
        self.occ_thresh = occ_thresh
    
    def infer(self, tgt_object, acr_object_a, acr_object_b):
        ''' inputs target_object and anchor_object, outputs their vertical relation
        @param tgt_object (ThreeDObject): target object
        @param acr_object_a (ThreeDObject): anchor object a
        @param acr_object_b (ThreeDObject): anchor object b
        @return (bool): whether the target object is between the two anchor objects
        '''
        occ_thresh = self.occ_thresh
        min_forbidden_occ_ratio = self.min_forbidden_occ_ratio
        target_anchor_intersect_ratio_thresh = self.target_anchor_intersect_ratio_thresh
        min_target_to_anchor_z_intersection = self.min_target_to_anchor_z_intersection

        anchor_a_z_face = acr_object_a._z_face()
        anchor_a_points = tuple(map(tuple, anchor_a_z_face[:, :2]))

        anchor_b_z_face = acr_object_b._z_face()
        anchor_b_points = tuple(map(tuple, anchor_b_z_face[:, :2]))

        tgt_object_z_face = tgt_object._z_face()
        target_points = tuple(map(tuple, tgt_object_z_face[:, :2]))

        # Check whether a target object lies in the convex hull of the two anchors.
        forbidden_occ_range = [min_forbidden_occ_ratio, occ_thresh - 0.001]
        intersect_ratio_thresh = target_anchor_intersect_ratio_thresh

        convex_hull = MultiPoint(anchor_a_points + anchor_b_points).convex_hull
        polygon_a = MultiPoint(anchor_a_points).convex_hull
        polygon_b = MultiPoint(anchor_b_points).convex_hull
        polygon_t = MultiPoint(target_points).convex_hull

        # Candidate should fall completely/with a certain ratio in the convex_hull polygon
        occ_ratio = convex_hull.intersection(polygon_t).area / polygon_t.area
        if occ_ratio < occ_thresh: 
            return False
        # Candidate target should never be intersecting any of the anchors
        if polygon_t.intersection(polygon_a).area / polygon_t.area > intersect_ratio_thresh:
            return False
        if polygon_t.intersection(polygon_b).area / polygon_t.area > intersect_ratio_thresh:
            return False
        
        # Target should be in the same z range for each of the two anchors
        _, t_anc_a, _ = tgt_object.intersection(acr_object_a)
        _, t_anc_b, _ = tgt_object.intersection(acr_object_b)
        if t_anc_a < min_target_to_anchor_z_intersection or \
                t_anc_b < min_target_to_anchor_z_intersection:
            return False
        
        return True


class AllocentricHelper():
    def __init__(self,
        angle = 5,# 5, # 0.05, # 10
		max_dl = 0.01, # 1
		max_df = 0.01, # 1
		d2 = 20, # 20, # 4
        edge = 0.1, # 0.01,    # None
		positive_occ_thresh = 0.51): # 0.8): # 0.9
        '''left, right, front, back.
        
        it first forms fours sections, then calculate the occupancy ratio of the target object in each section
        @param angle (float): the angle to form the section
        @param max_dl (float): the maximum length of the anchor object's left/right side
        @param max_df (float): the maximum length of the anchor object's front/back side
        @param d2 (float): the distance to extend the candidate section
        @param positive_occ_thresh (float): the minimum ratio of (# target in section pc)/(# target pc)
        '''
        self.angle = angle
        self.max_dl = max_dl
        self.max_df = max_df
        self.d2 = d2
        self.edge = edge
        self.positive_occ_thresh = positive_occ_thresh

    def infer(self, tgt_object, acr_object):
        ''' inputs target_object and anchor_object, outputs their vertical relation
        @param tgt_object (ThreeDObject): target object
        @param acr_object (ThreeDObject): anchor object
        # @return (str): allocentric relation including 'left', 'right', 'front', 'back', 'none'
        @return (list[str]): a list of allocentric relation including left', 'right', 'front', 'back'
        '''
        allocentric_relations = []
        max_df = self.max_df
        max_dl = self.max_dl
        a = self.angle
        d2 = self.d2
        positive_occ_thresh = self.positive_occ_thresh

        df = min(2 * acr_object.bbox[3], max_df)
        dl = min(2 * acr_object.bbox[4], max_dl)
        

        [xmin, ymin, _, xmax, ymax, _] = acr_object.extrema
        anchor_bbox_extrema = [xmin, xmax, ymin, ymax]
        anchor_sections = self.get_anchor_sections(anchor_bbox_extrema, a, dl, df, d2, self.edge)
        
        oriented_sections = defaultdict(int)
        
        target_points = tgt_object.point_clouds
        for point in target_points:
            section_names = self.which_section_point_in(acr_object.bbox, anchor_sections, point)
            for section_name in section_names:
                oriented_sections[section_name.value] += 1

        if OrientedSections.left.value in oriented_sections and not OrientedSections.right.value in oriented_sections:
            if self.occupy_main_area(oriented_sections, OrientedSections.left.value, tgt_object, positive_occ_thresh):
                allocentric_relations.append("left")
        elif OrientedSections.right.value in oriented_sections and not OrientedSections.left.value in oriented_sections:
            if self.occupy_main_area(oriented_sections, OrientedSections.right.value, tgt_object, positive_occ_thresh):
                allocentric_relations.append("right")
        elif OrientedSections.left.value in oriented_sections and OrientedSections.right.value in oriented_sections:
            if oriented_sections[OrientedSections.left.value]>oriented_sections[OrientedSections.right.value]:
                if self.occupy_main_area(oriented_sections, OrientedSections.left.value, tgt_object, positive_occ_thresh):
                    allocentric_relations.append("left")
            else:
                if self.occupy_main_area(oriented_sections, OrientedSections.right.value, tgt_object, positive_occ_thresh):
                    allocentric_relations.append("right")

        if OrientedSections.front.value in oriented_sections and not OrientedSections.back.value in oriented_sections:
            if self.occupy_main_area(oriented_sections, OrientedSections.front.value, tgt_object, positive_occ_thresh):
                allocentric_relations.append("front")
        elif OrientedSections.back.value in oriented_sections and not OrientedSections.front.value in oriented_sections:
            if self.occupy_main_area(oriented_sections, OrientedSections.back.value, tgt_object, positive_occ_thresh):
                allocentric_relations.extend(["back","behind"])
        elif OrientedSections.front.value in oriented_sections and OrientedSections.back.value in oriented_sections:
            if oriented_sections[OrientedSections.front.value]>oriented_sections[OrientedSections.back.value]:
                if self.occupy_main_area(oriented_sections, OrientedSections.front.value, tgt_object, positive_occ_thresh):
                    allocentric_relations.append("front")
            else:
                if self.occupy_main_area(oriented_sections, OrientedSections.back.value, tgt_object, positive_occ_thresh):
                    allocentric_relations.extend(["back","behind"])
        
        if len(allocentric_relations)==0 or len(oriented_sections)==1 and OrientedSections.grey_area.value in oriented_sections:
            acr_x, acr_y = acr_object.bbox[0], acr_object.bbox[1]
            tgt_x, tgt_y = tgt_object.bbox[0], tgt_object.bbox[1]
            if tgt_x < acr_x:
                allocentric_relations.append("left")
            else:
                allocentric_relations.append("right")
            if tgt_y > acr_y:
                allocentric_relations.append("front")
            else:
                allocentric_relations.extend(["back","behind"])
        
        return allocentric_relations
        
    @staticmethod
    def occupy_main_area(oriented_sections, section_value, tgt_object, positive_occ_thresh):
        n_points = oriented_sections[section_value]
        target_occupancy_ratios = n_points / len(tgt_object.point_clouds)
        if target_occupancy_ratios > positive_occ_thresh:
            return True
        return False

    @staticmethod
    def get_anchor_sections(extrema, a, dl, df, d2, edge=0.01, plot=False):
        """

        @param extrema:
        @param a:
        @param dl:
        @param df:
        @param d2:

        @return:
        """
        xmin, xmax, ymin, ymax = extrema
        if edge:
            xmin, xmax, ymin, ymax = -edge, edge, -edge, edge
        b = 90 - a
        a = np.deg2rad(a)
        b = np.deg2rad(b)

        section_names = [OrientedSections.front, OrientedSections.back, OrientedSections.right, OrientedSections.left]
        ret = {}
        
        diff = d2/2
        df = diff / np.tan(b)
        dl = df
        for section in section_names:
            if section.name == 'left': # front
                p1 = (xmin, ymin)
                p2 = (xmin, ymax)
                p3 = (xmin - df, ymax + diff)
                p4 = (xmin - df, ymin - diff)
                p5 = (xmin - df - d2, ymin - diff)
                p6 = (xmin - df - d2, ymax + diff)
                ret[section] = MultiPoint([p1, p2, p3, p4, p5, p6]).convex_hull
            elif section.name == 'right': # back
                p1 = (xmax, ymin)
                p2 = (xmax, ymax)
                p3 = (xmax + df, ymax + diff)
                p4 = (xmax + df, ymin - diff)
                p5 = (xmax + df + d2, ymin - diff)
                p6 = (xmax + df + d2, ymax + diff)
                ret[section] = MultiPoint([p1, p2, p3, p4, p5, p6]).convex_hull
            elif section.name == 'front': # left
                p1 = (xmin, ymax)
                p2 = (xmax, ymax)
                p3 = (xmax + diff, ymax + dl)
                p4 = (xmin - diff, ymax + dl)
                p6 = (xmin - diff, ymax + dl + d2)
                p5 = (xmax + diff, ymax + dl + d2)
                ret[section] = MultiPoint([p1, p2, p3, p4, p5, p6]).convex_hull
            elif section.name == 'back': # right
                p1 = (xmin, ymin)
                p2 = (xmax, ymin)
                p3 = (xmax + diff, ymin - dl)
                p4 = (xmin - diff, ymin - dl)
                p5 = (xmin - diff, ymin - dl - d2)
                p6 = (xmax + diff, ymin - dl - d2)
                ret[section] = MultiPoint([p1, p2, p3, p4, p5, p6]).convex_hull

            if plot:
                hull = ret[section]
                fig, ax = plt.subplots()
                ax.plot(*hull.exterior.xy)
                hull = MultiPoint([(xmin, ymin),(xmax, ymax),(xmin, ymax),(xmax, ymin)]).convex_hull
                ax.plot(*hull.exterior.xy)
                points = [p1,p2,p3,p4,p5,p6]
                for i,p in enumerate(points):
                    x,y = p
                    ax.scatter(x, y, label=f'P{i+1}')
                    ax.annotate(f'P{i+1}', (x, y))
                plt.axhline(0, color='black', linewidth=0.8)  # 水平坐标轴
                plt.axvline(0, color='black', linewidth=0.8)  # 垂直坐标轴
                ax.axis('equal')
                plt.savefig(f'{section}.png')
        return ret
    
    @staticmethod
    def which_section_point_in(anchor_bbox, anchor_sections, target_point):
        # Transform the point in order to be compared with the object's
        # axes aligned bbox
        [px, py, _] = target_point
        section_names = []
        for sec_name, section in anchor_sections.items():
            if section.contains(Point(px, py)):
                # return sec_name
                section_names.append(sec_name)

        # No section
        if len(section_names) == 0:
            section_names.append(OrientedSections.grey_area)
        return section_names