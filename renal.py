import SimpleITK as sitk
import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull
from collections import defaultdict


class Renal:
    def __init__(self, image_path, kidney_num, tumor_num):
        self.image = sitk.ReadImage(image_path)
        self.matrix = np.array(sitk.GetArrayViewFromImage(self.image))
        self.shape = self.matrix.shape

        if type(kidney_num) == str:
            kidney_num = int(kidney_num.split(",")[-1])
        if type(tumor_num) == str:
            tumor_num = int(tumor_num.split(",")[-1])

        self.kidney_num = kidney_num
        self.tumor_num = tumor_num
        self.none_num = 0
        self.spacing = self.image.GetSpacing()
        self.tumor_count = np.count_nonzero(self.matrix == self.tumor_num)
        self.kidney_count = np.count_nonzero(self.matrix == self.kidney_num)
        if self.tumor_count == 0:
            raise Exception("Tumor segmentation not found", self.tumor_num, image_path)
        if self.kidney_count == 0:
            raise Exception("Kidney segmentation not found", self.kidney_num, image_path)
        self.tumor_radius = None

    def get_all_scores(self):
        r = self.get_radius()
        e = self.get_exophyticness()
        n = self.get_nearness()
        a = self.get_anterior()
        l = self.get_location()

        r_score = 1
        if r > 7:
            r_score = 3
        elif r > 4:
            r_score = 2

        e_score = 1
        if e > 0.99:
            e_score = 3
        elif e > 0.5:
            e_score = 2

        n_score = 1
        if n < 4:
            n_score = 3
        elif n < 7:
            n_score = 2

        l_score = 1
        if l > 0.5:
            l_score = 3
        elif l > 0.01:
            l_score = 2

        acronym = str(r_score + e_score + n_score + l_score) + a

        return {
            "Radius": round(r, 2),
            "Exophyticness": round(e, 2),
            "Nearness": round(n, 2),
            "Location": round(l, 2),
            "Radius_score": r_score,
            "Exophyticness_score": e_score,
            "Nearness_score": n_score,
            "Anterior/Posterior_score": a,
            "Location_score": l_score,
            "Acronym": acronym,
        }

    def print_c_index_score(self):
        c_index = self.get_c_index()

        print("C-index score: " + str(round(c_index, 2)))

    def get_radius(self):
        if self.tumor_radius:
            return self.tumor_radius

        avg_spacing = np.power(self.spacing[0] * self.spacing[1] * self.spacing[2], 1. / 3.)
        radius_pixels = np.power(0.75 * self.tumor_count / np.pi, 1. / 3.)
        radius = avg_spacing * radius_pixels

        self.tumor_radius = radius

        return radius

    def get_exophyticness(self, version="convex_hull"):
        if version == "ray_tracing":
            inside = 0

            for x in range(self.shape[2]):
                for y in range(self.shape[1]):
                    for z in range(self.shape[0]):
                        if self.matrix[z][y][x] == self.tumor_num and self._inside_kidney_periphery(x, y, z):
                            inside += 1
            return inside / self.tumor_count
        elif version == "convex_hull":
            kidney_voxels = np.argwhere(self.matrix == self.kidney_num)
            convex_hull_voxels = self._get_convex_hull_voxels(kidney_voxels)
            tumor_voxels = np.argwhere(self.matrix == self.tumor_num)
            overlap = set(tuple(p) for p in convex_hull_voxels) & set(tuple(p) for p in tumor_voxels)

            return len(overlap) / len(tumor_voxels)

    def _inside_kidney_periphery(self, x, y, z):

        faces_count = 0

        for x_it in range(x, self.shape[2]):
            content = self.matrix[z][y][x_it]
            # if content == self.none_num:
            #     break
            if content == self.kidney_num:
                faces_count += 1
                break

        for x_it in range(x, -1, -1):
            content = self.matrix[z][y][x_it]
            # if content == self.none_num:
            #     break
            if content == self.kidney_num:
                faces_count += 1
                break

        if faces_count > 1:
            return True

        for y_it in range(y, self.shape[1]):
            content = self.matrix[z][y_it][x]
            # if content == self.none_num:
            #     break
            if content == self.kidney_num:
                faces_count += 1
                break

        if faces_count > 1:
            return True

        for y_it in range(y, -1, -1):
            content = self.matrix[z][y_it][x]
            # if content == self.none_num:
            #     break
            if content == self.kidney_num:
                faces_count += 1
                break

        if faces_count > 1:
            return True

        for z_it in range(z, self.shape[0]):
            content = self.matrix[z_it][y][x]
            # if content == self.none_num:
            #     break
            if content == self.kidney_num:
                faces_count += 1
                break

        if faces_count > 1:
            return True

        for z_it in range(z, -1, -1):
            content = self.matrix[z_it][y][x]
            # if content == self.none_num:
            #     break
            if content == self.kidney_num:
                faces_count += 1
                break

        return faces_count > 1

    def _get_boundary_cube(self, points):
        return np.mgrid[
           np.amin(points[:, 0]): np.amax(points[:, 0]),
           np.amin(points[:, 1]): np.amax(points[:, 1]),
           np.amin(points[:, 2]): np.amax(points[:, 2])
        ].transpose(1, 2, 3, 0).reshape(-1, 3)

    def _approximate_closest_distance(self, points_a, points_b):
        a_center_of_mass = points_a.sum(axis=0) // len(points_a)
        b_center_of_mass = points_b.sum(axis=0) // len(points_b)

        closest_a_point = sorted(
            points_a,
            key=lambda p: np.linalg.norm(p - b_center_of_mass),
        )[0]

        closest_b_point = sorted(
            points_b,
            key=lambda p: np.linalg.norm(p - a_center_of_mass),
        )[0]

        closest_a_point = np.asarray(closest_a_point, dtype=np.float64) * self.spacing
        closest_b_point = np.asarray(closest_b_point, dtype=np.float64) * self.spacing

        distance = np.linalg.norm(closest_a_point - closest_b_point)
        return distance

    def _get_largest_region(self, points, matrix_shape):
        blank_matrix = np.zeros(matrix_shape)
        for p in points:
            blank_matrix[p] = 1

        visited_matrix = np.zeros(self.matrix.shape)
        ssc_sizes = defaultdict(int)

        directions = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
        i = 0
        st = []
        for p in points:
            if visited_matrix[p] == 0:
                st.append(p)
                i += 1
                while len(st) > 0:
                    u = st.pop()

                    # dfs body
                    if visited_matrix[u] == 0:
                        ssc_sizes[p] += 1
                        visited_matrix[u] = i
                        for d in directions:
                            v = tuple(np.array(u) + np.array(d))
                            if 0 <= v[0] < blank_matrix.shape[0] and \
                                    0 <= v[1] < blank_matrix.shape[1] and \
                                    0 <= v[2] < blank_matrix.shape[2] and \
                                    blank_matrix[v] == 1 and visited_matrix[v] == 0:
                                st.append(v)

        largest_ssc_repr = sorted(list(ssc_sizes.keys()), key=lambda k: ssc_sizes[k], reverse=True)[0]
        largest_ssc_points = np.argwhere(visited_matrix == visited_matrix[largest_ssc_repr])

        return largest_ssc_points

    def _get_convex_hull_voxels(self, points):
        potential_hull_voxels = self._get_boundary_cube(points)
        convex_hull_voxels = []

        # for points_chunk in np.array_split(potential_hull_voxels, len(potential_hull_voxels) // 1024):
        #     points_chunk_with_ones = np.concatenate((points_chunk, np.ones((len(points_chunk), 1))), axis=1)
        #     ind = np.argwhere(np.amax(points_chunk_with_ones @ hull.equations.T, axis=1) <= 0).reshape(-1)
        #     print(ind.shape)
        #     convex_hull_voxels.append(potential_hull_voxels[ind])
        #
        # convex_hull_voxels = np.concatenate(convex_hull_voxels, axis=0).reshape(-1, 3)

        hull = ConvexHull(points)
        potential_hull_voxels_with_ones = np.concatenate(
            [potential_hull_voxels, np.ones((len(potential_hull_voxels), 1))],
            axis=1
        )

        for i in range(len(potential_hull_voxels)):

            # check if the point is inside the hull
            if np.amax(potential_hull_voxels_with_ones[i] @ hull.equations.T) <= 0:
                convex_hull_voxels.append(potential_hull_voxels[i])

        return convex_hull_voxels

    def get_nearness(self):
        """
        Get nearness to renal sinus or collecting system.

        :return:
        """
        kidney_voxels = np.argwhere(self.matrix == self.kidney_num)
        print("Kidney volume: ", len(kidney_voxels))

        convex_hull_voxels = self._get_convex_hull_voxels(kidney_voxels)
        print("Convex hull volume: ", len(convex_hull_voxels))

        diff_voxels = set(tuple(p) for p in convex_hull_voxels) - set(tuple(p) for p in kidney_voxels)
        print("Difference volume: ", len(diff_voxels))

        ssc_voxels = self._get_largest_region(diff_voxels, self.matrix.shape)
        print("Largest diff region volume: ", len(ssc_voxels))

        tumor_voxels = np.argwhere(self.matrix == self.tumor_num)
        print("Tumor volume: ", len(tumor_voxels))

        distance = self._approximate_closest_distance(tumor_voxels, ssc_voxels)
        print("Distance: ", distance)

        return distance

    def get_anterior(self, version="largest_plane"):
        """
        Get anterior/posterior location i.e. whether the tumor is in front or back of the kidney.
        :return:
        """

        y_cutoff = None

        if version == "center_of_mass":
            kidney_indexes = np.argwhere(self.matrix == self.kidney_num)
            kidney_center_of_mass = kidney_indexes.sum(axis=0) // len(kidney_indexes)
            y_cutoff = kidney_center_of_mass[1]
        elif version == "largest_plane":
            largest_plane_area = 0
            for y in range(self.shape[1]):
                area = np.count_nonzero(self.matrix[:, y, :] == self.kidney_num)
                if area > largest_plane_area:
                    y_cutoff = y
                    largest_plane_area = area
        else:
            raise ValueError("Invalid version parameter. Use either 'center_of_mass' or 'largest_plane'")

        front_matrix = self.matrix[:, :y_cutoff, :]
        fraction_of_tumor_in_front = np.count_nonzero(front_matrix == self.tumor_num) / front_matrix.size

        # print("Fraction of tumor in front: ", fraction_of_tumor_in_front)

        return "A" if fraction_of_tumor_in_front > 0.5 else "P"

    def get_location(self):
        kidney_counts = []

        for matrix_slice in self.matrix:
            kidney_counts.append(np.count_nonzero(matrix_slice == self.kidney_num))

        kidney_start = -1
        kidney_stop = -1
        for i in range(len(kidney_counts)):
            if kidney_counts[i] != 0:
                kidney_start = i
                break
        for i in range(len(kidney_counts) - 1, -1, -1):
            if kidney_counts[i] != 0:
                kidney_stop = i
                break

        if kidney_start == -1 or kidney_stop == -1:
            raise Exception("You have no kidney! Seek help!")

        kidney_middle = (kidney_start + kidney_stop * 2) // 3

        max_down = -1
        polar_line_down = -1
        for i in range(kidney_start, kidney_middle):
            if kidney_counts[i] > max_down:
                max_down = kidney_counts[i]
                polar_line_down = i

        max_up = -1
        polar_line_up = -1
        for i in range(kidney_middle, kidney_stop):
            if kidney_counts[i] > max_up:
                max_up = kidney_counts[i]
                polar_line_up = i

        between_polar_lines_tumor_count = 0
        for i in range(polar_line_down, polar_line_up + 1):
            between_polar_lines_tumor_count += np.count_nonzero(self.matrix[i] == self.tumor_num)

        return between_polar_lines_tumor_count / self.tumor_count

    def get_c_index(self):

        front_matrix = np.copy(self.matrix)
        front_matrix = np.swapaxes(front_matrix, 0, 1)

        max_kidney_slice = -1
        max_tumor_slice = -1

        max_kidney_val = 1
        max_tumor_val = 1

        for i in range(len(front_matrix)):
            matrix_slice = front_matrix[i]

            kidney_val = np.count_nonzero(matrix_slice == self.kidney_num)
            if kidney_val > max_kidney_val:
                max_kidney_slice = i
                max_kidney_val = kidney_val

            tumor_val = np.count_nonzero(matrix_slice == self.tumor_num)
            if tumor_val > max_tumor_val:
                max_tumor_slice = i
                max_tumor_val = tumor_val

        if max_kidney_slice == -1:
            raise Exception("No kidney found")
        if max_tumor_slice == -1:
            raise Exception("No tumor found")

        avg_spacing = np.sqrt(self.spacing[0] * self.spacing[1])

        kidney_slice = (lambda x: (x == self.kidney_num) * 1)(np.array(front_matrix[max_kidney_slice]))
        tumor_slice = (lambda x: (x == self.tumor_num) * 1)(front_matrix[max_tumor_slice])

        kidney_center = ndimage.measurements.center_of_mass(kidney_slice)
        tumor_center = ndimage.measurements.center_of_mass(tumor_slice)

        horizontal_dist = np.sqrt((kidney_center[0] - tumor_center[0]) ** 2 + (kidney_center[1] - tumor_center[1]) ** 2)
        horizontal_dist *= avg_spacing
        tumor_radius = self.get_radius()
        c_index = horizontal_dist / tumor_radius

        return c_index


nii_path = 'data/KA53_20131213_nerka_guz_tetnice_ukm/D14F982C/guz nerka i tetnice/nerka i guz.nii.gz'
renal = Renal(nii_path, 2, 6)
# print(renal.get_radius())
# print(renal.get_exophyticness())
# print(renal.get_anterior())
# print(renal.get_location())
# print(renal.get_c_index())

renal.print_c_index_score()
renal.print_renal_scores()
