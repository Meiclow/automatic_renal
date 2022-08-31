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
        self.kidney_num = kidney_num
        self.tumor_num = tumor_num
        self.none_num = 0
        self.spacing = self.image.GetSpacing()
        self.tumor_count = np.count_nonzero(self.matrix == self.tumor_num)
        self.kidney_count = np.count_nonzero(self.matrix == self.kidney_num)
        self.tumor_radius = None

    def print_renal_scores(self):
        r = self.get_radius()
        e = self.get_exophyticness()

        # TODO
        # n = self.get_nearness()
        n = 5

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

        print("Renal scores values:\n"
              "Radius: " + str(round(r, 2)) + "\n"
              "Exophyticness: " + str(round(e, 2)) + "\n"
              "Nearness: " + str(round(n, 2)) + "\n"
              "Anterior/Posterior: " + a + "\n"
              "Location: " + str(round(l, 2)) + "\n")

        print("Renal scores:\n"
              "Radius: " + str(r_score) + "\n"
              "Exophyticness: " + str(e_score) + "\n"
              "Nearness: " + str(n_score) + "\n"
              "Anterior/Posterior: " + a + "\n"
              "Location: " + str(l_score) + "\n")

        print("Renal acronym: " + acronym)

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

    def get_exophyticness(self):
        inside = 0

        for x in range(self.shape[2]):
            for y in range(self.shape[1]):
                for z in range(self.shape[0]):
                    if self.matrix[z][y][x] == self.tumor_num and self._inside_kidney_periphery(x, y, z):
                        inside += 1

        return inside / self.tumor_count

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

    def get_nearness(self):
        """
        Get nearness to renal sinus or collecting system.

        :return:
        """

        kidney_points = np.argwhere(self.matrix == self.kidney_num)
        print(kidney_points)
        hull = ConvexHull(kidney_points)

        all_points = self._get_boundary_cube(kidney_points)

        convex_hull_voxels = []
        # for points_chunk in np.array_split(all_points, len(all_points) // 1024):
        #     points_chunk_with_ones = np.concatenate((points_chunk, np.ones((len(points_chunk), 1))), axis=1)
        #     ind = np.argwhere(np.amax(points_chunk_with_ones @ hull.equations.T, axis=1) <= 0).reshape(-1)
        #     print(ind.shape)
        #     convex_hull_voxels.append(all_points[ind])
        #
        # convex_hull_voxels = np.concatenate(convex_hull_voxels, axis=0).reshape(-1, 3)

        all_points_with_ones = np.concatenate([all_points, np.ones((len(all_points), 1))], axis=1)
        for i in range(len(all_points)):
            if np.amax(all_points_with_ones[i] @ hull.equations.T) <= 0:
                convex_hull_voxels.append(all_points[i])

        diff_points = set(tuple(p) for p in convex_hull_voxels) - set(tuple(p) for p in kidney_points)

        print("Kidney volume: ", len(kidney_points))
        print("Convex hull volume: ", len(convex_hull_voxels))
        print("Difference volume: ", len(diff_points))

        blank_matrix = np.zeros(self.matrix.shape)
        for p in diff_points:
            blank_matrix[p] = 1

        visited_matrix = np.zeros(self.matrix.shape)
        ssc_sizes = defaultdict(int)

        i = 1
        st = []
        for p in diff_points:
            if visited_matrix[p] == 0:
                st.append(p)
                while len(st) > 0:
                    u = st.pop()
                    if visited_matrix[u] == 0:
                        ssc_sizes[p] += 1
                        visited_matrix[u] = i
                        i += 1
                        for d in [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]:
                            v = tuple(np.array(u) + np.array(d))
                            if 0 <= v[0] < blank_matrix.shape[0] and \
                                    0 <= v[1] < blank_matrix.shape[1] and \
                                    0 <= v[2] < blank_matrix.shape[2] and \
                                    blank_matrix[v] == 1 and visited_matrix[v] == 0:
                                st.append(v)

        largest_ssc_repr = sorted(list(ssc_sizes.keys()), key=lambda k: ssc_sizes[k], reverse=True)[0]
        print("Largest_diff_size = ", ssc_sizes[largest_ssc_repr])

        ssc_points = np.argwhere(visited_matrix == visited_matrix[largest_ssc_repr])
        tumor_points = np.argwhere(self.matrix == self.tumor_num)

        ssc_center_of_mass = ssc_points.sum(axis=0) // self.kidney_count
        tumor_center_of_mass = tumor_points.sum(axis=0) // self.tumor_count

        closest_ssc_point = sorted(
            ssc_points,
            key=lambda p: np.linalg.norm(p-tumor_center_of_mass),
        )[0]

        closest_tumor_point = sorted(
            tumor_points,
            key=lambda p: np.linalg.norm(p-ssc_center_of_mass),
        )[0]

        avg_spacing = np.sqrt(self.spacing[0] * self.spacing[1])

        distance = np.linalg.norm(closest_tumor_point - closest_ssc_point) * avg_spacing
        print("Distance: ", distance)
        if distance < 4:
            return 3
        elif distance < 7:
            return 2
        else:
            return 1

    def get_anterior(self, version="center_of_mass"):
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
        for i in range(len(kidney_counts)-1, -1, -1):
            if kidney_counts[i] != 0:
                kidney_stop = i
                break

        if kidney_start == -1 or kidney_stop == -1:
            raise Exception("You have no kidney! Seek help!")

        kidney_middle = (kidney_start + kidney_stop*2) // 3

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
        for i in range(polar_line_down, polar_line_up+1):
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

        horizontal_dist = np.sqrt((kidney_center[0] - tumor_center[0])**2 + (kidney_center[1] - tumor_center[1])**2)
        horizontal_dist *= avg_spacing
        tumor_radius = self.get_radius()
        c_index = horizontal_dist/tumor_radius

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
