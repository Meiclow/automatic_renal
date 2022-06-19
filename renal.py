import SimpleITK as sitk
import numpy as np
from scipy import ndimage


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
        self.tumor_count = np.count_nonzero(self.matrix == self.tumor_num)
        self.tumor_radius = None

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

    def get_nearness(self):
        """
        Get nearness to renal sinus or collecting system.

        :return:
        """

        # hull = convex_hull(kidney)
        # diff = hull - kidney
        # distance = shortest_dist(diff, tumor)
        # return distance

        pass

    def get_anterior(self):
        """
        Get anterior/posterior location i.e. whether the tumor is in front or back of the kidney.
        :return:
        """

        kidney_indexes = np.argwhere(self.matrix == self.kidney_num)
        kidney_center_of_mass = kidney_indexes.sum(axis=0) // len(kidney_indexes)

        front_matrix = self.matrix[:, :, :kidney_center_of_mass[2]]
        fraction_of_tumor_in_front = front_matrix.where(front_matrix == self.tumor_num).size() / front_matrix.size()

        return "Anterior" if fraction_of_tumor_in_front > 0.5 else "Posterior"

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

        max_kidney_slice = -1
        max_tumor_slice = -1

        max_kidney_val = 1
        max_tumor_val = 1

        for i in range(len(self.matrix)):
            matrix_slice = self.matrix[i]

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

        kidney_slice = (lambda x: (x == self.kidney_num) * 1)(np.array(self.matrix[max_kidney_slice]))
        tumor_slice = (lambda x: (x == self.tumor_num) * 1)(self.matrix[max_tumor_slice])

        kidney_center = ndimage.measurements.center_of_mass(kidney_slice)
        tumor_center = ndimage.measurements.center_of_mass(tumor_slice)

        horizontal_dist = np.sqrt((kidney_center[0] - tumor_center[0])**2 + (kidney_center[1] - tumor_center[1])**2)
        horizontal_dist *= avg_spacing
        tumor_radius = self.get_radius()
        c_index = horizontal_dist/tumor_radius

        return c_index




nii_path = 'data/KA53_20131213_nerka_guz_tetnice_ukm/D14F982C/guz nerka i tetnice/nerka i guz.nii.gz'
renal = Renal(nii_path, 2, 6)
print(renal.get_radius())
print(renal.get_exophyticness())
print(renal.get_location())
print(renal.get_anterior())
print(renal.get_c_index())
