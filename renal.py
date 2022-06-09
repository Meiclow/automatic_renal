import SimpleITK as sitk
import numpy as np


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

    def get_radius(self):
        avg_spacing = np.power(self.spacing[0] * self.spacing[1] * self.spacing[2], 1. / 3.)
        radius_pixels = np.power(0.75 * self.tumor_count / np.pi, 1. / 3.)
        radius = avg_spacing * radius_pixels

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


nii_path = 'data/KA53_20131213_nerka_guz_tetnice_ukm/D14F982C/guz nerka i tetnice/nerka i guz.nii.gz'
renal = Renal(nii_path, 2, 6)
print(renal.get_radius())
print(renal.get_exophyticness())
