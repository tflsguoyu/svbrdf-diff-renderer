# -*- coding: utf-8 -*-
#
# Copyright (c) 2023, Yu Guo. All rights reserved.

import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pupil_apriltags import Detector  # https://github.com/pupil-labs/apriltags
from .imageio import imread, imwrite, img9to1

np.set_printoptions(precision=4, suppress=True)


class Capture:
    def __init__(self, folder):
        self.detector = Detector()

        raw_dir = folder / "raw"
        im_paths = sorted(list(raw_dir.glob("[!.]*.*")))
        self.H, self.W = imread(im_paths[0]).shape[:2]

        self.ims = []
        for im_path in im_paths:
            self.ims.append(imread(im_path))

        self.n_of_imgs = len(self.ims)
        self.full_res = 1600
        self.crop_res = 1024

        self.json_256_dir = folder / "optim_latent_256.json"
        self.json_512_dir = folder / "optim_pixel_256_to_512.json"
        self.json_1024_dir = folder / "optim_pixel_512_to_1024.json"
        self.save_to_relative = "target"
        self.save_to = folder / self.save_to_relative

    def eval(self, size, depth, fisheye=True):
        point2d_list = self.point2d(debug=False)
        point3d_list = self.point3d(size=size, debug=False)
        calibs = self.calibrate(point3d_list, point2d_list)
        if fisheye:
            self.undistort(calibs[0], calibs[1])
            point2d_list = self.point2d(debug=False)
            calibs = self.calibrate(point3d_list, point2d_list)
        crops, fulls = self.rectify(point3d_list, point2d_list, calibs, size=size, d=depth, debug=False)
        camera_pos = self.get_camera_pos(calibs[2], calibs[3])
        self.save(crops, fulls, camera_pos, size)

    def save(self, ims, tmps, camera_pos, size, debug=False):
        self.save_to.mkdir(parents=True, exist_ok=True)
        for i in range(self.n_of_imgs):
            imwrite(ims[i], self.save_to / f"{i:02d}.png", dim=(self.crop_res, self.crop_res))
        if self.n_of_imgs == 9:
            img9to1(self.save_to)

        if debug:
            tmp_dir = self.save_to / "tmp"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            for i in range(self.n_of_imgs):
                imwrite(tmps[i], tmp_dir / f"{i:02d}.png")

        data = {
            "_comment": "in cm uint",
            "target_dir": self.save_to_relative,
            "optimize_dir": "optim_latent/256",
            "rerender_dir": "optim_latent/256",
            "im_size": size / self.full_res * self.crop_res,
            "idx": list(range(self.n_of_imgs)),
            "camera_pos": camera_pos.tolist(),
            "light_pos": camera_pos.tolist(),
            "light_pow": [1500, 1500, 1500]
        }
        with open(self.json_256_dir, "w") as f:
            json.dump(data, f, indent=4)

        data = {
            "_comment": "in cm uint",
            "reference_dir": "optim_latent/256",
            "target_dir": self.save_to_relative,
            "optimize_dir": "optim_latent/512",
            "rerender_dir": "optim_latent/512",
            "im_size": size / self.full_res * self.crop_res,
            "idx": list(range(self.n_of_imgs)),
            "camera_pos": camera_pos.tolist(),
            "light_pos": camera_pos.tolist(),
            "light_pow": [1500, 1500, 1500]
        }
        with open(self.json_512_dir, "w") as f:
            json.dump(data, f, indent=4)

        data = {
            "_comment": "in cm uint",
            "reference_dir": "optim_latent/512",
            "target_dir": self.save_to_relative,
            "optimize_dir": "optim_latent/1024",
            "rerender_dir": "optim_latent/1024",
            "im_size": size / self.full_res * self.crop_res,
            "idx": list(range(self.n_of_imgs)),
            "camera_pos": camera_pos.tolist(),
            "light_pos": camera_pos.tolist(),
            "light_pow": [1500, 1500, 1500]
        }
        with open(self.json_1024_dir, "w") as f:
            json.dump(data, f, indent=4)

    def rectify(self, point3d_list, point2d_list, calibs, size, d=0, debug=False):
        mtx, dist, rvecs, tvecs = calibs
        ims_full = []
        ims_crop = []
        for i in range(self.n_of_imgs):
            im = self.ims[i]
            point3d = point3d_list[i]
            # material plane is lower than markers plane
            # Warning: point3d[:, 2] -= d is incorrect here
            point3d[:, 2] = -d
            point2d = point2d_list[i]

            src_points, _ = cv2.projectPoints(
                point3d, rvecs[i], tvecs[i], mtx, dist)
            src_points = np.squeeze(src_points)

            if debug:
                plt.imshow(im)
                self.plot_marker(None, point2d)
                self.plot_marker(None, src_points)
                plt.show()

            dst_points = (point3d[:, :2] / size + 0.5) * self.full_res
            dst_points[:, 1] = self.full_res - dst_points[:, 1]

            homo_mat, _ = cv2.findHomography(src_points, dst_points)
            im_full = cv2.warpPerspective(im, homo_mat, (self.full_res, self.full_res))
            ims_full.append(im_full)

            tmp = int((self.full_res - self.crop_res) / 2)
            im_crop = im_full[tmp:tmp+self.crop_res, tmp:tmp+self.crop_res, :]
            ims_crop.append(im_crop)

        print("[DONE:Capture] Warp image")
        return ims_crop, ims_full

    def get_camera_pos(self, rvecs, tvecs):
        camera_pos = []
        for i in range(self.n_of_imgs):
            rmat, _ = cv2.Rodrigues(rvecs[i])
            rmat_inv = np.linalg.inv(rmat)
            tvec = -tvecs[i]
            camera_pos.append(np.matmul(rmat_inv, tvec))

        return np.hstack(camera_pos).transpose()

    def undistort(self, mtx, dist):
        ims_undistort = []
        for i in range(self.n_of_imgs):
            ims_undistort.append(cv2.undistort(self.ims[i], mtx, dist))
        self.ims = ims_undistort

    def calibrate(self, point3d_list, point2d_list):
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                            point3d_list, point2d_list,
                            (self.W, self.H), None, None)
        print("[DONE:Capture] Calibration")
        return mtx, dist, rvecs, tvecs

    def point2d(self, debug=False):
        corners_list = []
        ims_list = [] 
        for im in self.ims:
            corners = self.detect_2dmarker(im, debug)
            if corners.shape[0] != 64:
                continue
            ims_list.append(im)
            corners_list.append(corners)
        self.ims = ims_list
        self.n_of_imgs = len(self.ims)
        return corners_list

    def point3d(self, size, debug=False):
        corners = self.generate_3dmarker(size=size, debug=debug)
        return [corners] * self.n_of_imgs

    def detect_2dmarker(self, im, debug=False):
        im = (np.mean(im, axis=2) * 255).astype("uint8")
        results = self.detector.detect(im)

        tag_id = []
        centers = []
        corners = []
        for marker in results:
            if marker.tag_id >= 0 and marker.tag_id <= 15:  # we use markers tag36h11 from 0 to 15
                tag_id.append(marker.tag_id)
                centers.append(marker.center)
                corners.append(marker.corners)

        centers = np.vstack(centers).astype('float32')
        corners = np.vstack(corners).astype('float32')

        if debug:
            plt.imshow(im, cmap='gray', vmin=0, vmax=255)
            self.plot_marker(centers, corners)
            plt.show()

        return corners

    def generate_3dmarker(self, size=18, n=5, debug=False):
        tmp = np.arange(0, n) * 4
        tmp -= tmp[int((n-1)/2)]
        x, y = np.meshgrid(tmp, tmp, indexing='xy')
        centers = np.stack((x, -y, np.zeros_like(x)), 2).reshape(n*n, 3)
        idx = np.arange(0, n)
        idx = np.append(idx, np.arange(n*2-1, n*(n-1), n))
        idx = np.append(idx, np.arange(n*n-1, n*(n-1)-1, -1))
        idx = np.append(idx, np.arange(n*(n-2), 0, -n))
        centers = centers[idx, :]
        corners = []
        for i in range(centers.shape[0]):
            corners.append(centers[i, :] + np.array([-1, -1, 0]))
            corners.append(centers[i, :] + np.array([ 1, -1, 0]))
            corners.append(centers[i, :] + np.array([ 1,  1, 0]))
            corners.append(centers[i, :] + np.array([-1,  1, 0]))

        corners = np.vstack(corners).astype('float32')
        corners = corners * size / (4*n-2)
        centers = centers * size / (4*n-2)

        if debug:
            self.plot_marker(centers, corners)
            plt.show()

        return corners

    def plot_marker(self, centers, corners):
        for i in range(int(corners.shape[0]/4)):
            if centers is not None:
                plt.plot(centers[i, 0], centers[i, 1], 'm.')
            a = corners[i*4+0, :2]
            b = corners[i*4+1, :2]
            c = corners[i*4+2, :2]
            d = corners[i*4+3, :2]
            plt.plot([a[0], b[0]], [a[1], b[1]], 'r')
            plt.plot([b[0], c[0]], [b[1], c[1]], 'g')
            plt.plot([c[0], d[0]], [c[1], d[1]], 'b')
            plt.plot([d[0], a[0]], [d[1], a[1]], 'y')
            plt.plot(a[0], a[1], 'r.')
            plt.plot(b[0], b[1], 'g.')
            plt.plot(c[0], c[1], 'b.')
            plt.plot(d[0], d[1], 'y.')
        plt.axis('equal')
